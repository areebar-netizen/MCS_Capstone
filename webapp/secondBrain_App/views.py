from uuid import uuid4

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.utils import timezone
from django.conf import settings
from django.core.cache import cache
import random
import string
import json
from pathlib import Path

from urllib3 import request
from .services.prediction_service import PredictionService
from .services.eeg_service import EEGService
from .tasks import run_live_inference, get_task_status
from .tasks_realtime import run_live_inference_streaming
from django.views.decorators.csrf import csrf_exempt
import csv

from .models import UserProfile, Recommendation, SessionSummary
from .services.eeg_service import EEGService

# Create your views here.

MODEL_SERVICE = PredictionService(models_dir=Path(settings.BASE_DIR.parent)/ 'core_engine' / 'artifacts', model_name='xgboost')

def calendar_view(request):
    """Dynamic calendar view driven by SessionSummary data"""
    user_email = request.session.get('user_email')
    if not user_email:
        return redirect('/')
    
    try:
        user_profile = UserProfile.objects.get(email=user_email)
    except UserProfile.DoesNotExist:
        return redirect('/onboarding/')
    
    # Get month and year from GET parameters or default to current date
    from datetime import datetime
    import calendar
    
    try:
        requested_month = int(request.GET.get('month', ''))
        requested_year = int(request.GET.get('year', ''))
        
        # Validate month and year ranges
        if requested_month < 1 or requested_month > 12:
            raise ValueError("Invalid month")
        if requested_year < 2020 or requested_year > 2030:
            raise ValueError("Invalid year")
            
        current_month = requested_month
        current_year = requested_year
    except (ValueError, TypeError):
        # Default to current date if parameters are invalid or missing
        now = timezone.localtime(timezone.now())  # Use local timezone
        current_month = now.month
        current_year = now.year
    
    # Query SessionSummary for current user and requested month/year
    # Note: Database times are UTC, so we need to get a range that covers the local month
    from datetime import datetime, timedelta
    
    # Convert local month/year to UTC range for query
    # Create the month boundaries in local timezone, then convert to UTC
    month_start_local = datetime(current_year, current_month, 1, 0, 0, 0)
    month_start = timezone.make_aware(month_start_local)
    
    if current_month == 12:
        month_end_local = datetime(current_year + 1, 1, 1, 0, 0, 0)
    else:
        month_end_local = datetime(current_year, current_month + 1, 1, 0, 0, 0)
    month_end = timezone.make_aware(month_end_local)
    
    sessions = SessionSummary.objects.filter(
        user=user_profile,
        start_time__gte=month_start,
        start_time__lt=month_end
    ).order_by('start_time')
    
    # Create session data mapping by day (using local timezone)
    session_data_by_day = {}
    for session in sessions:
        local_time = timezone.localtime(session.start_time)
        day = local_time.day
        
        # Calculate focus percentage from 2-10 scale to 0-100% for image mapping
        focus_percentage = (session.average_focus_score / 10) * 100
        
        # If multiple sessions on same day, keep the one with highest focus score
        if day not in session_data_by_day or session.average_focus_score > session_data_by_day[day]['average_focus_score']:
            session_data_by_day[day] = {
                'session_id': session.session_id,
                'start_time': session.start_time,
                'average_focus_score': session.average_focus_score,
                'focus_percentage': focus_percentage,
                'total_duration_seconds': session.total_duration_seconds,
                'concentrating_seconds': session.concentrating_seconds,
                'peak_focus_score': session.peak_focus_score
            }
    
    # Generate calendar weeks using calendar.monthcalendar(year, month)
    # Set Sunday as first day of week to match typical calendar display
    calendar.setfirstweekday(calendar.SUNDAY)
    cal = calendar.monthcalendar(current_year, current_month)
    calendar_weeks = []
    
    for week in cal:
        week_days = []
        for day_num in week:
            if day_num == 0:  # Day from previous/next month
                day_obj = {
                    'day_num': day_num,
                    'in_month': False,
                    'has_data': False,
                    'image_path': None,
                    'focus_pct': None
                }
            else:
                has_data = day_num in session_data_by_day
                if has_data:
                    session_data = session_data_by_day[day_num]
                    focus_pct = session_data['focus_percentage']
                    
                    # Map focus percentage to image path based on requirements
                    if focus_pct >= 75:
                        image_path = '/static/images/Master.jpg'
                    elif focus_pct >= 60:
                        image_path = '/static/images/LockedIn.jpg'
                    elif focus_pct >= 45:
                        image_path = '/static/images/Steady.jpg'
                    elif focus_pct >= 30:
                        image_path = '/static/images/Neutral.jpg'
                    elif focus_pct >= 15:
                        image_path = '/static/images/Distracted.jpg'
                    else:
                        image_path = '/static/images/BrainFog.jpg'
                else:
                    focus_pct = None
                    image_path = None
                
                day_obj = {
                    'day_num': day_num,
                    'in_month': True,
                    'has_data': has_data,
                    'image_path': image_path,
                    'focus_pct': focus_pct
                }
            
            week_days.append(day_obj)
        
        calendar_weeks.append(week_days)
    
    # Calculate monthly statistics
    sessions_count = sessions.count()
    avg_focus = 0
    if sessions_count > 0:
        avg_focus = sum(session.average_focus_score for session in sessions) / sessions_count
        avg_focus = avg_focus * 100  # Convert to percentage
    
    active_days = len(session_data_by_day)
    
    # Calculate navigation context
    if current_month == 1:
        prev_month = 12
        prev_year = current_year - 1
    else:
        prev_month = current_month - 1
        prev_year = current_year
    
    if current_month == 12:
        next_month = 1
        next_year = current_year + 1
    else:
        next_month = current_month + 1
        next_year = current_year
    
    # Get month name for display
    month_name = calendar.month_name[current_month]
    
    context = {
        'user': request.user if request.user.is_authenticated else None,
        'user_profile': user_profile,
        'calendar_weeks': calendar_weeks,
        'current_month': current_month,
        'current_year': current_year,
        'month_name': month_name,
        'sessions_this_month': sessions_count,
        'avg_focus_pct': avg_focus,
        'active_days': active_days,
        'prev_month': prev_month,
        'prev_year': prev_year,
        'next_month': next_month,
        'next_year': next_year,
        'focus_legend': [
            {'percentage': '75%+', 'image': '/static/images/Master.jpg', 'label': 'Master'},
            {'percentage': '60-74%', 'image': '/static/images/LockedIn.jpg', 'label': 'Locked In'},
            {'percentage': '45-59%', 'image': '/static/images/Steady.jpg', 'label': 'Steady'},
            {'percentage': '30-44%', 'image': '/static/images/Neutral.jpg', 'label': 'Neutral'},
            {'percentage': '15-29%', 'image': '/static/images/Distracted.jpg', 'label': 'Distracted'},
            {'percentage': '<15%', 'image': '/static/images/BrainFog.jpg', 'label': 'Brain Fog'}
        ]
    }
    
    return render(request, 'focus_calendar.html', context)

def calendar_api_data(request):
    """API endpoint for calendar data used in AJAX requests"""
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'User not authenticated'}, status=401)
    
    try:
        user_profile = UserProfile.objects.get(email=user_email)
    except UserProfile.DoesNotExist:
        return JsonResponse({'error': 'User profile not found'}, status=404)
    
    # Get month and year from GET parameters or default to current date
    from datetime import datetime
    import calendar
    
    try:
        requested_month = int(request.GET.get('month', ''))
        requested_year = int(request.GET.get('year', ''))
        
        # Validate month and year ranges
        if requested_month < 1 or requested_month > 12:
            raise ValueError("Invalid month")
        if requested_year < 2020 or requested_year > 2030:
            raise ValueError("Invalid year")
            
        current_month = requested_month
        current_year = requested_year
    except (ValueError, TypeError):
        # Default to current date if parameters are invalid or missing
        now = timezone.now()
        current_month = now.month
        current_year = now.year
    
    # Query SessionSummary for current user and requested month/year
    # Note: Database times are UTC, so we need to get a range that covers the local month
    from datetime import datetime, timedelta
    
    # Convert local month/year to UTC range for query
    # Create the month boundaries in local timezone, then convert to UTC
    month_start_local = datetime(current_year, current_month, 1, 0, 0, 0)
    month_start = timezone.make_aware(month_start_local)
    
    if current_month == 12:
        month_end_local = datetime(current_year + 1, 1, 1, 0, 0, 0)
    else:
        month_end_local = datetime(current_year, current_month + 1, 1, 0, 0, 0)
    month_end = timezone.make_aware(month_end_local)
    
    sessions = SessionSummary.objects.filter(
        user=user_profile,
        start_time__gte=month_start,
        start_time__lt=month_end
    ).order_by('start_time')
    
    # Create session data mapping by day (using local timezone)
    session_data_by_day = {}
    for session in sessions:
        local_time = timezone.localtime(session.start_time)
        day = local_time.day
        
        # Calculate focus percentage from 2-10 scale to 0-100% for image mapping
        focus_percentage = (session.average_focus_score / 10) * 100
        
        # If multiple sessions on same day, keep the one with highest focus score
        if day not in session_data_by_day or session.average_focus_score > session_data_by_day[day]['average_focus_score']:
            session_data_by_day[day] = {
                'session_id': session.session_id,
                'start_time': session.start_time,
                'average_focus_score': session.average_focus_score,
                'focus_percentage': focus_percentage,
                'total_duration_seconds': session.total_duration_seconds,
                'concentrating_seconds': session.concentrating_seconds,
                'peak_focus_score': session.peak_focus_score
            }
    
    # Generate calendar weeks using calendar.monthcalendar(year, month)
    # Set Sunday as first day of week to match typical calendar display
    calendar.setfirstweekday(calendar.SUNDAY)
    cal = calendar.monthcalendar(current_year, current_month)
    calendar_weeks = []
    
    for week in cal:
        week_days = []
        for day_num in week:
            if day_num == 0:  # Day from previous/next month
                day_obj = {
                    'day_num': day_num,
                    'in_month': False,
                    'has_data': False,
                    'image_path': None,
                    'focus_pct': None
                }
            else:
                has_data = day_num in session_data_by_day
                if has_data:
                    session_data = session_data_by_day[day_num]
                    focus_pct = session_data['focus_percentage']
                    
                    # Map focus percentage to image path based on requirements
                    if focus_pct >= 75:
                        image_path = '/static/images/Master.jpg'
                    elif focus_pct >= 60:
                        image_path = '/static/images/LockedIn.jpg'
                    elif focus_pct >= 45:
                        image_path = '/static/images/Steady.jpg'
                    elif focus_pct >= 30:
                        image_path = '/static/images/Neutral.jpg'
                    elif focus_pct >= 15:
                        image_path = '/static/images/Distracted.jpg'
                    else:
                        image_path = '/static/images/BrainFog.jpg'
                else:
                    focus_pct = None
                    image_path = None
                
                day_obj = {
                    'day_num': day_num,
                    'in_month': True,
                    'has_data': has_data,
                    'image_path': image_path,
                    'focus_pct': focus_pct
                }
            
            week_days.append(day_obj)
        
        calendar_weeks.append(week_days)
    
    # Calculate monthly statistics
    sessions_count = sessions.count()
    avg_focus = 0
    if sessions_count > 0:
        avg_focus = sum(session.average_focus_score for session in sessions) / sessions_count
        # Convert from 2-10 scale to 0-100% for display
        avg_focus_10_point = (avg_focus * 10)  # 2-10 scale becomes 20-100%
    else:
        avg_focus_10_point = 0
    
    active_days = len(session_data_by_day)
    
    # Calculate navigation context
    if current_month == 1:
        prev_month = 12
        prev_year = current_year - 1
    else:
        prev_month = current_month - 1
        prev_year = current_year
    
    if current_month == 12:
        next_month = 1
        next_year = current_year + 1
    else:
        next_month = current_month + 1
        next_year = current_year
    
    # Get month name for display
    month_name = calendar.month_name[current_month]
    
    # Return JSON response
    return JsonResponse({
        'calendar_weeks': calendar_weeks,
        'current_month': current_month,
        'current_year': current_year,
        'month_name': month_name,
        'sessions_this_month': sessions_count,
        'avg_focus_10_point': avg_focus_10_point,
        'active_days': active_days,
        'prev_month': prev_month,
        'prev_year': prev_year,
        'next_month': next_month,
        'next_year': next_year,
        'focus_legend': [
            {'percentage': '75%+', 'image': '/static/images/Master.jpg', 'label': 'Master'},
            {'percentage': '60-74%', 'image': '/static/images/LockedIn.jpg', 'label': 'Locked In'},
            {'percentage': '45-59%', 'image': '/static/images/Steady.jpg', 'label': 'Steady'},
            {'percentage': '30-44%', 'image': '/static/images/Neutral.jpg', 'label': 'Neutral'},
            {'percentage': '15-29%', 'image': '/static/images/Distracted.jpg', 'label': 'Distracted'},
            {'percentage': '<15%', 'image': '/static/images/BrainFog.jpg', 'label': 'Brain Fog'}
        ]
    })

def study_time_api_data(request):
    """API endpoint for study time analysis data"""
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'User not authenticated'}, status=401)
    
    try:
        user_profile = UserProfile.objects.get(email=user_email)
    except UserProfile.DoesNotExist:
        return JsonResponse({'error': 'User profile not found'}, status=404)
    
    # Get scale parameter (week, month, year)
    scale = request.GET.get('scale', 'week')
    if scale not in ['week', 'month', 'year']:
        scale = 'week'
    
    from datetime import datetime, timedelta
    from django.db.models import Sum, Count
    from django.db.models.functions import TruncDay, TruncWeek, TruncMonth, Extract
    
    now = timezone.now()
    
    if scale == 'week':
        # Get data for current week (last 7 days) - use local timezone
        start_date = timezone.localtime(now) - timedelta(days=6)
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = timezone.localtime(now).replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Convert to UTC for database query
        start_date_utc = timezone.make_aware(start_date.replace(tzinfo=None))
        end_date_utc = timezone.make_aware(end_date.replace(tzinfo=None))
        
        sessions = SessionSummary.objects.filter(
            user=user_profile,
            start_time__gte=start_date_utc,
            start_time__lte=end_date_utc
        ).annotate(
            date=TruncDay('start_time')
        ).values('date').annotate(
            total_seconds=Sum('total_duration_seconds')
        ).order_by('date')
        
        # Prepare data for each day of the week
        week_data = []
        day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        
        for i in range(7):
            date = start_date + timedelta(days=i)
            # Convert date to UTC for comparison with database dates
            date_utc = timezone.make_aware(datetime.combine(date.date(), datetime.min.time()))
            day_sessions = [s for s in sessions if s['date'].date() == date_utc.date()]
            total_seconds = day_sessions[0]['total_seconds'] if day_sessions else 0
            week_data.append({
                'label': day_names[i],
                'seconds': total_seconds
            })
        
        # Calculate statistics
        total_seconds = sum(item['seconds'] for item in week_data)
        active_days = len([item for item in week_data if item['seconds'] > 0])
        daily_avg = total_seconds / active_days if active_days > 0 else 0
        
        # Find best day
        best_day_idx = max(range(len(week_data)), key=lambda i: week_data[i]['seconds'])
        best_day = day_names[best_day_idx] if week_data[best_day_idx]['seconds'] > 0 else 'None'
        
    elif scale == 'month':
        # Get data for current month grouped by week - use local timezone
        local_now = timezone.localtime(now)
        month_start = timezone.make_aware(datetime(local_now.year, local_now.month, 1))
        if local_now.month == 12:
            month_end = timezone.make_aware(datetime(local_now.year + 1, 1, 1))
        else:
            month_end = timezone.make_aware(datetime(local_now.year, local_now.month + 1, 1))
        
        sessions = SessionSummary.objects.filter(
            user=user_profile,
            start_time__gte=month_start,
            start_time__lt=month_end
        ).annotate(
            week=TruncWeek('start_time')
        ).values('week').annotate(
            total_seconds=Sum('total_duration_seconds')
        ).order_by('week')
        
        # Create consistent 4-week structure
        month_data = []
        week_seconds = {i: 0 for i in range(4)}  # Initialize all 4 weeks to 0
        
        # Map sessions to week indices
        for session in sessions:
            # Calculate week of month (0-3) using local timezone
            week_start = timezone.localtime(session['week']).date()
            month_start_local = month_start.date()
            week_index = (week_start - month_start_local).days // 7
            if 0 <= week_index <= 3:  # Ensure we don't exceed 4 weeks
                week_seconds[week_index] = session['total_seconds'] or 0
        
        # Create data for all 4 weeks consistently
        for i in range(4):
            month_data.append({
                'label': f'Week {i+1}',
                'seconds': week_seconds[i]
            })
        
        # Calculate statistics
        total_seconds = sum(item['seconds'] for item in month_data)
        active_days = SessionSummary.objects.filter(
            user=user_profile,
            start_time__gte=month_start,
            start_time__lt=month_end
        ).values('start_time__date').distinct().count()
        daily_avg = total_seconds / active_days if active_days > 0 else 0
        
        # Find best week
        best_week_idx = max(range(4), key=lambda i: month_data[i]['seconds'])
        best_day = f'Week {best_week_idx + 1}' if month_data[best_week_idx]['seconds'] > 0 else 'None'
        
    else:  # year
        # Get data for current year grouped by month - use local timezone
        local_now = timezone.localtime(now)
        year_start = timezone.make_aware(datetime(local_now.year, 1, 1))
        year_end = timezone.make_aware(datetime(local_now.year + 1, 1, 1))
        
        sessions = SessionSummary.objects.filter(
            user=user_profile,
            start_time__gte=year_start,
            start_time__lt=year_end
        ).annotate(
            month=TruncMonth('start_time')
        ).values('month').annotate(
            total_seconds=Sum('total_duration_seconds')
        ).order_by('month')
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        year_data = []
        
        # Create data for all 12 months
        month_seconds = {i: 0 for i in range(1, 13)}
        for session in sessions:
            month = timezone.localtime(session['month']).month
            month_seconds[month] = session['total_seconds'] or 0
        
        for i in range(1, 13):
            year_data.append({
                'label': month_names[i-1],
                'seconds': month_seconds[i]
            })
        
        # Calculate statistics
        total_seconds = sum(year_data[i]['seconds'] for i in range(12))
        
        # For year view, calculate daily average based on days passed in current year
        from datetime import date
        year_start_local = date(local_now.year, 1, 1)
        days_passed = (local_now.date() - year_start_local).days + 1  # +1 to include current day
        
        # Use actual active days if they exist, otherwise use days passed
        active_days = SessionSummary.objects.filter(
            user=user_profile,
            start_time__gte=year_start,
            start_time__lt=year_end
        ).values('start_time__date').distinct().count()
        
        daily_avg = total_seconds / active_days if active_days > 0 else total_seconds / days_passed
        
        # Find best month
        best_month_idx = max(range(12), key=lambda i: year_data[i]['seconds'])
        best_day = month_names[best_month_idx] if year_data[best_month_idx]['seconds'] > 0 else 'None'
    
    # Format time strings
    def format_seconds(seconds):
        if seconds < 60:
            return f"{seconds}min"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        if hours == 0:
            return f"{minutes}min"
        elif minutes == 0:
            return f"{hours}h"
        else:
            return f"{hours}h {minutes}min"
    
    # Calculate max value for percentage scaling
    data = week_data if scale == 'week' else (month_data if scale == 'month' else year_data)
    max_seconds = max(item['seconds'] for item in data) if data else 1
    
    # Scale data to percentages
    scaled_data = []
    for item in data:
        percentage = (item['seconds'] / max_seconds * 100) if max_seconds > 0 else 0
        scaled_data.append({
            'label': item['label'],
            'seconds': item['seconds'],
            'height': percentage,
            'formatted_time': format_seconds(item['seconds'])
        })
    
    # Determine the best period label based on scale
    if scale == 'week':
        best_period_label = 'Best Day'
    elif scale == 'month':
        best_period_label = 'Best Week'
    else:  # year
        best_period_label = 'Best Month'
    
    return JsonResponse({
        'scale': scale,
        'data': scaled_data,
        'total_study_time': format_seconds(total_seconds),
        'daily_average': format_seconds(daily_avg),
        'best_day': best_day,
        'best_period_label': best_period_label,
        'active_days': active_days
    })

def email_entry(request):
    """Landing page for email entry"""
    return render(request, 'email_entry.html')

def send_otp(request):
    """Generate and send OTP for email verification"""
    if request.method == 'POST':
        email = request.POST.get('email')
        
        if not email:
            return JsonResponse({'error': 'Email is required'}, status=400)
        
        # Generate 6-digit OTP
        otp_code = ''.join(random.choices(string.digits, k=6))
        
        # Save OTP to database
        from .models import EmailOTP
        EmailOTP.objects.filter(email=email).delete()  # Remove any existing OTPs
        EmailOTP.objects.create(email=email, otp_code=otp_code)
        
        # Send OTP email
        try:
            from django.core.mail import send_mail
            subject = 'BrainWave - Your OTP Code'
            message = f'Your BrainWave verification code is: {otp_code}\n\nThis code will expire in 5 minutes.'
            from_email = 'noreply@brainwave.com'
            recipient_list = [email]
            
            send_mail(
                subject,
                message,
                from_email,
                recipient_list,
                fail_silently=False,
            )
        except Exception as e:
            pass
            # For now, we'll continue with the flow even if email fails
        
        # Redirect to OTP verification page
        return render(request, 'otp_verification.html', {'email': email})
    
    return redirect('/')

def verify_otp(request):
    """Verify OTP and handle user routing"""
    if request.method == 'POST':
        email = request.POST.get('email')
        otp_code = request.POST.get('otp_code')
        
        if not email or not otp_code:
            return JsonResponse({'error': 'Email and OTP are required'}, status=400)
        
        from .models import EmailOTP, UserProfile
        
        # Verify OTP
        try:
            otp_record = EmailOTP.objects.get(email=email, otp_code=otp_code)
            if not otp_record.is_valid():
                return JsonResponse({'error': 'OTP has expired'}, status=400)
        except EmailOTP.DoesNotExist:
            return JsonResponse({'error': 'Invalid OTP'}, status=400)
        
        # OTP is valid - set session
        request.session['user_email'] = email
        request.session.modified = True
        
        # Clean up OTP
        otp_record.delete()
        
        # Check if user exists and has completed survey
        user_exists = UserProfile.objects.filter(email=email).exists()
        if user_exists:
            user_profile = UserProfile.objects.get(email=email)
            # Check if profile is complete (has basic info)
            if user_profile.name and user_profile.academic_level:
                return redirect('/dashboard/')
            else:
                return redirect('/onboarding/')
        else:
            # New user - redirect to survey
            return redirect('/onboarding/')
    
    return redirect('/')

def dashboard_view(request):
    """Dashboard view with user profile and focus tracking data"""
    # Security check - ensure user is authenticated via OTP
    user_email = request.session.get('user_email')
    if not user_email:
        return redirect('/')
    
    # Get user profile data from database
    from .models import UserProfile
    try:
        user_profile = UserProfile.objects.get(email=user_email)
    except UserProfile.DoesNotExist:
        return redirect('/onboarding/')
    
    # Import data service
    from .services.data_service import FocusDataService
    data_service = FocusDataService(user_email)
    
    # Helper functions to map indices to human-readable text
    def get_academic_level_text(level_id):
        levels = {
            0: 'High School',
            1: 'Undergraduate', 
            2: 'Graduate/Masters',
            3: 'PhD/Doctoral',
            4: 'Professional/Continuing Education',
            5: 'Other'
        }
        return levels.get(int(level_id), 'Not Set')
    
    def get_sleep_quality_text(quality_id):
        qualities = {
            0: 'Poor (frequently disrupted)',
            1: 'Fair (occasional issues)',
            2: 'Good (generally restful)',
            3: 'Excellent (consistently deep)'
        }
        return qualities.get(int(quality_id), 'Not Set')
    
    def get_alert_time_text(time_id):
        times = {
            0: 'Early Morning (5am-9am)',
            1: 'Mid-Morning to Afternoon (9am-5pm)',
            2: 'Evening to Late Night (5pm-12am)',
            3: 'Late Night (12am-5am)'
        }
        return times.get(int(time_id), 'Not Set')
    
    def get_learning_style_text(style_id):
        styles = {
            0: 'Visual Learner',
            1: 'Auditory Learner',
            2: 'Kinesthetic Learner',
            3: 'Reading/Writing Learner'
        }
        return styles.get(int(style_id), 'Not Set')
    
    def get_session_length_text(length_id):
        lengths = {
            0: 'Short (15-30 min)',
            1: 'Medium (30-60 min)',
            2: 'Long (60-90 min)',
            3: 'Extended (90+ min)'
        }
        return lengths.get(int(length_id), 'Not Set')
    
    def get_sound_environment_text(env_id):
        environments = {
            0: 'Silent',
            1: 'Quiet',
            2: 'Moderate Noise',
            3: 'Background Music',
            4: 'Cafe/Office',
            5: 'Outdoor Environment'
        }
        return environments.get(int(env_id), 'Not Set')
    
    def get_main_goals_display(goals_string):
        """Convert comma-separated goal indices to display text"""
        if not goals_string:
            return 'Not Set'
        
        # Define goal choices mapping
        GOAL_CHOICES = {
            '1': 'Improve Grades',
            '2': 'Learn New Skill', 
            '3': 'Complete Assignments',
            '4': 'Prepare for Exams',
            '5': 'Increase Study Time',
            '6': 'Better Time Management',
            '7': 'Reduce Distractions',
            '8': 'Improve Focus',
            '9': 'Career Development',
            '10': 'Personal Growth'
        }
        
        # Parse the string and map each index to text
        try:
            # Handle both comma-separated and bracketed formats
            goals_str = goals_string.strip('[]').replace("'", "").replace(" ", "")
            goal_indices = [g.strip() for g in goals_str.split(',') if g.strip()]
            
            # Map each index to its text
            goal_texts = []
            for index in goal_indices:
                goal_text = GOAL_CHOICES.get(index.strip())
                if goal_text:
                    goal_texts.append(goal_text)
            
            return ', '.join(goal_texts) if goal_texts else 'Not Set'
        except Exception:
            return goals_string
    
    def get_study_location_display(location_string):
        """Convert comma-separated location indices to display text"""
        if not location_string:
            return 'Not Set'
        
        # Define location choices mapping
        LOCATION_CHOICES = {
            '1': 'Library',
            '2': 'Home Office',
            '3': 'Bedroom',
            '4': 'Living Room',
            '5': 'Cafe/Coffee Shop',
            '6': 'Study Room',
            '7': 'Classroom',
            '8': 'Co-working Space',
            '9': 'Outdoor',
            '10': 'Other'
        }
        
        # Parse the string and map each index to text
        try:
            # Handle both comma-separated and bracketed formats
            location_str = location_string.strip('[]').replace("'", "").replace(" ", "")
            location_indices = [l.strip() for l in location_str.split(',') if l.strip()]
            
            # Map each index to its text
            location_texts = []
            for index in location_indices:
                location_text = LOCATION_CHOICES.get(index.strip())
                if location_text:
                    location_texts.append(location_text)
            
            return ', '.join(location_texts) if location_texts else 'Not Set'
        except Exception:
            return location_string
    
    def get_health_conditions_display(health_string):
        """Convert comma-separated health condition indices to display text"""
        if not health_string:
            return 'Not Set'
        
        # Define health condition choices mapping
        HEALTH_CHOICES = {
            '1': 'Anxiety',
            '2': 'ADHD',
            '3': 'Depression',
            '4': 'Insomnia',
            '5': 'Migraines',
            '6': 'None',
            '7': 'Other'
        }
        
        # Parse the string and map each index to text
        try:
            # Handle both comma-separated and bracketed formats
            health_str = health_string.strip('[]').replace("'", "").replace(" ", "")
            health_indices = [h.strip() for h in health_str.split(',') if h.strip()]
            
            # Map each index to its text
            health_texts = []
            for index in health_indices:
                health_text = HEALTH_CHOICES.get(index.strip())
                if health_text:
                    health_texts.append(health_text)
            
            return ', '.join(health_texts) if health_texts else 'Not Set'
        except Exception:
            return health_string
    
    def get_study_subjects_display(subjects_string):
        """Convert comma-separated subject indices to display text"""
        if not subjects_string:
            return 'Not Set'
        
        # Define subject choices mapping
        SUBJECT_CHOICES = {
            '1': 'Mathematics',
            '2': 'Science',
            '3': 'English/Literature',
            '4': 'History/Social Studies',
            '5': 'Computer Science',
            '6': 'Arts',
            '7': 'Languages',
            '8': 'Business',
            '9': 'Other'
        }
        
        # Parse the string and map each index to text
        try:
            # Handle both comma-separated and bracketed formats
            subjects_str = subjects_string.strip('[]').replace("'", "").replace(" ", "")
            subject_indices = [s.strip() for s in subjects_str.split(',') if s.strip()]
            
            # Map each index to its text
            subject_texts = []
            for index in subject_indices:
                subject_text = SUBJECT_CHOICES.get(index.strip())
                if subject_text:
                    subject_texts.append(subject_text)
            
            return ', '.join(subject_texts) if subject_texts else 'Not Set'
        except Exception:
            return subjects_string
        environments = {
            0: 'Complete Silence',
            1: 'White Noise',
            2: 'Soft Music',
            3: 'Nature Sounds',
            4: 'Cafe/Background Noise',
            5: 'Instrumental Music'
        }
        return environments.get(int(env_id), 'Not Set')
    
    def get_study_time_text(time_id):
        times = {
            0: 'Early Morning',
            1: 'Morning',
            2: 'Afternoon', 
            3: 'Evening',
            4: 'Night'
        }
        return times.get(int(time_id), 'Not Set')
    
    # Create user profile snapshot with human-readable text
    profile_snapshot = {
        'name': user_profile.name,
        'age': user_profile.age,
        'academic_level': get_academic_level_text(user_profile.academic_level),
        'sleep_hours': user_profile.sleep_hours,
        'sleep_quality': get_sleep_quality_text(user_profile.sleep_quality),
        'learning_style': get_learning_style_text(user_profile.learning_style),
        'study_subjects': get_study_subjects_display(user_profile.study_subjects),
        'caffeine_servings': user_profile.caffeine_servings,
        'procrastination_level': user_profile.procrastination_level,
        'main_goals': get_main_goals_display(user_profile.main_goals),
        'sound_environment': get_sound_environment_text(user_profile.sound_environment),
        'study_location': get_study_location_display(user_profile.study_location),
        'phone_location': user_profile.phone_location,
        'distractions': user_profile.distractions,
        'exercise_frequency': user_profile.exercise_frequency,
        'eating_timing': user_profile.eating_timing,
        'health_conditions': get_health_conditions_display(user_profile.health_conditions),
        'session_length': get_session_length_text(user_profile.session_length),
        'study_time_of_day': get_study_time_text(user_profile.study_time_of_day),
        'alert_time': get_alert_time_text(user_profile.alert_time)
    }
    
    # Get focus tracking data with filter support
    filter_value = request.GET.get('filter', '5')  # Default to last 5 sessions
    if filter_value == '7':
        recent_sessions = data_service.get_recent_sessions(days=7)
    elif filter_value == '30':
        recent_sessions = data_service.get_recent_sessions(days=30)
    else:
        recent_sessions = data_service.get_recent_sessions(limit=5)
    
    # Calculate aggregate statistics for the filtered sessions
    from django.db.models import Sum, Avg
    from secondBrain_App.models import SessionSummary
    
    # Get the base queryset for filtered sessions
    if filter_value == '7':
        from datetime import timedelta
        cutoff_date = timezone.now() - timedelta(days=7)
        sessions_queryset = SessionSummary.objects.filter(
            user=user_profile, 
            start_time__gte=cutoff_date
        )
    elif filter_value == '30':
        from datetime import timedelta
        cutoff_date = timezone.now() - timedelta(days=30)
        sessions_queryset = SessionSummary.objects.filter(
            user=user_profile, 
            start_time__gte=cutoff_date
        )
    else:
        # Get the session IDs from recent_sessions to ensure consistency
        session_ids = [s['id'] for s in recent_sessions]
        sessions_queryset = SessionSummary.objects.filter(
            user=user_profile,
            id__in=session_ids
        )
    
    # Calculate aggregate stats
    total_minutes = sessions_queryset.aggregate(
        total=Sum('total_duration_seconds')
    )['total'] or 0
    total_minutes = int(total_minutes / 60)  # Convert seconds to minutes
    
    avg_focus = sessions_queryset.aggregate(
        avg=Avg('average_focus_score')
    )['avg'] or 0
    
    # For check-ins, we'll use a placeholder since the field doesn't exist yet
    # TODO: Update when check_ins_count field is added to SessionSummary
    total_checkins = sessions_queryset.count()  # Using session count as placeholder
    
    aggregate_stats = {
        'total_minutes': total_minutes,
        'avg_focus': round(avg_focus, 1),
        'total_checkins': total_checkins
    }
    
    # Calculate overall stats for progress tracker (all sessions, not filtered)
    all_sessions = SessionSummary.objects.filter(user=user_profile)
    total_sessions_count = all_sessions.count()
    overall_avg_focus = 0
    if total_sessions_count > 0:
        overall_avg_focus = all_sessions.aggregate(avg=Avg('average_focus_score'))['avg'] or 0
    total_study_minutes = all_sessions.aggregate(total=Sum('total_duration_seconds'))['total'] or 0
    total_study_minutes = int(total_study_minutes / 60)  # Convert to minutes
    
    # Determine current level based on average focus score (already 0-10 scale in DB)
    avg_focus_10 = min(round(overall_avg_focus, 1), 10.0)
    if avg_focus_10 < 5.0:
        current_level = {'emoji': '🎯', 'name': 'Beginner'}
    elif avg_focus_10 < 6.0:
        current_level = {'emoji': '🌱', 'name': 'Developing'}
    elif avg_focus_10 < 7.0:
        current_level = {'emoji': '📈', 'name': 'Intermediate'}
    elif avg_focus_10 < 8.0:
        current_level = {'emoji': '🌟', 'name': 'Advanced'}
    else:
        current_level = {'emoji': '🏆', 'name': 'Expert'}
    
    # Calculate progress for each level
    levels = [
        {'emoji': '🎯', 'name': 'Beginner', 'range': '< 5.0', 'status': 'completed' if avg_focus_10 >= 5.0 else 'current' if avg_focus_10 < 5.0 else 'locked', 'progress': 100 if avg_focus_10 >= 5.0 else min((avg_focus_10 / 5.0) * 100, 100)},
        {'emoji': '🌱', 'name': 'Developing', 'range': '5.0-5.9', 'status': 'completed' if avg_focus_10 >= 6.0 else 'current' if 5.0 <= avg_focus_10 < 6.0 else 'locked', 'progress': 100 if avg_focus_10 >= 6.0 else min(((avg_focus_10 - 5.0) / 1.0) * 100, 100) if avg_focus_10 >= 5.0 else 0},
        {'emoji': '📈', 'name': 'Intermediate', 'range': '6.0-6.9', 'status': 'completed' if avg_focus_10 >= 7.0 else 'current' if 6.0 <= avg_focus_10 < 7.0 else 'locked', 'progress': 100 if avg_focus_10 >= 7.0 else min(((avg_focus_10 - 6.0) / 1.0) * 100, 100) if avg_focus_10 >= 6.0 else 0},
        {'emoji': '🌟', 'name': 'Advanced', 'range': '7.0-7.9', 'status': 'completed' if avg_focus_10 >= 8.0 else 'current' if 7.0 <= avg_focus_10 < 8.0 else 'locked', 'progress': 100 if avg_focus_10 >= 8.0 else min(((avg_focus_10 - 7.0) / 1.0) * 100, 100) if avg_focus_10 >= 7.0 else 0},
        {'emoji': '🏆', 'name': 'Expert', 'range': '≥ 8.0', 'status': 'current' if avg_focus_10 >= 8.0 else 'locked', 'progress': min(((avg_focus_10 - 8.0) / 2.0) * 100, 100) if avg_focus_10 >= 8.0 else 0}
    ]
    
    session_stats = data_service.get_session_average_stats()
    recommendations = data_service.get_recommendations()
    
    # Generate calendar data for the requested month
    from datetime import datetime
    import calendar
    
    try:
        requested_month = int(request.GET.get('month', ''))
        requested_year = int(request.GET.get('year', ''))
        
        # Validate month and year ranges
        if requested_month < 1 or requested_month > 12:
            raise ValueError("Invalid month")
        if requested_year < 2020 or requested_year > 2030:
            raise ValueError("Invalid year")
            
        current_month = requested_month
        current_year = requested_year
    except (ValueError, TypeError):
        # Default to current date if parameters are invalid or missing
        now = timezone.now()
        current_month = now.month
        current_year = now.year
    
    # Query SessionSummary for current user and requested month/year
    from .models import SessionSummary
    sessions = SessionSummary.objects.filter(
        user=user_profile,
        start_time__year=current_year,
        start_time__month=current_month
    ).order_by('start_time')
    
    # Create session data mapping by day - collect all sessions for averaging
    session_data_by_day = {}
    for session in sessions:
        day = session.start_time.day
        
        # Collect all sessions for this day to calculate average
        if day not in session_data_by_day:
            session_data_by_day[day] = {
                'total_focus_score': 0,
                'session_count': 0
            }
        
        session_data_by_day[day]['total_focus_score'] += session.average_focus_score
        session_data_by_day[day]['session_count'] += 1
    
    # Calculate average focus for each day
    for day, data in session_data_by_day.items():
        avg_focus_score = data['total_focus_score'] / data['session_count']
        # Convert 0-10 scale to 0-100% for display, cap at 100
        focus_percentage = min(avg_focus_score * 10, 100)
        
        session_data_by_day[day] = {
            'average_focus_score': avg_focus_score,
            'focus_percentage': focus_percentage,
            'session_count': data['session_count']
        }
    
    # Generate calendar weeks using calendar.monthcalendar(year, month)
    # Set Sunday as first day of week to match typical calendar display
    calendar.setfirstweekday(calendar.SUNDAY)
    cal = calendar.monthcalendar(current_year, current_month)
    calendar_weeks = []
    
    for week in cal:
        week_days = []
        for day_num in week:
            if day_num == 0:  # Day from previous/next month
                day_obj = {
                    'day_num': day_num,
                    'in_month': False,
                    'has_data': False,
                    'image_path': None,
                    'focus_pct': None
                }
            else:
                has_data = day_num in session_data_by_day
                if has_data:
                    session_data = session_data_by_day[day_num]
                    focus_pct = session_data['focus_percentage']
                    
                    # Map focus percentage to image path based on requirements
                    if focus_pct >= 75:
                        image_path = '/static/images/Master.jpg'
                    elif focus_pct >= 60:
                        image_path = '/static/images/LockedIn.jpg'
                    elif focus_pct >= 45:
                        image_path = '/static/images/Steady.jpg'
                    elif focus_pct >= 30:
                        image_path = '/static/images/Neutral.jpg'
                    elif focus_pct >= 15:
                        image_path = '/static/images/Distracted.jpg'
                    else:
                        image_path = '/static/images/BrainFog.jpg'
                else:
                    focus_pct = None
                    image_path = None
                
                day_obj = {
                    'day_num': day_num,
                    'in_month': True,
                    'has_data': has_data,
                    'image_path': image_path,
                    'focus_pct': focus_pct
                }
            
            week_days.append(day_obj)
        
        calendar_weeks.append(week_days)
    
    # Calculate monthly statistics
    sessions_count = sessions.count()
    avg_focus = 0
    if sessions_count > 0:
        avg_focus = sum(session.average_focus_score for session in sessions) / sessions_count
        # Convert from 2-10 scale to 0-100% for display
        avg_focus_10_point = min(round(avg_focus * 10, 1), 100.0)  # 2-10 scale becomes 20-100%, cap at 100%
    else:
        avg_focus_10_point = 0
    
    active_days = len(session_data_by_day)
    
    # Calculate navigation context
    if current_month == 1:
        prev_month = 12
        prev_year = current_year - 1
    else:
        prev_month = current_month - 1
        prev_year = current_year
    
    if current_month == 12:
        next_month = 1
        next_year = current_year + 1
    else:
        next_month = current_month + 1
        next_year = current_year
    
    # Get month name for display
    month_name = calendar.month_name[current_month]
    
    cache_key = f"recommendation_{user_email}"
    cached_data = cache.get(cache_key)
    ai_recommendation_text = cached_data.get('text') if cached_data else None
    
    if not ai_recommendation_text:
        from .models import Recommendation
        latest_rec = Recommendation.objects.filter(user__email=user_email).order_by('-created_at').first()
        ai_recommendation_text = latest_rec.message if latest_rec else None
        if ai_recommendation_text:
            # Remove markdown symbols so they don't clutter the UI
            ai_recommendation_text = ai_recommendation_text.replace('**', '').replace('###', '').replace('##', '')
    
    context = {
        'user': request.user if request.user.is_authenticated else None,
        'user_profile': profile_snapshot,
        'recent_sessions': recent_sessions,
        'session_stats': session_stats,
        'recommendations': recommendations,
        'ai_recommendation': ai_recommendation_text,
        'aggregate_stats': aggregate_stats,
        'calendar_weeks': calendar_weeks,
        'current_month': current_month,
        'current_year': current_year,
        'month_name': month_name,
        'sessions_this_month': sessions_count,
        'avg_focus_10_point': avg_focus_10_point,
        'active_days': active_days,
        'prev_month': prev_month,
        'prev_year': prev_year,
        'next_month': next_month,
        'next_year': next_year,
        'focus_legend': [
            {'percentage': '75%+', 'image': '/static/images/Master.jpg', 'label': 'Master'},
            {'percentage': '60-74%', 'image': '/static/images/LockedIn.jpg', 'label': 'Locked In'},
            {'percentage': '45-59%', 'image': '/static/images/Steady.jpg', 'label': 'Steady'},
            {'percentage': '30-44%', 'image': '/static/images/Neutral.jpg', 'label': 'Neutral'},
            {'percentage': '15-29%', 'image': '/static/images/Distracted.jpg', 'label': 'Distracted'},
            {'percentage': '<15%', 'image': '/static/images/BrainFog.jpg', 'label': 'Brain Fog'}
        ],
        'progress_stats': {
            'total_sessions': total_sessions_count,
            'avg_focus': avg_focus_10,  # Already 0-10 scale
            'total_study_minutes': total_study_minutes,
            'study_hours': total_study_minutes // 60,
            'study_remaining_minutes': total_study_minutes % 60
        },
        'current_level': current_level,
        'levels': levels
    }
    
    return render(request, 'dashboard.html', context)

def onboarding_view(request):
    """Render the onboarding page with multi-step form handling"""
    
    # Security check - ensure user is authenticated via OTP
    user_email = request.session.get('user_email')
    if not user_email:
        return redirect('/')
    
    # Define all sections and their questions
    SECTIONS = {
        1: {
            'title': 'SECTION 1: BASIC INFORMATION',
            'questions': [
                {'id': 'name', 'type': 'text', 'label': "What's your name?", 'placeholder': 'Enter your name'},
                {'id': 'age', 'type': 'number', 'label': 'How old are you?', 'placeholder': 'Enter your age', 'min': 13, 'max': 99},
                {'id': 'academic_level', 'type': 'radio', 'label': 'What\'s your academic level?', 
                 'options': ['High School', 'Undergraduate', 'Graduate/Masters', 'PhD/Doctoral', 'Professional/Continuing Education', 'Other']}
            ]
        },
        2: {
            'title': 'SECTION 2: YOUR NATURAL RHYTHMS',
            'questions': [
                {'id': 'alert_time', 'type': 'radio', 'label': 'When do you feel most alert and energetic?',
                 'options': ['Early Morning (5am-9am) - Morning Person', 'Mid-Morning to Afternoon (9am-5pm) - Intermediate', 
                          'Evening to Late Night (5pm-12am) - Night Owl', 'Late Night (12am-5am) - Night Owl']},
                {'id': 'sleep_hours', 'type': 'range', 'label': 'How many hours of sleep do you typically get per night?', 'min': 4, 'max': 11, 'default': 7},
                {'id': 'sleep_quality', 'type': 'radio', 'label': 'How would you rate your sleep quality?',
                 'options': ['Poor (frequently disrupted, unrefreshing)', 'Fair (occasional issues)', 
                          'Good (generally restful)', 'Excellent (consistently deep and refreshing)']}
            ]
        },
        3: {
            'title': 'SECTION 3: CAFFEINE & STIMULANTS',
            'questions': [
                {'id': 'consumes_caffeine', 'type': 'radio', 'label': 'Do you consume caffeine?', 'options': ['Yes', 'No']},
                {'id': 'caffeine_types', 'type': 'checkbox', 'label': 'What type(s) of caffeine? (if Yes)',
                 'options': ['Coffee', 'Tea (black/green)', 'Energy drinks', 'Soda', 'Other']},
                {'id': 'caffeine_servings', 'type': 'number', 'label': 'How many servings per day? (if Yes)', 'min': 1, 'max': 10},
                {'id': 'caffeine_timing', 'type': 'radio', 'label': 'When do you typically consume caffeine? (if Yes)',
                 'options': ['Early morning only (before 10am)', 'Morning to noon (before 12pm)', 
                          'Throughout the day (morning to afternoon)', 'Anytime (including evenings)']}
            ]
        },
        4: {
            'title': 'SECTION 4: LEARNING STYLE & PREFERENCES',
            'questions': [
                {'id': 'learning_style', 'type': 'radio', 'label': 'How do you learn best?',
                 'options': ['Visual (diagrams, charts, videos, reading)', 'Auditory (listening to lectures, discussions, audio)', 
                          'Kinesthetic (hands-on practice, movement, doing)', 'Reading/Writing (taking notes, writing summaries)', 'Not sure']},
                {'id': 'study_subjects', 'type': 'checkbox', 'label': 'What subjects/topics do you typically study?',
                 'options': ['Math/Statistics/Calculus', 'Sciences (Biology, Chemistry, Physics)', 'Reading/Literature/Humanities', 
                          'Writing/Essays/Creative Work', 'Languages', 'Programming/Computer Science', 'Memorization-heavy subjects', 'Other']}
            ]
        },
        5: {
            'title': 'SECTION 5: CURRENT STUDY HABITS',
            'questions': [
                {'id': 'session_length', 'type': 'radio', 'label': 'How long do you typically study in one session?',
                 'options': ['15-30 minutes', '30-45 minutes', '45-60 minutes', '1-2 hours', '2+ hours', 'It varies widely']},
                {'id': 'takes_breaks', 'type': 'radio', 'label': 'Do you currently take breaks during study sessions?',
                 'options': ['No, I study straight through', 'Yes, when I feel tired', 'Yes, every 25-30 minutes (Pomodoro-style)', 
                          'Yes, every 45-60 minutes', 'Other']},
                {'id': 'study_time_of_day', 'type': 'radio', 'label': 'What time of day do you usually study?',
                 'options': ['Early Morning (5am-9am)', 'Mid-Morning (9am-12pm)', 'Afternoon (12pm-5pm)', 
                          'Evening (5pm-9pm)', 'Night (9pm-12am)', 'Late Night (12am-5am)', 'Varies by day']},
                {'id': 'procrastination_level', 'type': 'range', 'label': 'On a scale of 1-10, how much do you struggle with procrastination?', 'min': 1, 'max': 10, 'default': 5}
            ]
        },
        6: {
            'title': 'SECTION 6: STUDY ENVIRONMENT PREFERENCES',
            'questions': [
                {'id': 'study_location', 'type': 'checkbox', 'label': 'Where do you typically study?',
                 'options': ['Library or study hall', 'Home (quiet room/desk)', 'Home (shared/noisy space)', 
                          'Coffee shop or café', 'Outdoors', 'Other']},
                {'id': 'sound_environment', 'type': 'radio', 'label': 'What sound environment do you prefer while studying?',
                 'options': ['Complete silence', 'White noise or brown noise', 'Nature sounds or ambient sounds', 
                          'Lo-fi or instrumental music', 'Classical music', 'Music with lyrics', 'Coffee shop ambience', 
                          'I\'m not sure yet / want to experiment', 'Other']},
                {'id': 'lighting_preference', 'type': 'radio', 'label': 'What lighting do you prefer for studying?',
                 'options': ['Natural daylight', 'Bright artificial light', 'Dim or warm lighting', 
                          'Mix of natural and artificial', 'Not sure / no strong preference']}
            ]
        },
        7: {
            'title': 'SECTION 7: DISTRACTIONS & FOCUS',
            'questions': [
                {'id': 'phone_location', 'type': 'radio', 'label': 'Where is your phone when you study?',
                 'options': ['Right next to me on my desk', 'In my pocket or bag (nearby)', 'In another room', 
                          'On "Do Not Disturb" mode nearby', 'I don\'t have specific habits around this']},
                {'id': 'distractions', 'type': 'checkbox', 'label': 'Which of these distractions affect you most?',
                 'options': ['Social media notifications', 'Text messages', 'Background noise/people talking', 
                          'Hunger/thirst', 'Uncomfortable seating', 'Temperature (too hot/cold)', 
                          'Wandering thoughts/daydreaming', 'Other']}
            ]
        },
        8: {
            'title': 'SECTION 8: PHYSICAL ACTIVITY & LIFESTYLE',
            'questions': [
                {'id': 'exercise_frequency', 'type': 'radio', 'label': 'How often do you engage in physical activity or exercise?',
                 'options': ['Daily', '3-5 times per week', '1-2 times per week', 'Rarely', 'Never']},
                {'id': 'eating_timing', 'type': 'radio', 'label': 'How long after eating do you typically study?',
                 'options': ['Immediately or within 30 minutes', '30 minutes to 1 hour', '1-2 hours', 
                          '2-4 hours', '4+ hours or on empty stomach', 'It varies']}
            ]
        },
        9: {
            'title': 'SECTION 9: HEALTH & WELLNESS (OPTIONAL)',
            'questions': [
                {'id': 'health_conditions', 'type': 'checkbox', 'label': 'Do you have any conditions that might affect your focus? (Optional)',
                 'options': ['ADHD or attention difficulties', 'Anxiety or high stress', 'Depression', 
                          'Sleep disorder', 'None', 'Prefer not to say', 'Other']}
            ]
        },
        10: {
            'title': 'SECTION 10: GOALS & EXPECTATIONS',
            'questions': [
                {'id': 'main_goals', 'type': 'checkbox', 'label': 'What are your main goals with this app?',
                 'options': ['Improve focus and concentration', 'Study more efficiently (less time, better results)', 
                          'Identify my optimal study environment', 'Build better study habits', 'Reduce procrastination', 
                          'Track my progress over time', 'Understand when I\'m most productive', 'Other']},
                {'id': 'study_effectiveness', 'type': 'range', 'label': 'How would you rate your current study effectiveness?', 'min': 1, 'max': 10, 'default': 5}
            ]
        }
    }
    
    # Handle form submission
    if request.method == 'POST':
        # Get current step from form data or default to 1
        current_step = int(request.POST.get('current_step', 1))
        
        # Store form data in session
        if 'onboarding_data' not in request.session:
            request.session['onboarding_data'] = {}
        
        # Save current step data
        step_data = {}
        section = SECTIONS.get(current_step, {})
        for question in section.get('questions', []):
            field_name = question['id']
            if question['type'] == 'checkbox':
                step_data[field_name] = request.POST.getlist(field_name)
            else:
                step_data[field_name] = request.POST.get(field_name, '')
        
        # Update session data
        request.session['onboarding_data'].update(step_data)
        request.session.modified = True
        
        # Move to next step
        next_step = current_step + 1
        
        if next_step <= 10:
            # Redirect to next step
            return redirect(f'/onboarding/?step={next_step}')
        else:
            # Complete onboarding - save to database and redirect to dashboard
            try:
                from .models import UserProfile
                
                # Get user email from session
                user_email = request.session.get('user_email')
                if not user_email:
                    return redirect('/')
                
                # Get all onboarding data
                onboarding_data = request.session.get('onboarding_data', {})
                
                # Handle multi-select fields (checkboxes)
                caffeine_types_list = onboarding_data.get('caffeine_types', [])
                study_subjects_list = onboarding_data.get('study_subjects', [])
                distractions_list = onboarding_data.get('distractions', [])
                
                # Check if user consumes caffeine
                consumes_caffeine = onboarding_data.get('consumes_caffeine', 'false').lower() == 'true'
                
                # If user doesn't consume caffeine, set caffeine details to None/empty defaults
                if not consumes_caffeine:
                    caffeine_types_list = None  # Explicitly set to None
                    caffeine_servings = 0
                    caffeine_timing = ''
                else:
                    # Only process caffeine data if user actually consumes caffeine
                    caffeine_types_list = onboarding_data.get('caffeine_types', [])
                    caffeine_servings = int(onboarding_data.get('caffeine_servings', 0))
                    caffeine_timing = onboarding_data.get('caffeine_timing', '')
                
                profile_data = {
                    'email': user_email,  # Use 'email' to match model field name
                    'name': onboarding_data.get('name', ''),
                    'age': int(onboarding_data.get('age', 0)),
                    'academic_level': onboarding_data.get('academic_level', ''),
                    
                    # Section 2: Rhythms
                    'alert_time': onboarding_data.get('alert_time', ''),
                    'sleep_hours': float(onboarding_data.get('sleep_hours', 7)),
                    'sleep_quality': onboarding_data.get('sleep_quality', ''),
                    
                    # Section 3: Caffeine
                    'consumes_caffeine': consumes_caffeine,
                    'caffeine_types': ', '.join(caffeine_types_list) if caffeine_types_list and isinstance(caffeine_types_list, list) else '',
                    'caffeine_servings': caffeine_servings,
                    'caffeine_timing': caffeine_timing,
                    
                    # Section 4: Styles
                    'learning_style': onboarding_data.get('learning_style', ''),
                    'study_subjects': ', '.join(study_subjects_list) if isinstance(study_subjects_list, list) else str(study_subjects_list),
                    
                    # Section 5: Habits
                    'session_length': onboarding_data.get('session_length', ''),
                    'takes_breaks': onboarding_data.get('takes_breaks', ''),
                    'study_time_of_day': onboarding_data.get('study_time_of_day', ''),
                    'procrastination_level': int(onboarding_data.get('procrastination_level', 0)),
                    
                    # Section 6 & 7: Environment & Distractions
                    'study_location': onboarding_data.get('study_location', ''),
                    'sound_environment': onboarding_data.get('sound_environment', ''),
                    'lighting_preference': onboarding_data.get('lighting_preference', ''),
                    'phone_location': onboarding_data.get('phone_location', ''),
                    'distractions': ', '.join(distractions_list) if isinstance(distractions_list, list) else str(distractions_list),
                    
                    # Section 8 & 9: Lifestyle & Health
                    'exercise_frequency': onboarding_data.get('exercise_frequency', ''),
                    'eating_timing': onboarding_data.get('eating_timing', ''),
                    'health_conditions': onboarding_data.get('health_conditions', ''),
                    
                    # Section 10: Goals
                    'main_goals': onboarding_data.get('main_goals', ''),
                    'study_effectiveness': int(onboarding_data.get('study_effectiveness', 0))
                }
                
                # Save or update UserProfile in database
                profile, created = UserProfile.objects.update_or_create(
                    email=user_email,  # Use 'email' to match model field name
                    defaults=profile_data
                )
                
                # Clear session data
                if 'onboarding_data' in request.session:
                    del request.session['onboarding_data']
                request.session.modified = True
                
            except Exception as e:
                pass
            
            return redirect('/dashboard/')
    
    # Handle GET request
    current_step = int(request.GET.get('step', 1))
    progress_percentage = (current_step / 10) * 100
    
    # Get saved data from session
    onboarding_data = request.session.get('onboarding_data', {})
    
    # Get current section data
    current_section = SECTIONS.get(current_step, SECTIONS[1])
    
    # Add saved data to context for template access
    context = {
        'current_step': current_step,
        'progress_percentage': int(progress_percentage),
        'section': current_section,
        'user': request.user if request.user.is_authenticated else None,
        'session_history': [],  # Would be populated from database
        'focus_percentage': None,  # Would be calculated from current session
        'saved_data': onboarding_data,
    }
    
    return render(request, 'onboarding.html', context)

@csrf_exempt
def prediction_view(request):
    """Generates the live preditictions for EEG data streamed from the device"""
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=400)
    #Live prediction integration
    try:
        data = json.loads(request.body)
        rows = data.get('rows', [])

        if not rows:
            return JsonResponse({'error': 'No rows provided'}, status = 400)

        user_profile = UserProfile.objects.get(email=user_email)

        result = MODEL_SERVICE.run(rows)
        if result.get('ok') == False:
            return JsonResponse(result, status=400)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status = 500)

@csrf_exempt
def start_realtime_eeg_view(request):
    """Start real-time EEG inference with per-second streaming"""
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=400)
    
    try:
        # Get duration and session_id from request
        data = json.loads(request.body) if request.body else {}
        duration = int(data.get('duration', 1))
        session_id = data.get('session_id')  # Get session_id from pre-session check-in for linking
        
        # Trigger real-time Celery task with session_id
        task = run_live_inference_streaming.delay(user_email, duration, session_id)
        
        # Store task ID in session
        request.session['current_eeg_task_id'] = task.id
        request.session['realtime_session_active'] = True
        request.session.modified = True

        
        return JsonResponse({
            'ok': True,
            'message': 'Real-time EEG inference started',
            'task_id': task.id,
            'duration_minutes': duration,
            'session_id': session_id,
            'session_type': 'realtime',
            'status': 'initializing'
        })
        
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@csrf_exempt
def stop_realtime_eeg_view(request):
    """Stop real-time EEG session and get final summary"""
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    try:
        request.session['realtime_session_active'] = False
        request.session.modified = True

        task_id = request.session.get('current_eeg_task_id')
        if not task_id:
            return JsonResponse({'error': 'No active EEG task found'}, status=400)

        task_result = get_task_status(task_id)

        if task_result['status'] == 'SUCCESS':
            result       = task_result['result']
            session_id   = result.get('session_id')
            final_summary = result.get('final_summary', {})

            # ── READ recommendation from cache (generated in tasks_realtime.py) ──
            from django.core.cache import cache
            cache_key       = f"recommendation_{user_email}"
            cached_rec      = cache.get(cache_key)
            recommendation  = cached_rec.get('text') if cached_rec else None

            # Save to Django session for recommendation page
            request.session['latest_recommendation'] = recommendation
            request.session['latest_session_id']     = session_id
            request.session.modified = True

            return JsonResponse({
                'ok'             : True,
                'status'         : 'completed',
                'session_id'     : session_id,
                'final_summary'  : final_summary,
                'recommendation' : recommendation,
                'csv_file_path'  : result.get('csv_file_path'),
                'duration_minutes': result.get('duration_minutes')
            })

        elif task_result['status'] == 'FAILURE':
            return JsonResponse({
                'ok'     : False,
                'status' : 'error',
                'message': 'EEG inference task failed',
                'error'  : str(task_result.get('result', 'Unknown error'))
            })
        else:
            return JsonResponse({
                'ok'    : True,
                'status': task_result['status'],
                'message': 'EEG inference still in progress',
                'task_id': task_id
            })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def get_realtime_eeg_status_view(request):
    """Get real-time status and current focus data"""
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    
    try:
        # Get task ID from session
        task_id = request.session.get('current_eeg_task_id')
        if not task_id:
            return JsonResponse({'error': 'No active EEG task found'}, status=400)
        
        # Check if realtime session is active
        is_active = request.session.get('realtime_session_active', False)
        
        # Get task status
        task_result = get_task_status(task_id)
        # Get live prediction data from cache if session is active
        current_state = None
        confidence = None
        focus_score = None
        cached_data = None
        
        # Check if task is PENDING (running) - this is more reliable than session flag
        if task_result['status'] == 'PENDING':
            try:
                from django.core.cache import cache
                LIVE_CACHE_KEY = f"live_eeg_stream_{user_email}"
                cached_data = cache.get(LIVE_CACHE_KEY)
                
                # Cache logging reduced for cleaner output
                if cached_data:
                    current_state = cached_data.get('state')
                    confidence = cached_data.get('confidence')
                    focus_score = cached_data.get('focus_score')
            except Exception as e:
                pass
        # Determine session status for frontend
        if task_result['status'] == 'PENDING':
            # Check if we have cache data (session is actually running)
            if cached_data:
                session_status = cached_data.get('status', 'active')
            else:
                session_status = 'initializing'  # Loading brainwaves...
        elif task_result['status'] == 'SUCCESS':
            session_status = 'completed'
        else:
            session_status = 'idle'
        
        return JsonResponse({
            'ok': True,
            'task_id': task_id,
            'status': session_status,
            'is_realtime_active': is_active,
            'current_state': current_state,
            'confidence': round(confidence * 100, 1) if confidence else None,
            'focus_score': focus_score,
            'wave_data': cached_data.get('waves') if cached_data else None,
            'result': task_result.get('result') if task_result['status'] == 'SUCCESS' else None
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def get_latest_eeg_state_view(request):
    """Lightning-fast cache-only view for live brain data"""
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    
    try:
        from django.core.cache import cache
        cache_key = f"live_eeg_stream_{user_email}"
        data = cache.get(cache_key)
        
        if not data:
            return JsonResponse({'ok': False, 'status': 'idle', 'message': 'Waiting for worker...', 'cache_key': cache_key})
        
        # Add ok: true to successful responses for JavaScript compatibility
        data['ok'] = True
        return JsonResponse(data)
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def test_cache_view(request):
    """Test cache functionality manually"""
    try:
        from django.core.cache import cache
        from django.utils import timezone
        
        user_email = request.session.get('user_email', 'test@example.com')
        cache_key = f"live_eeg_state_{user_email}"
        
        # Create test data
        test_data = {
            'ok': True,
            'status': 'active',
            'state': 'TEST_CONCENTRATING',
            'confidence': 95.5,
            'focus_score': 88.0,
            'waves': {
                'delta': 45.2,
                'theta': 30.1,
                'alpha': 25.8,
                'beta': 55.3,
                'gamma': 15.7
            },
            'last_updated': timezone.now().strftime("%I:%M:%S %p")
        }
        
        # Write to cache
        cache.set(cache_key, test_data, timeout=10)
        
        # Try to retrieve it
        retrieved_data = cache.get(cache_key)
        # Return the retrieved data
        if retrieved_data:
            return JsonResponse({
                'success': True,
                'message': 'Cache test successful',
                'cache_key': cache_key,
                'data': retrieved_data
            })
        else:
            return JsonResponse({
                'success': False,
                'message': 'Cache test failed - no data retrieved',
                'cache_key': cache_key
            })
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

EEGSERVICE = EEGService()  
@csrf_exempt
def start_live_eeg_view(request):
    """Start EEG inference as a Celery task"""
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=400)
    
    task_id = request.session.get('current_eeg_task_id')
    #cache.set(f"stop eeg task{task_id}", False, timeout=60*60)
   

    try:
        # Get duration from request (default to 1 minute)
        data = json.loads(request.body) if request.body else {}
        duration = int(data.get('duration', 1))
        
        # Trigger Celery task
        task = run_live_inference.delay(user_email, duration)
        
        # Store task ID in session for status checking
        request.session['current_eeg_task_id'] = task.id
        request.session.modified = True
        
        return JsonResponse({
            'ok': True,
            'message': 'EEG inference task started',
            'task_id': task.id,
            'duration_minutes': duration,
            'status': 'processing'
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def stop_live_eeg_view(request):
    """Check EEG inference task status and return results"""
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    
    try:
        # Get task ID from session
        task_id = request.session.get('current_eeg_task_id')
        if not task_id:
            return JsonResponse({'error': 'No active EEG task found'}, status=400)
        
        
        # Check task status
        task_result = get_task_status(task_id)
        
        if task_result['status'] == 'SUCCESS':
            # Task completed successfully
            result = task_result['result']
            return JsonResponse({
                'ok': True,
                'status': 'completed',
                'session_id': result.get('session_id'),
                'final_result': result.get('final_result'),
                'raw_data_path': result.get('raw_data_path'),
                'duration_minutes': result.get('duration_minutes')
            })
        elif task_result['status'] == 'PENDING':
            # Task still running
            result = task_result['result']
            #cache.set(f"stop eeg task{task_id}", True, timeout=60*60)  # Reset stop flag for next check
            return JsonResponse({
                'ok': True,
                'status': 'processing',
                'message': 'EEG inference still in progress',
                'task_id': task_id,
            })
        elif task_result['status'] == 'FAILURE':
            # Task failed
            return JsonResponse({
                'ok': False,
                'status': 'error',
                'message': 'EEG inference task failed',
                'error': str(task_result.get('result', 'Unknown error'))
            })
        else:
            return JsonResponse({
                'ok': True,
                'status': task_result['status'],
                'task_id': task_id
            })
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def eeg_task_status_view(request):
    """Check the status of an ongoing EEG inference task"""
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    
    try:
        # Get task ID from session or request parameter
        task_id = request.GET.get('task_id') or request.session.get('current_eeg_task_id')
        if not task_id:
            return JsonResponse({'error': 'No task ID provided'}, status=400)
        
        # Check task status
        task_result = get_task_status(task_id)
        
        return JsonResponse({
            'ok': True,
            'task_id': task_id,
            'status': task_result['status'],
            'result': task_result['result'] if task_result['status'] == 'SUCCESS' else None
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    

def upload_csv_view(request):
    """Upload csv online and recvieve predictions for it"""

    if request.method == 'POST' and request.FILES.get('csv_file'):
        file = request.FILES['csv_file']
        try:
            rows = []
            dfile = file.read().decode('utf-8').splitlines()
            reader = csv.reader(dfile)
            next(reader)

            for row in reader:
                rows.append([float(x) for x in row])
            result = MODEL_SERVICE.run(rows)

            return JsonResponse(result)
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return render(request, 'upload_csv.html')

def test_csv():
    rows = []
    with open(r"C:\Users\binom\OneDrive\Desktop\KeystoneProject\MCS_Capstone\dataset\our_data\areeba_new\areeba_concentrating_3min.csv") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            rows.append([float(x) for x in row])
    result = MODEL_SERVICE.run(rows)

def recommendation_view(request):
    """Fetch the latest recommendation for the user"""
    user_email = request.session.get('user_email')
    if not user_email:
        return redirect('/')

    # 1. Try to get from Cache first (for immediate results after a session)
    cache_key = f"recommendation_{user_email}"
    cached_data = cache.get(cache_key)
    
    recommendation_text = None
    session_id = None
    
    if cached_data:
        recommendation_text = cached_data.get('text')
        session_id = cached_data.get('session_id')
    else:
        # 2. Fallback: Get the most recent recommendation from the database
        latest_rec = Recommendation.objects.filter(user__email=user_email).order_by('-created_at').first()
        if latest_rec:
            recommendation_text = latest_rec.message
            session_id = latest_rec.session.session_id if latest_rec.session else None

    # 3. Get past recommendations for the history section
    past_recommendations = Recommendation.objects.filter(user__email=user_email).order_by('-created_at')[1:6]

    context = {
        'recommendation': recommendation_text,
        'session_id': session_id,
        'past_recommendations': past_recommendations
    }
    return render(request, 'recommendation.html', context)

def end_session(request):
    """End the current EEG session by creating a stop signal file"""
    if request.method == 'POST':
        user_email = request.session.get('user_email')
        if not user_email:
            return JsonResponse({'error': 'Unauthorized'}, status=400)
        
        try:
            # Create stop signal file for the user
            import os
            stop_signal_path = f"/tmp/stop_session_{user_email}.flag"
            
            # Create the stop signal file
            with open(stop_signal_path, 'w') as f:
                f.write('stop')
            
            return JsonResponse({
                'success': True,
                'message': 'Session termination signal sent'
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


def get_latest_recommendation_api(request):
    user_email = request.session.get('user_email')
    cache_key = f"recommendation_{user_email}"
    cached_data = cache.get(cache_key)
    
    if cached_data:
        return JsonResponse({'recommendation': cached_data['text']})
    
    latest_rec = Recommendation.objects.filter(user__email=user_email).order_by('-created_at').first()
    if latest_rec:
        return JsonResponse({'recommendation': latest_rec.message})
        
    return JsonResponse({'recommendation': None})


def focus_track_history(request):
    """Focus Track History page with SessionSummary data integration"""
    user_email = request.session.get('user_email')
    if not user_email:
        return redirect('/')
    
    try:
        user_profile = UserProfile.objects.get(email=user_email)
    except UserProfile.DoesNotExist:
        return redirect('/onboarding/')
    
    # Get filter parameter (default to "5" for Top 5 Sessions)
    filter_value = request.GET.get('filter', '5')
    
    # Get sessions based on filter
    if filter_value == '5':
        # Top 5 Sessions
        recent_sessions = SessionSummary.objects.filter(
            user=user_profile
        ).order_by('-start_time')[:5]
    elif filter_value == '7':
        # Last 7 Days
        from datetime import datetime, timedelta
        seven_days_ago = timezone.now() - timedelta(days=7)
        recent_sessions = SessionSummary.objects.filter(
            user=user_profile,
            start_time__gte=seven_days_ago
        ).order_by('-start_time')
    elif filter_value == '30':
        # Last 30 Days
        from datetime import datetime, timedelta
        thirty_days_ago = timezone.now() - timedelta(days=30)
        recent_sessions = SessionSummary.objects.filter(
            user=user_profile,
            start_time__gte=thirty_days_ago
        ).order_by('-start_time')
    else:
        # Default to Top 5
        recent_sessions = SessionSummary.objects.filter(
            user=user_profile
        ).order_by('-start_time')[:5]
    
    # Calculate aggregate statistics
    aggregate_stats = calculate_aggregate_stats(recent_sessions)
    
    # Prepare session data for template
    sessions_data = []
    for session in recent_sessions:
        # Calculate state percentages
        total_time = session.concentrating_seconds + session.neutral_seconds + session.relaxed_seconds
        if total_time > 0:
            concentrated_pct = round((session.concentrating_seconds / total_time) * 100, 1)
            neutral_pct = round((session.neutral_seconds / total_time) * 100, 1)
            relaxed_pct = round((session.relaxed_seconds / total_time) * 100, 1)
        else:
            concentrated_pct = neutral_pct = relaxed_pct = 0
        
        # Get recommendations for this session
        recommendations = Recommendation.objects.filter(
            user=user_profile,
            session=session
        ).order_by('-created_at')
        
        # Try to get session_name from PreSessionCheckIn
        try:
            presession = PreSessionCheckIn.objects.filter(
                user=user_profile,
                session_id=session.session_id
            ).first()
            # Use session_name if available, otherwise fall back to task_id
            display_name = presession.session_name if presession and presession.session_name else (
                session.task_id if session.task_id else session.session_id.replace('_', ' ').title()
            )
        except Exception as e:
            display_name = session.task_id if session.task_id else session.session_id.replace('_', ' ').title()
        
        session_data = {
            'id': session.id,
            'session_id': session.session_id,
            'name': display_name,
            'date': timezone.localtime(session.start_time).date(),
            'time': timezone.localtime(session.start_time).strftime('%H:%M'),
            'duration': round(session.total_duration_seconds / 60, 1),  # Convert to minutes
            'focus_score': round(session.average_focus_score * 10, 1),  # Convert to 0-10 scale
            'peak_focus': round(session.peak_focus_score * 10, 1),
            'focus_streak': session.longest_focus_streak,
            'states': {
                'concentrated': concentrated_pct,
                'neutral': neutral_pct,
                'relaxed': relaxed_pct
            },
            'start_time': session.start_time,
            'recommendations': recommendations
        }
        sessions_data.append(session_data)
    
    context = {
        'user_email': user_email,
        'aggregate_stats': aggregate_stats,
        'recent_sessions': sessions_data,
        'current_filter': filter_value
    }
    
    return render(request, 'focus_track_history.html', context)


def calculate_aggregate_stats(sessions):
    """Calculate aggregate statistics for the filtered sessions"""
    total_minutes = 0
    total_focus_score = 0
    session_count = 0
    
    for session in sessions:
        total_minutes += session.total_duration_seconds / 60
        total_focus_score += session.average_focus_score
        session_count += 1
    
    avg_focus = 0
    if session_count > 0:
        avg_focus = round((total_focus_score / session_count) * 10, 1)  # Convert from 2-10 scale to 0-100%
    
    return {
        'total_minutes': round(total_minutes, 1),
        'avg_focus': avg_focus,
        'session_count': session_count
    }


@csrf_exempt
def presession_checkin_view(request):
    """Handle pre-session check-in questionnaire submission"""
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    
    if request.method == 'POST':
        try:
            from .models import UserProfile, PreSessionCheckIn
            from datetime import datetime
            from django.utils import timezone
            
            data = json.loads(request.body)
            
            # Get user profile
            user_profile = UserProfile.objects.get(email=user_email)
            
            # Generate session_id if not provided
            session_id = data.get('session_id') or f"{user_email}_{timezone.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Parse deadline if provided
            assignment_deadline = None
            deadline_val = data.get('assignment_deadline')
            if deadline_val and deadline_val != '':
                from django.utils import timezone
                now = timezone.now()
                if deadline_val == 'today':
                    assignment_deadline = now.replace(hour=23, minute=59, second=59)
                elif deadline_val == 'tomorrow':
                    assignment_deadline = now + timezone.timedelta(days=1)
                    assignment_deadline = assignment_deadline.replace(hour=23, minute=59, second=59)
                elif deadline_val == 'this_week':
                    assignment_deadline = now + timezone.timedelta(days=7)
                elif deadline_val == 'next_week':
                    assignment_deadline = now + timezone.timedelta(days=14)
            
            # Map mood emoji to model choice
            mood_mapping = {
                'Happy': 'Happy',
                'Calm': 'Calm',
                'Anxious': 'Anxious',
                'Overwhelmed': 'Stressed',
                'Motivated': 'Focused',
                'Tired': 'Tired'
            }
            mood_emoji = mood_mapping.get(data.get('mood'), 'Neutral')
            
            # Map subject to model choice
            subject_mapping = {
                'Math/Problem-solving': 'Math',
                'Reading/Comprehension': 'Reading',
                'Writing/Essay': 'Writing',
                'Note-taking': 'Studying',
                'Memorization': 'Studying',
                'Creative work': 'Creative Work',
                'Coding/Technical': 'Coding',
                'Other': 'Other'
            }
            subject_task = subject_mapping.get(data.get('subject'), 'Studying')
            
            # Map task length to model choice
            task_length_mapping = {
                '15-30 minutes': '15-30m',
                '30-60 minutes': '30-60m',
                '1-2 hours': '1-2h',
                '2+ hours': '2h+'
            }
            estimated_length = task_length_mapping.get(data.get('tasklength'), '30-60m')
            
            # Map time since meal to model choice
            meal_mapping = {
                '<1 hour': '<1h',
                '1-2 hours': '1-2h',
                '2-4 hours': '2-4h',
                '4+ hours': '4h+'
            }
            time_since_meal = meal_mapping.get(data.get('meal'), '1-2h')
            
            # Map caffeine to model choice
            caffeine_mapping = {
                'None': 'None',
                '1 cup': '1 cup',
                '2 cups': '2 cups',
                '3-5 cups': '3-5 cups'
            }
            caffeine_intake = caffeine_mapping.get(data.get('caffeine'), 'None')
            
            # Map time since waking to model choice
            wake_mapping = {
                '<1 hour': '<1h',
                '1-3 hours': '1-3h',
                '3-6 hours': '3-6h',
                '6+ hours': '6h+'
            }
            time_since_waking = wake_mapping.get(data.get('wake'), '1-3h')
            
            # Map physical activity to model choice
            activity_mapping = {
                'None': 'None',
                'Light (walk)': 'Light',
                'Moderate (jog)': 'Moderate',
                'Intense (workout)': 'Intense'
            }
            physical_activity = activity_mapping.get(data.get('activity'), 'None')
            
            # Create PreSessionCheckIn record
            checkin = PreSessionCheckIn.objects.create(
                user=user_profile,
                session_id=session_id,
                session_name=data.get('session_name', ''),
                subject_task=subject_task,
                task_difficulty=int(data.get('difficulty', 5)),
                estimated_length=estimated_length,
                assignment_deadline=assignment_deadline,
                session_goal=data.get('goal', ''),
                energy_level=int(data.get('energy', 5)),
                mood_emoji=mood_emoji,
                stress_level=int(data.get('stress', 5)),
                time_since_meal=time_since_meal,
                caffeine_intake=caffeine_intake,
                time_since_waking=time_since_waking,
                physical_activity=physical_activity,
                current_noise=data.get('noise', ''),
                lighting_conditions=data.get('lighting', ''),
                study_method=data.get('method', ''),
                current_location=data.get('location', '')
            )
            
            return JsonResponse({
                'ok': True,
                'session_id': session_id,
                'checkin_id': str(checkin.check_in_id),
                'message': 'Pre-session check-in saved successfully'
            })
            
        except Exception as e:
            pass
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)