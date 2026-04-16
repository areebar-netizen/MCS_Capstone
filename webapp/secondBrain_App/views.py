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

from .models import UserProfile, Recommendation, Prediction, SessionSummary
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
        now = datetime.now()
        current_month = now.month
        current_year = now.year
    
    # Query SessionSummary for current user and requested month/year
    sessions = SessionSummary.objects.filter(
        user=user_profile,
        start_time__year=current_year,
        start_time__month=current_month
    ).order_by('start_time')
    
    # Create session data mapping by day
    session_data_by_day = {}
    for session in sessions:
        day = session.start_time.day
        
        # Calculate focus percentage using the formula: ((Score - 1.0) / 2.0) * 100
        focus_percentage = ((session.average_focus_score - 1.0) / 2.0) * 100
        
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
        avg_focus = ((avg_focus - 1.0) / 2.0) * 100  # Convert to percentage
    
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
        now = datetime.now()
        current_month = now.month
        current_year = now.year
    
    # Query SessionSummary for current user and requested month/year
    sessions = SessionSummary.objects.filter(
        user=user_profile,
        start_time__year=current_year,
        start_time__month=current_month
    ).order_by('start_time')
    
    # Create session data mapping by day
    session_data_by_day = {}
    for session in sessions:
        day = session.start_time.day
        
        # Calculate focus percentage using the formula: ((Score - 1.0) / 2.0) * 100
        focus_percentage = ((session.average_focus_score - 1.0) / 2.0) * 100
        
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
        avg_focus_10_point = avg_focus * 3.33  # Scale to 10-point display
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
    
    now = datetime.now()
    
    if scale == 'week':
        # Get data for current week (last 7 days)
        start_date = now - timedelta(days=6)
        sessions = SessionSummary.objects.filter(
            user=user_profile,
            start_time__gte=start_date,
            start_time__lte=now
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
            day_sessions = [s for s in sessions if s['date'].date() == date.date()]
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
        # Get data for current month grouped by week
        sessions = SessionSummary.objects.filter(
            user=user_profile,
            start_time__year=now.year,
            start_time__month=now.month
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
            # Calculate week of month (0-3)
            week_start = session['week'].date()
            month_start = datetime(now.year, now.month, 1).date()
            week_index = (week_start - month_start).days // 7
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
            start_time__year=now.year,
            start_time__month=now.month
        ).values('start_time__date').distinct().count()
        daily_avg = total_seconds / active_days if active_days > 0 else 0
        
        # Find best week
        best_week_idx = max(range(4), key=lambda i: month_data[i]['seconds'])
        best_day = f'Week {best_week_idx + 1}' if month_data[best_week_idx]['seconds'] > 0 else 'None'
        
    else:  # year
        # Get data for current year grouped by month
        sessions = SessionSummary.objects.filter(
            user=user_profile,
            start_time__year=now.year
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
            month = session['month'].month
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
        year_start = date(now.year, 1, 1)
        days_passed = (now.date() - year_start).days + 1  # +1 to include current day
        
        # Use actual active days if they exist, otherwise use days passed
        active_days = SessionSummary.objects.filter(
            user=user_profile,
            start_time__year=now.year
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
            print(f"OTP email sent successfully to {email}")
        except Exception as e:
            print(f"Failed to send OTP email: {e}")
            # For development, you might want to still show the OTP in console
            print(f"DEBUG: OTP for {email} is {otp_code}")
            # In production, you might want to handle this differently
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
    
    # Get focus tracking data
    recent_sessions = data_service.get_recent_sessions()
    session_stats = data_service.get_session_average_stats()
    brainwave_data = data_service.get_brainwave_averages()
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
        now = datetime.now()
        current_month = now.month
        current_year = now.year
    
    # Query SessionSummary for current user and requested month/year
    from .models import SessionSummary
    sessions = SessionSummary.objects.filter(
        user=user_profile,
        start_time__year=current_year,
        start_time__month=current_month
    ).order_by('start_time')
    
    # Create session data mapping by day
    session_data_by_day = {}
    for session in sessions:
        day = session.start_time.day
        
        # Calculate focus percentage using the formula: ((Score - 1.0) / 2.0) * 100
        focus_percentage = ((session.average_focus_score - 1.0) / 2.0) * 100
        
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
        avg_focus_10_point = avg_focus * 3.33  # Scale to 10-point display
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
        'brainwave_data': brainwave_data,
        'recommendations': recommendations,
        'ai_recommendation': ai_recommendation_text,
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
                
                print(f"User profile {'created' if created else 'updated'} for {user_email}")
                
            except Exception as e:
                print(f"Error saving user profile: {e}")
                # Continue to dashboard even if save fails
            
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
        print(f'Error in prediction view: {e}')
        return JsonResponse({'error': str(e)}, status = 500)

@csrf_exempt
def start_realtime_eeg_view(request):
    """Start real-time EEG inference with per-second streaming"""
    print("Starting Real-time EEG Session")
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=400)
    
    try:
        # Get duration from request
        data = json.loads(request.body) if request.body else {}
        duration = int(data.get('duration', 1))
        
        # Trigger real-time Celery task
        task = run_live_inference_streaming.delay(user_email, duration)
        
        # Store task ID in session
        request.session['current_eeg_task_id'] = task.id
        request.session['realtime_session_active'] = True
        request.session.modified = True

        
        return JsonResponse({
            'ok': True,
            'message': 'Real-time EEG inference started',
            'task_id': task.id,
            'duration_minutes': duration,
            'session_type': 'realtime',
            'status': 'initializing'
        })

        
        
    except Exception as e:
        print(f'Error starting real-time EEG task: {e}')
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@csrf_exempt
def stop_realtime_eeg_view(request):
    """Stop real-time EEG session and get final summary"""
    print("Stopping Real-time EEG Session")
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
        print(f'Error stopping real-time EEG task: {e}')
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
        
        return JsonResponse({
            'ok': True,
            'task_id': task_id,
            'status': task_result['status'],
            'is_realtime_active': is_active,
            'result': task_result.get('result') if task_result['status'] == 'SUCCESS' else None
        })
        
    except Exception as e:
        print(f'Error getting real-time EEG status: {e}')
        return JsonResponse({'error': str(e)}, status=500)

EEGSERVICE = EEGService()  
@csrf_exempt
def start_live_eeg_view(request):
    """Start EEG inference as a Celery task"""
    print("Starting EEG inference task")
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
        print(f'Error starting EEG task: {e}')
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def stop_live_eeg_view(request):
    """Check EEG inference task status and return results"""
    print("Checking EEG task status")
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
        print(f'Error checking EEG task status: {e}')
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
        print(f'Error checking task status: {e}')
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
            print(f'Error processsing uploaded csv: {e}')
            return JsonResponse({'error': str(e)}, status=500)
    return render(request, 'upload_csv.html')

def test_csv():
    print("Running test_csv...")  # add this

    rows = []
    with open(r"C:\Users\binom\OneDrive\Desktop\KeystoneProject\MCS_Capstone\dataset\our_data\areeba_new\areeba_concentrating_3min.csv") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append([float(x) for x in row])

    result = MODEL_SERVICE.run(rows)
    print("RESULT:", result)

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
            session_id = latest_rec.session_id

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
            print(f'Error creating stop signal: {e}')
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