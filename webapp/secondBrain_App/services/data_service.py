"""
Data Service Layer for Focus Track Components
Handles database queries and data processing for focus tracking features
"""
import os
import sys
from datetime import datetime, timedelta
from django.utils import timezone

# Add Django path to use Django models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'secondBrain.settings')

import django
django.setup()

from secondBrain_App.models import UserProfile, Recommendation

class FocusDataService:
    """Service class for handling focus track data operations"""
    
    def __init__(self, user_email):
        self.user_email = user_email
    
    def get_recent_sessions(self, limit=5, days=None):
        """
        Get recent focus sessions for the user from SessionSummary database
        Returns formatted session data with calculated state percentages
        
        Args:
            limit: Maximum number of sessions to return
            days: Filter to sessions within last N days (overrides limit)
        """
        from secondBrain_App.models import SessionSummary, UserProfile
        from datetime import timedelta
        
        try:
            user_profile = UserProfile.objects.get(email=self.user_email)
        except UserProfile.DoesNotExist:
            return []
        
        # Build query
        query = SessionSummary.objects.filter(user=user_profile)
        
        # Apply date filter if specified
        if days:
            cutoff_date = timezone.now() - timedelta(days=days)
            query = query.filter(start_time__gte=cutoff_date)
        
        # Fetch sessions ordered by start time
        sessions = query.order_by('-start_time')
        
        # Apply limit if not using days filter
        if not days:
            sessions = sessions[:limit]
        
        formatted_sessions = []
        for session in sessions:
            # Calculate state percentages based on duration
            total_duration = session.total_duration_seconds
            if total_duration > 0:
                concentrated_pct = (session.concentrating_seconds / total_duration) * 100
                neutral_pct = (session.neutral_seconds / total_duration) * 100
                relaxed_pct = (session.relaxed_seconds / total_duration) * 100
            else:
                concentrated_pct = neutral_pct = relaxed_pct = 0
            
            # Create session name from date/time if no task name
            session_name = f"Focus Session"
            if session.task_id:
                session_name = f"Task {session.task_id[:8]}"
            
            formatted_sessions.append({
                'id': session.session_id,
                'name': session_name,
                'date': session.start_time.date(),
                'time': session.start_time.strftime('%I:%M %p'),
                'duration': int(session.total_duration_seconds / 60),  # Convert to minutes
                'focus_score': session.average_focus_score,
                'states': {
                    'concentrated': round(concentrated_pct, 1),
                    'neutral': round(neutral_pct, 1),
                    'relaxed': round(relaxed_pct, 1)
                },
                'start_time': session.start_time,
                'end_time': session.end_time,
                'peak_focus': session.peak_focus_score,
                'focus_streak': session.longest_focus_streak,
                'created_at': session.created_at
            })
        
        return formatted_sessions
    
    def get_session_average_stats(self):
        """
        Calculate average statistics from real sessions for the current month
        """
        from secondBrain_App.models import SessionSummary
        
        # Get current month's sessions
        now = timezone.now()
        first_day = datetime(now.year, now.month, 1)
        if now.month == 12:
            next_month = datetime(now.year + 1, 1, 1)
        else:
            next_month = datetime(now.year, now.month + 1, 1)
        last_day = next_month - timedelta(days=1)
        
        # Query sessions for current month
        sessions = SessionSummary.objects.filter(
            session_date__gte=first_day.date(),
            session_date__lte=last_day.date()
        ).order_by('session_date')
        
        if not sessions:
            return {
                'avg_focus': 0,
                'total_sessions': 0,
                'active_days': 0,
                'focus_trend': 'stable'
            }
        
        # Calculate real statistics
        avg_focus_raw = sum(s.average_focus_score for s in sessions if s.average_focus_score > 0) / len(sessions)
        total_duration = sum(s.total_duration_seconds for s in sessions) / len(sessions)
        
        # Convert to 1-10 scale (multiply by 10)
        avg_focus = min(10, max(1, avg_focus_raw * 10))
        
        # Count unique active days
        active_days = sessions.values('session_date').distinct().count()
        
        # Simple trend calculation (compare last 3 sessions vs previous 3)
        trend = 'stable'
        if len(sessions) >= 6:
            recent_sessions = sessions[:3]
            older_sessions = sessions[3:6]
            
            recent_avg = sum(s.average_focus_score for s in recent_sessions if s.average_focus_score > 0) / len(recent_sessions)
            older_avg = sum(s.average_focus_score for s in older_sessions if s.average_focus_score > 0) / len(older_sessions)
            
            if recent_avg > older_avg + 0.001:  # Small threshold for 0-1 scale
                trend = 'improving'
            elif recent_avg < older_avg - 0.001:
                trend = 'declining'
            else:
                trend = 'stable'
        
        return {
            'avg_focus': round(avg_focus, 1),
            'total_sessions': len(sessions),
            'active_days': active_days,
            'focus_trend': trend
        }
    
    def get_brainwave_averages(self):
        """
        Calculate average brainwave data from recent sessions
        """
        # TODO: Replace with actual brainwave data when available
        return {
            'delta': 15,
            'theta': 28,
            'alpha': 42,
            'beta': 58,
            'gamma': 35
        }
    
    def get_recommendations(self, limit=5):
        """
        Get recent recommendations for the user
        """
        try:
            recommendations = Recommendation.objects.filter(user_id=self.user_email).order_by('-action_started_at')[:limit]
            return [
                {
                    'id': rec.recommendation_id,
                    'category': rec.recommendation_category,
                    'message': rec.message,
                    'stimulus': rec.stimulus_name,
                    'trigger': rec.trigger_reason,
                    'created_at': rec.action_started_at,
                    'session_id': rec.session_id
                }
                for rec in recommendations
            ]
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []
    
    def get_calendar_data(self, year=None, month=None):
        """
        Get calendar data for a specific month/year using actual session data
        """
        if year is None:
            year = timezone.now().year
        if month is None:
            month = timezone.now().month
        
        # Get actual session data from database
        from secondBrain_App.models import SessionSummary
        
        # Get first and last day of month
        first_day = datetime(year, month, 1)
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        last_day = next_month - timedelta(days=1)
        
        # Query sessions for this month
        sessions = SessionSummary.objects.filter(
            session_date__gte=first_day.date(),
            session_date__lte=last_day.date()
        ).order_by('session_date')
        
        # Create a mapping of date -> session data
        session_map = {}
        for session in sessions:
            date_key = session.session_date.day
            # Calculate focus percentage based on concentrating time
            if session.total_duration_seconds > 0:
                focus_percentage = (session.concentrating_seconds / session.total_duration_seconds) * 100
            else:
                focus_percentage = 0
            
            session_map[date_key] = {
                'focus_score': session.average_focus_score,
                'focus_percentage': focus_percentage,
                'session_id': session.session_id,
                'concentrating_seconds': session.concentrating_seconds,
                'total_seconds': session.total_duration_seconds
            }
        
        # Generate calendar data for each day
        calendar_data = []
        days_in_month = (next_month - first_day).days
        
        for day in range(1, days_in_month + 1):
            if day in session_map:
                # Real session data
                session_data = session_map[day]
                focus_percentage = session_data['focus_percentage']
                
                calendar_data.append({
                    'day': day,
                    'focus_score': round(session_data['focus_score'] * 100, 1),  # Convert to 1-100 scale
                    'focus_percentage': round(focus_percentage, 1),
                    'has_session': True,
                    'session_id': session_data['session_id']
                })
            else:
                # No session data
                calendar_data.append({
                    'day': day,
                    'focus_score': 0,
                    'focus_percentage': 0,
                    'emoji': None,
                    'has_session': False
                })
        
        return calendar_data
    
    def get_session_summary(self, session_id):
        """
        Get detailed summary for a specific session
        """
        # TODO: Implement when session tracking is available
        sessions = self.get_recent_sessions()
        for session in sessions:
            if session['id'] == session_id:
                return session
        return None