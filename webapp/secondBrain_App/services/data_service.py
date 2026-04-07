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
    
    def get_recent_sessions(self, limit=10):
        """
        Get recent focus sessions for the user
        For now, returns mock data since we don't have a sessions table yet
        """
        # TODO: Replace with actual database queries when sessions table is available
        return [
            {
                'id': 1,
                'name': 'Morning Study Session',
                'date': timezone.now().date(),
                'time': '9:30 AM',
                'duration': 45,
                'focus_score': 8.2,
                'states': {'concentrated': 65, 'neutral': 25, 'relaxed': 10},
                'created_at': timezone.now() - timedelta(hours=6)
            },
            {
                'id': 2,
                'name': 'Afternoon Review',
                'date': timezone.now().date(),
                'time': '2:15 PM',
                'duration': 30,
                'focus_score': 6.8,
                'states': {'concentrated': 45, 'neutral': 40, 'relaxed': 15},
                'created_at': timezone.now() - timedelta(hours=4)
            },
            {
                'id': 3,
                'name': 'Evening Practice',
                'date': timezone.now().date() - timedelta(days=1),
                'time': '7:45 PM',
                'duration': 60,
                'focus_score': 7.5,
                'states': {'concentrated': 55, 'neutral': 30, 'recent_sessions': 15},
                'created_at': timezone.now() - timedelta(days=1, hours=2)
            }
        ][:limit]
    
    def get_session_average_stats(self):
        """
        Calculate average statistics from recent sessions
        """
        sessions = self.get_recent_sessions()
        if not sessions:
            return {
                'avg_focus': 0,
                'total_sessions': 0,
                'avg_duration': 0,
                'focus_trend': 'stable'
            }
        
        avg_focus = sum(s['focus_score'] for s in sessions) / len(sessions)
        avg_duration = sum(s['duration'] for s in sessions) / len(sessions)
        
        # Simple trend calculation
        if len(sessions) >= 2:
            recent_avg = sum(s['focus_score'] for s in sessions[:3]) / min(3, len(sessions))
            older_avg = sum(s['focus_score'] for s in sessions[-3:]) / min(3, len(sessions))
            trend = 'improving' if recent_avg > older_avg else 'declining' if recent_avg < older_avg else 'stable'
        else:
            trend = 'stable'
        
        return {
            'avg_focus': round(avg_focus, 1),
            'total_sessions': len(sessions),
            'avg_duration': round(avg_duration),
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
        Get calendar data for a specific month/year
        """
        if year is None:
            year = timezone.now().year
        if month is None:
            month = timezone.now().month
        
        # TODO: Replace with actual session data from database
        # For now, generate sample calendar data
        calendar_data = []
        
        # Get first day of month
        first_day = datetime(year, month, 1)
        
        # Get number of days in month
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        days_in_month = (next_month - first_day).days
        
        # Generate sample data for each day
        import random
        for day in range(1, days_in_month + 1):
            # Random focus score between 3 and 10
            focus_score = random.uniform(3.0, 10.0)
            
            # Determine emoji based on focus score
            if focus_score >= 7.5:
                emoji = '😊'  # Excellent
            elif focus_score >= 6.0:
                emoji = '😄'  # Great
            elif focus_score >= 4.5:
                emoji = '🙂'  # Good
            elif focus_score >= 3.0:
                emoji = '😐'  # Fair
            else:
                emoji = '🙁'  # Poor
            
            calendar_data.append({
                'day': day,
                'focus_score': round(focus_score, 1),
                'emoji': emoji,
                'has_session': random.choice([True, False])  # Random session presence
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
