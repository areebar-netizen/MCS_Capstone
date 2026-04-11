from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import datetime, timedelta
import random
from secondBrain_App.models import UserProfile, SessionSummary


class Command(BaseCommand):
    help = 'Inject demo SessionSummary data for testing the calendar'

    def handle(self, *args, **options):
        # Create or get user profile for sachijs@uci.edu
        email = 'sachijs@uci.edu'
        
        try:
            user_profile = UserProfile.objects.get(email=email)
            self.stdout.write(f'Found existing user: {email}')
        except UserProfile.DoesNotExist:
            # Create user profile with minimal required fields
            user_profile = UserProfile.objects.create(
                email=email,
                name='Sachi',
                age=21,
                academic_level='Undergraduate',
                alert_time='Morning',
                sleep_hours=7.0,
                sleep_quality='Good',
                consumes_caffeine=True,
                caffeine_types='Coffee',
                caffeine_servings=2,
                caffeine_timing='Morning',
                learning_style='Visual',
                study_subjects='Computer Science',
                session_length='30-45 minutes',
                takes_breaks='Yes',
                study_time_of_day='Morning',
                procrastination_level=5,
                study_location='Home Office',
                sound_environment='Quiet',
                lighting_preference='Natural daylight',
                phone_location='In another room',
                distractions='Social media notifications',
                exercise_frequency='3-5 times per week',
                eating_timing='1-2 hours',
                health_conditions='None',
                main_goals='Improve focus and concentration',
                study_effectiveness=7
            )
            self.stdout.write(f'Created new user: {email}')

        # Clear existing session data for this user
        SessionSummary.objects.filter(user=user_profile).delete()
        self.stdout.write('Cleared existing session data')

        # Generate demo data for the current month
        now = timezone.now()
        current_year = now.year
        current_month = now.month
        
        # Generate sessions for various days in the current month
        demo_sessions = []
        
        # Create sessions for about 15-20 days in the month
        days_with_sessions = random.sample(range(1, 28), min(18, 27))  # Random 18 days
        
        for day in days_with_sessions:
            # Random time during the day
            hour = random.choice([9, 10, 11, 14, 15, 16, 19, 20])
            minute = random.choice([0, 15, 30, 45])
            
            start_time = datetime(current_year, current_month, day, hour, minute)
            
            # Random duration between 20-90 minutes
            duration_minutes = random.randint(20, 90)
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            # Random focus scores (1.0 to 3.0 scale)
            avg_focus_score = round(random.uniform(1.2, 2.9), 2)
            peak_focus_score = round(avg_focus_score + random.uniform(0.1, 0.5), 2)
            
            # Calculate state times based on focus level
            total_seconds = duration_minutes * 60
            if avg_focus_score >= 2.5:  # High focus
                concentrating_pct = random.uniform(0.6, 0.8)
            elif avg_focus_score >= 2.0:  # Medium focus
                concentrating_pct = random.uniform(0.4, 0.6)
            else:  # Low focus
                concentrating_pct = random.uniform(0.2, 0.4)
            
            concentrating_seconds = total_seconds * concentrating_pct
            relaxed_seconds = total_seconds * random.uniform(0.1, 0.3)
            neutral_seconds = total_seconds - concentrating_seconds - relaxed_seconds
            
            demo_sessions.append(SessionSummary(
                session_id=f'demo_session_{day}_{hour}{minute}',
                user=user_profile,
                task_id=f'task_{day}',
                csv_file_path=f'/demo_data/session_{day}.csv',
                start_time=start_time,
                end_time=end_time,
                total_duration_seconds=total_seconds,
                average_focus_score=avg_focus_score,
                peak_focus_score=peak_focus_score,
                relaxed_seconds=relaxed_seconds,
                neutral_seconds=neutral_seconds,
                concentrating_seconds=concentrating_seconds,
                data_points_count=random.randint(100, 500),
                longest_focus_streak=random.uniform(60, 300),  # 1-5 minutes
                focus_latency=random.uniform(30, 180),  # 30 seconds to 3 minutes
                state_switch_count=random.randint(5, 25),
                avg_confidence=random.uniform(0.7, 0.95)
            ))

        # Bulk create all sessions
        SessionSummary.objects.bulk_create(demo_sessions)
        
        session_count = len(demo_sessions)
        avg_focus = sum(s.average_focus_score for s in demo_sessions) / session_count
        avg_focus_pct = ((avg_focus - 1.0) / 2.0) * 100
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully created {session_count} demo sessions for {email}\n'
                f'Average focus score: {avg_focus:.2f} ({avg_focus_pct:.1f}%)\n'
                f'Sessions spread across {len(days_with_sessions)} days in {current_month}/{current_year}'
            )
        )
