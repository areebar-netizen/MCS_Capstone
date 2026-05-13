#!/usr/bin/env python
"""
Simplified test script for the recommendation system.
This script creates minimal database records and tests the recommendation generation.
"""


import os
import sys
from pathlib import Path
import uuid
import django
from django.utils import timezone

# Setup Django environment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webapp'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'secondBrain.settings')


from secondBrain_App.models import UserProfile, SessionSummary, PreSessionCheckIn, Recommendation
from core_engine.recommendation import generate_recommendation_for_session


# Constants for default user profile data
DEFAULT_USER_DATA = {
    'name': 'Test User',
    'age': 25,
    'academic_level': 1,
    'learning_style': 0,
    'sound_environment': 0,
    'study_location': 0,
    'phone_location': 0,
    'distractions': 0,
    'exercise_frequency': 1,
    'eating_timing': 0,
    'sleep_hours': 7,
    'sleep_quality': 2,
    'caffeine_servings': 1,
    'procrastination_level': 2,
    'main_goals': '1,2,3',
    'study_subjects': '1,2',
    'session_length': 1,
    'study_time_of_day': 1,
    'alert_time': 1,
    'study_effectiveness': 5
}

# Constants for historical data creation
HISTORICAL_COUNT = 5
HISTORICAL_PREFIX = "HIST_SESS_"
DUMMY_CSV_PREFIX = "/tmp/hist_"


def create_test_user_profile(email: str) -> UserProfile:
    """Create or get user profile for testing."""
    user_profile, created = UserProfile.objects.get_or_create(
        email=email,
        defaults=DEFAULT_USER_DATA
    )
    return user_profile

def create_test_session_summary(user_profile: UserProfile, session_id: str, summary: dict) -> SessionSummary:
    """Create a minimal SessionSummary for testing."""
    csv_path = Path(f"/tmp/test_{session_id}.csv")

    session_summary = SessionSummary.objects.create(
        user=user_profile,
        session_id=session_id,
        task_id=session_id,
        csv_file_path=str(csv_path),
        start_time=timezone.now(),
        end_time=timezone.now(),
        session_date=timezone.now().date(),
        total_duration_seconds=(
            summary.get('concentrating_seconds', 900) +
            summary.get('neutral_seconds', 600) +
            summary.get('relaxed_seconds', 300)
        ),
        average_focus_score=summary.get('average_focus_score', 0.5),
        peak_focus_score=summary.get('average_focus_score', 0.5) + 0.1,
        relaxed_seconds=summary.get('relaxed_seconds', 300),
        neutral_seconds=summary.get('neutral_seconds', 600),
        concentrating_seconds=summary.get('concentrating_seconds', 900),
        data_points_count=60,
        longest_focus_streak=120,
        focus_latency=5,
        state_switch_count=10,
        avg_confidence=0.8
    )
    return session_summary

def create_test_pre_session_check_in(user_profile: UserProfile, session_id: str, checkin_data: dict) -> PreSessionCheckIn:
    """Create a PreSessionCheckIn for testing."""
    pre_check = PreSessionCheckIn.objects.create(
        user=user_profile,
        session_id=session_id,
        session_name=checkin_data.get('session_name', 'Test Session'),
        subject_task=checkin_data.get('subject_task', 'Other'),
        task_difficulty=checkin_data.get('task_difficulty', 5),
        estimated_length=checkin_data.get('estimated_length', '30-60m'),
        session_goal=checkin_data.get('session_goal', 'Test goal'),
        energy_level=checkin_data.get('energy_level', 5),
        mood_emoji=checkin_data.get('mood_emoji', 'Neutral'),
        stress_level=checkin_data.get('stress_level', 5),
        time_since_meal=checkin_data.get('time_since_meal', '1-2h'),
        lighting_conditions=checkin_data.get('lighting_conditions', 'Unknown'),
        current_noise=checkin_data.get('current_noise', 'Unknown'),
        study_method=checkin_data.get('study_method', 'Standard')
    )
    return pre_check

def create_historical_data(user_profile: UserProfile, count: int = HISTORICAL_COUNT) -> None:
    """Create dummy historical SessionSummary and Recommendation objects to trigger Phase 2 logic."""
    # Cleanup existing historical data first
    for i in range(count):
        session_id = f"{HISTORICAL_PREFIX}{i:03d}"
        SessionSummary.objects.filter(session_id=session_id).delete()
        PreSessionCheckIn.objects.filter(session_id=session_id).delete()
        Recommendation.objects.filter(session__session_id=session_id).delete()

    for i in range(count):
        session_id = f"{HISTORICAL_PREFIX}{i:03d}"

        # Create historical SessionSummary with poor focus
        session_summary = SessionSummary.objects.create(
            user=user_profile,
            session_id=session_id,
            task_id=session_id,
            csv_file_path=f"{DUMMY_CSV_PREFIX}{session_id}.csv",
            start_time=timezone.now() - timezone.timedelta(days=count - i),
            end_time=timezone.now() - timezone.timedelta(days=count - i),
            session_date=(timezone.now() - timezone.timedelta(days=count - i)).date(),
            total_duration_seconds=1800,
            average_focus_score=0.3,  # Poor focus to trigger failure pattern detection
            peak_focus_score=0.4,
            relaxed_seconds=600,
            neutral_seconds=900,
            concentrating_seconds=300,
            data_points_count=60,
            longest_focus_streak=60,
            focus_latency=10,
            state_switch_count=15,
            avg_confidence=0.6
        )

        # Create corresponding PreSessionCheckIn with Dim Lighting
        PreSessionCheckIn.objects.create(
            user=user_profile,
            session_id=session_id,
            session_name=f"Historical Session {i}",
            subject_task='Math',
            task_difficulty=7,
            estimated_length='30-60m',
            session_goal='Study math',
            energy_level=5,
            mood_emoji='Neutral',
            stress_level=5,
            time_since_meal='1-2h',
            lighting_conditions='Dim',  # Poor lighting to create failure pattern
            current_noise='Rock Music',
            study_method='Standard'
        )

        # Create Recommendation with structured JSON
        recommendation = Recommendation.objects.create(
            recommendation_id=str(uuid.uuid4()),
            user=user_profile,
            session=session_summary,
            inference_id=f"hist_inf_{i}",
            recommendation_category='general',
            stimulus_name='study_tip',
            trigger_reason='session_end',
            action_started_at=timezone.now() - timezone.timedelta(days=count - i),
            session_highlights=['Low focus session', 'Dim lighting environment'],
            user_level='Beginner',
            total_study_time_display='30m',
            best_session_length='15-30m',
            study_method_name='Standard',
            study_method_description='Standard study method',
            study_techniques=['Reading', 'Practice'],
            opt_noise='Rock Music',
            opt_lighting='Dim',
            opt_space='Bedroom',
            opt_time_slot='Evening',
            message={'historical': True, 'focus_score': 0.3}
        )

def cleanup_session(session_id):
    Recommendation.objects.filter(session__session_id=session_id).delete()
    PreSessionCheckIn.objects.filter(session_id=session_id).delete()
    SessionSummary.objects.filter(session_id=session_id).delete()
