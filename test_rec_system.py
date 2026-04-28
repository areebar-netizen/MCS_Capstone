#!/usr/bin/env python
"""
Simplified test script for the recommendation system.
This script creates minimal database records and tests the recommendation generation.
"""

import os
import sys
import django

# Setup Django environment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webapp'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'secondBrain.settings')
django.setup()

from secondBrain_App.models import UserProfile, SessionSummary, PreSessionCheckIn, Recommendation
from core_engine.recommendation import generate_recommendation_for_session
from django.utils import timezone
import uuid


def create_test_user_profile(email):
    """Create or get user profile for testing"""
    user_profile, created = UserProfile.objects.get_or_create(
        email=email,
        defaults={
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
    )
    return user_profile


def create_test_session_summary(user_profile, session_id, summary):
    """Create a minimal SessionSummary for testing"""
    from pathlib import Path
    # Create a dummy CSV file path
    csv_path = Path(f"/tmp/test_{session_id}.csv")
    
    session_summary = SessionSummary.objects.create(
        user=user_profile,
        session_id=session_id,
        task_id=session_id,
        csv_file_path=str(csv_path),
        start_time=timezone.now(),
        end_time=timezone.now(),
        session_date=timezone.now().date(),
        total_duration_seconds=summary.get('concentrating_seconds', 900) + summary.get('neutral_seconds', 600) + summary.get('relaxed_seconds', 300),
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


def create_test_pre_session_check_in(user_profile, session_id, checkin_data):
    """Create a PreSessionCheckIn for testing"""
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


def create_historical_data(user_profile, count=5):
    """Create dummy historical SessionSummary and Recommendation objects to trigger Phase 2 logic"""
    # Cleanup existing historical data first
    for i in range(count):
        session_id = f"HIST_SESS_{i:03d}"
        SessionSummary.objects.filter(session_id=session_id).delete()
        PreSessionCheckIn.objects.filter(session_id=session_id).delete()
        Recommendation.objects.filter(session__session_id=session_id).delete()
    
    for i in range(count):
        session_id = f"HIST_SESS_{i:03d}"
        
        # Create historical SessionSummary with poor focus
        session_summary = SessionSummary.objects.create(
            user=user_profile,
            session_id=session_id,
            task_id=session_id,
            csv_file_path=f"/tmp/hist_{session_id}.csv",
            start_time=timezone.now() - timezone.timedelta(days=count-i),
            end_time=timezone.now() - timezone.timedelta(days=count-i),
            session_date=(timezone.now() - timezone.timedelta(days=count-i)).date(),
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
        import json
        recommendation = Recommendation.objects.create(
            recommendation_id=str(uuid.uuid4()),
            user=user_profile,
            session=session_summary,  # Pass the SessionSummary object, not the session_id string
            inference_id=f"hist_inf_{i}",
            recommendation_category='general',
            stimulus_name='study_tip',
            trigger_reason='session_end',
            action_started_at=timezone.now() - timezone.timedelta(days=count-i),
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
    
    print(f"[TEST] Created {count} historical sessions with Dim Lighting failure pattern")


# Mock Data for Scenario A: High Focus/High Gamma (Peak Performance)
mock_summary_high = {
    'average_focus_score': 0.85,
    'concentrating_seconds': 1800,
    'neutral_seconds': 300,
    'relaxed_seconds': 100,
    'beta_avg': 24.5,  # Filtered Average
    'gamma_avg': 48.2, # Filtered Average
    'alpha_avg': 10.1,
    'theta_avg': 5.5,
    # High-level inferences
    'neural_state': 'Focus',  # Beta 15-30 + Gamma > 30
    'signal_integrity': 'Clean',  # Max wave < 100
    'focus_depth': 'Deep Flow'  # Focus + Gamma > 40
}
mock_checkin_high = {
    'session_name': 'High Focus Session',
    'subject_task': 'UI Design',
    'task_difficulty': 6,
    'estimated_length': '30-60m',
    'session_goal': 'Complete design task',
    'energy_level': 8,
    'mood_emoji': 'Focused',
    'stress_level': 3,
    'time_since_meal': '1-2h',
    'lighting_conditions': 'Warm Natural Light',
    'current_noise': 'Jazz Music',
    'study_method': 'Flow State'
}

# Mock Data for Scenario B: Low Focus/High Theta (Distracted)
mock_summary_low = {
    'average_focus_score': 0.30,
    'concentrating_seconds': 400,
    'neutral_seconds': 1200,
    'relaxed_seconds': 600,
    'beta_avg': 12.5,
    'gamma_avg': 15.0,
    'alpha_avg': 8.0,
    'theta_avg': 18.5,  # High Theta = Drowsiness/Distraction
    # High-level inferences
    'neural_state': 'Distracted',  # High theta (>15) indicates distraction
    'signal_integrity': 'Clean',  # Max wave < 100
    'focus_depth': 'Surface Level'  # Not in Focus state
}
mock_checkin_low = {
    'session_name': 'Low Focus Session',
    'subject_task': 'Reading',
    'task_difficulty': 5,
    'estimated_length': '30-60m',
    'session_goal': 'Read textbook',
    'energy_level': 4,
    'mood_emoji': 'Tired',
    'stress_level': 6,
    'time_since_meal': '2-4h',
    'lighting_conditions': 'Warm Light',
    'current_noise': 'White Noise',
    'study_method': 'Standard'
}

# Mock Data for Scenario C: Math with low focus (Subject-Specific)
mock_summary_math = {
    'average_focus_score': 0.35,
    'concentrating_seconds': 500,
    'neutral_seconds': 700,
    'relaxed_seconds': 600,
    'beta_avg': 12.0,  # Low beta (low alertness)
    'gamma_avg': 25.0,
    'alpha_avg': 15.0,
    'theta_avg': 12.0,  # High theta (distraction)
    # High-level inferences
    'neural_state': 'Distracted',  # High theta (>15) indicates distraction
    'signal_integrity': 'Clean',  # Max wave < 100
    'focus_depth': 'Surface Level'  # Not in Focus state
}
mock_checkin_math = {
    'session_name': 'Math Session',
    'subject_task': 'Advanced Mathematics',  # Specific subject for testing
    'task_difficulty': 8,
    'estimated_length': '30-60m',
    'session_goal': 'Solve math problems',
    'energy_level': 5,
    'mood_emoji': 'Neutral',
    'stress_level': 5,
    'time_since_meal': '1-2h',
    'lighting_conditions': 'Dim',  # Poor lighting to test variable pivot
    'current_noise': 'Rock Music',  # Poor noise to test variable pivot
    'study_method': 'Standard'
}

# Mock Data for Scenario D: Artifact Overload (Noise Filtering Validation)
mock_summary_artifact = {
    'average_focus_score': 0.25,  # Low focus despite high raw waves
    'concentrating_seconds': 300,
    'neutral_seconds': 900,
    'relaxed_seconds': 600,
    'beta_avg': 45.0,  # High beta from muscle noise (jaw clenching)
    'gamma_avg': 65.0,  # High gamma from muscle noise (eye movement)
    'alpha_avg': 8.0,
    'theta_avg': 6.0,
    # High-level inferences
    'neural_state': 'Anxious',  # High beta (>30) indicates anxiety
    'signal_integrity': 'Artifact-Heavy',  # Max wave > 100
    'focus_depth': 'Surface Level'  # Not in Focus state
}
mock_checkin_artifact = {
    'session_name': 'Restless Session',
    'subject_task': 'General Study',
    'task_difficulty': 5,
    'estimated_length': '30-60m',
    'session_goal': 'Study session',
    'energy_level': 8,  # High energy but restless
    'mood_emoji': 'Restless',
    'stress_level': 7,
    'time_since_meal': '1-2h',
    'lighting_conditions': 'Bright Light',
    'current_noise': 'Silence',
    'study_method': 'Standard'
}

# Mock Data for Scenario E: Late Night Burnout (Circadian Rhythm)
mock_summary_burnout = {
    'average_focus_score': 0.40,  # Moderate focus but drowsy
    'concentrating_seconds': 600,
    'neutral_seconds': 600,
    'relaxed_seconds': 600,
    'beta_avg': 15.0,  # Moderate beta
    'gamma_avg': 20.0,
    'alpha_avg': 18.0,  # High alpha (drowsiness)
    'theta_avg': 16.0,  # High theta (drowsiness)
    # High-level inferences
    'neural_state': 'Drowsy',  # High alpha + high theta indicates drowsiness
    'signal_integrity': 'Clean',  # Max wave < 100
    'focus_depth': 'Surface Level'  # Not in Focus state
}
mock_checkin_burnout = {
    'session_name': 'Late Night Coding',
    'subject_task': 'Coding',
    'task_difficulty': 7,
    'estimated_length': '30-60m',
    'session_goal': 'Complete coding task',
    'energy_level': 3,  # Low energy
    'mood_emoji': 'Tired',
    'stress_level': 4,
    'time_since_meal': '3-4h',
    'lighting_conditions': 'Dim',
    'current_noise': 'Silence',
    'study_method': 'Standard'
}

# Mock Data for Scenario F: Subject Switch Confusion
mock_summary_switch = {
    'average_focus_score': 0.30,  # Low focus despite bright light
    'concentrating_seconds': 400,
    'neutral_seconds': 800,
    'relaxed_seconds': 600,
    'beta_avg': 10.0,
    'gamma_avg': 15.0,
    'alpha_avg': 20.0,  # High alpha (too harsh for creative work)
    'theta_avg': 10.0,
    # High-level inferences
    'neural_state': 'Distracted',  # High theta (>15) indicates distraction
    'signal_integrity': 'Clean',  # Max wave < 100
    'focus_depth': 'Surface Level'  # Not in Focus state
}
mock_checkin_switch = {
    'session_name': 'Creative Painting Session',
    'subject_task': 'Creative Painting',
    'task_difficulty': 5,
    'estimated_length': '30-60m',
    'session_goal': 'Painting practice',
    'energy_level': 6,
    'mood_emoji': 'Creative',
    'stress_level': 3,
    'time_since_meal': '1-2h',
    'lighting_conditions': 'Bright Light',  # Works for Math, too harsh for Painting
    'current_noise': 'Jazz Music',
    'study_method': 'Standard'
}

# Mock Data for Scenario G: Consistency Test (Reinforcement)
mock_summary_consistency = {
    'average_focus_score': 0.85,  # High focus, same conditions
    'concentrating_seconds': 1800,
    'neutral_seconds': 300,
    'relaxed_seconds': 100,
    'beta_avg': 22.0,
    'gamma_avg': 45.0,
    'alpha_avg': 9.0,
    'theta_avg': 5.0,
    # High-level inferences
    'neural_state': 'Focus',  # Beta 15-30 + Gamma > 30
    'signal_integrity': 'Clean',  # Max wave < 100
    'focus_depth': 'Deep Flow'  # Focus + Gamma > 40
}
mock_checkin_consistency = {
    'session_name': 'Reading Session',
    'subject_task': 'Reading',
    'task_difficulty': 5,
    'estimated_length': '30-60m',
    'session_goal': 'Read textbook',
    'energy_level': 8,
    'mood_emoji': 'Focused',
    'stress_level': 3,
    'time_since_meal': '1-2h',
    'lighting_conditions': 'Natural Light',  # Gold standard setup
    'current_noise': 'Jazz Music',
    'study_method': 'Standard'
}


def run_test(email, session_id, summary, checkin_data, description="", create_historical=False):
    """Run a test scenario with mock data"""
    print(f"\n{'='*80}")
    print(f"--- Testing Session: {session_id} - {description}")
    print(f"{'='*80}")
    
    # Create user profile
    user_profile = create_test_user_profile(email)
    
    # Create historical data if requested (to trigger Phase 2 logic)
    if create_historical:
        print(f"[TEST] Creating historical data for Phase 2 logic...")
        create_historical_data(user_profile, count=5)
    
    # Create SessionSummary
    session_summary = create_test_session_summary(user_profile, session_id, summary)
    print(f"[TEST] Created SessionSummary: {session_id}")
    
    # Create PreSessionCheckIn
    pre_check = create_test_pre_session_check_in(user_profile, session_id, checkin_data)
    print(f"[TEST] Created PreSessionCheckIn: {checkin_data.get('subject_task', 'Unknown')} - {checkin_data.get('lighting_conditions', 'Unknown')}")
    
    # Trigger AI Recommendation
    print(f"\n[TEST] Generating AI recommendation...")
    rec_json = generate_recommendation_for_session(email, session_id, summary)
    print(f"\n[AI OUTPUT]:\n{rec_json}\n")
    
    # Cleanup
    session_summary.delete()
    pre_check.delete()
    print(f"[TEST] Cleaned up SessionSummary and PreSessionCheckIn")
    
    return rec_json


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test recommendation system with mock data')
    parser.add_argument('--scenario', type=str, help='Test scenario to run (high, low, math, artifact, burnout, switch, consistency)')
    parser.add_argument('--user-email', type=str, default='test@example.com', help='User email')
    parser.add_argument('--phase2', action='store_true', help='Create historical data to trigger Phase 2 logic')
    
    args = parser.parse_args()
    
    user_email = args.user_email
    
    if args.scenario == 'high':
        # Scenario A: High Focus/High Gamma (Peak Performance)
        session_id = "SESS_HIGH_001"
        result = run_test(user_email, session_id, mock_summary_high, mock_checkin_high, "High Focus/High Gamma", args.phase2)
        
    elif args.scenario == 'low':
        # Scenario B: Low Focus/High Theta (Distracted)
        session_id = "SESS_LOW_002"
        result = run_test(user_email, session_id, mock_summary_low, mock_checkin_low, "Low Focus/High Theta", args.phase2)
        
    elif args.scenario == 'math':
        # Scenario C: Math with low focus (Subject-Specific)
        session_id = "SESS_MATH_003"
        result = run_test(user_email, session_id, mock_summary_math, mock_checkin_math, "Math Low Focus (Variable Pivot Test)", args.phase2)
        
    elif args.scenario == 'artifact':
        # Scenario D: Artifact Overload (Noise Filtering Validation)
        session_id = "SESS_ARTIFACT_004"
        result = run_test(user_email, session_id, mock_summary_artifact, mock_checkin_artifact, "Artifact Overload (Noise Filtering Validation)", args.phase2)
        
    elif args.scenario == 'burnout':
        # Scenario E: Late Night Burnout (Circadian Rhythm)
        session_id = "SESS_BURNOUT_005"
        result = run_test(user_email, session_id, mock_summary_burnout, mock_checkin_burnout, "Late Night Burnout (Circadian Rhythm)", args.phase2)
        
    elif args.scenario == 'switch':
        # Scenario F: Subject Switch Confusion
        session_id = "SESS_SWITCH_006"
        result = run_test(user_email, session_id, mock_summary_switch, mock_checkin_switch, "Subject Switch Confusion (Subject-Aware)", args.phase2)
        
    elif args.scenario == 'consistency':
        # Scenario G: Consistency Test (Reinforcement)
        session_id = "SESS_CONSISTENCY_007"
        result = run_test(user_email, session_id, mock_summary_consistency, mock_checkin_consistency, "Consistency Test (Reinforcement)", args.phase2)
        
    else:
        # Run all scenarios
        print("\n" + "="*80)
        print("RUNNING ALL TEST SCENARIOS")
        print("="*80)
        
        run_test(user_email, "SESS_HIGH_001", mock_summary_high, mock_checkin_high, "High Focus/High Gamma")
        run_test(user_email, "SESS_LOW_002", mock_summary_low, mock_checkin_low, "Low Focus/High Theta")
        run_test(user_email, "SESS_MATH_003", mock_summary_math, mock_checkin_math, "Math Low Focus (Variable Pivot Test)")
        run_test(user_email, "SESS_ARTIFACT_004", mock_summary_artifact, mock_checkin_artifact, "Artifact Overload (Noise Filtering Validation)")
        run_test(user_email, "SESS_BURNOUT_005", mock_summary_burnout, mock_checkin_burnout, "Late Night Burnout (Circadian Rhythm)")
        run_test(user_email, "SESS_SWITCH_006", mock_summary_switch, mock_checkin_switch, "Subject Switch Confusion (Subject-Aware)")
        run_test(user_email, "SESS_CONSISTENCY_007", mock_summary_consistency, mock_checkin_consistency, "Consistency Test (Reinforcement)")
        
    print(f"\n{'='*80}")
    print("TEST SUITE COMPLETED")
    print(f"{'='*80}\n")
