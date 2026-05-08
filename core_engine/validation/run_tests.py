import os
import sys
import argparse

from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEBAPP_DIR = os.path.join(BASE_DIR, "webapp")

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, WEBAPP_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "secondBrain.settings")

import django
django.setup()

from secondBrain_App.models import UserProfile, SessionSummary, PreSessionCheckIn, Recommendation
from core_engine.recommendation import generate_recommendation_for_session
from utils import (
    create_test_user_profile,
    create_test_session_summary,
    create_test_pre_session_check_in,
    create_historical_data,
    cleanup_session,
)



@dataclass
class MockSessionData:
    """Data class for mock session scenarios."""
    summary: dict
    checkin: dict
    description: str

# Mock data for various test scenarios
SCENARIOS = {
    'high': MockSessionData(
        summary={
            'average_focus_score': 0.85,
            'concentrating_seconds': 1800,
            'neutral_seconds': 300,
            'relaxed_seconds': 100,
            'beta_avg': 24.5,
            'gamma_avg': 48.2,
            'alpha_avg': 10.1,
            'theta_avg': 5.5,
            'neural_state': 'Focus',
            'signal_integrity': 'Clean',
            'focus_depth': 'Deep Flow'
        },
        checkin={
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
        },
        description="High Focus/High Gamma (Peak Performance)"
    ),
    'low': MockSessionData(
        summary={
            'average_focus_score': 0.30,
            'concentrating_seconds': 400,
            'neutral_seconds': 1200,
            'relaxed_seconds': 600,
            'beta_avg': 12.5,
            'gamma_avg': 15.0,
            'alpha_avg': 8.0,
            'theta_avg': 18.5,
            'neural_state': 'Distracted',
            'signal_integrity': 'Clean',
            'focus_depth': 'Surface Level'
        },
        checkin={
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
        },
        description="Low Focus/High Theta (Distracted)"
    ),
    'math': MockSessionData(
        summary={
            'average_focus_score': 0.35,
            'concentrating_seconds': 500,
            'neutral_seconds': 700,
            'relaxed_seconds': 600,
            'beta_avg': 12.0,
            'gamma_avg': 25.0,
            'alpha_avg': 15.0,
            'theta_avg': 12.0,
            'neural_state': 'Distracted',
            'signal_integrity': 'Clean',
            'focus_depth': 'Surface Level'
        },
        checkin={
            'session_name': 'Math Session',
            'subject_task': 'Advanced Mathematics',
            'task_difficulty': 8,
            'estimated_length': '30-60m',
            'session_goal': 'Solve math problems',
            'energy_level': 5,
            'mood_emoji': 'Neutral',
            'stress_level': 5,
            'time_since_meal': '1-2h',
            'lighting_conditions': 'Dim',
            'current_noise': 'Rock Music',
            'study_method': 'Standard'
        },
        description="Math Low Focus (Variable Pivot Test)"
    ),
    'artifact': MockSessionData(
        summary={
            'average_focus_score': 0.25,
            'concentrating_seconds': 300,
            'neutral_seconds': 900,
            'relaxed_seconds': 600,
            'beta_avg': 45.0,
            'gamma_avg': 65.0,
            'alpha_avg': 8.0,
            'theta_avg': 6.0,
            'neural_state': 'Anxious',
            'signal_integrity': 'Artifact-Heavy',
            'focus_depth': 'Surface Level'
        },
        checkin={
            'session_name': 'Restless Session',
            'subject_task': 'General Study',
            'task_difficulty': 5,
            'estimated_length': '30-60m',
            'session_goal': 'Study session',
            'energy_level': 8,
            'mood_emoji': 'Restless',
            'stress_level': 7,
            'time_since_meal': '1-2h',
            'lighting_conditions': 'Bright Light',
            'current_noise': 'Silence',
            'study_method': 'Standard'
        },
        description="Artifact Overload (Noise Filtering Validation)"
    ),
    'burnout': MockSessionData(
        summary={
            'average_focus_score': 0.40,
            'concentrating_seconds': 600,
            'neutral_seconds': 600,
            'relaxed_seconds': 600,
            'beta_avg': 15.0,
            'gamma_avg': 20.0,
            'alpha_avg': 18.0,
            'theta_avg': 16.0,
            'neural_state': 'Drowsy',
            'signal_integrity': 'Clean',
            'focus_depth': 'Surface Level'
        },
        checkin={
            'session_name': 'Late Night Coding',
            'subject_task': 'Coding',
            'task_difficulty': 7,
            'estimated_length': '30-60m',
            'session_goal': 'Complete coding task',
            'energy_level': 3,
            'mood_emoji': 'Tired',
            'stress_level': 4,
            'time_since_meal': '3-4h',
            'lighting_conditions': 'Dim',
            'current_noise': 'Silence',
            'study_method': 'Standard'
        },
        description="Late Night Burnout (Circadian Rhythm)"
    ),
    'switch': MockSessionData(
        summary={
            'average_focus_score': 0.30,
            'concentrating_seconds': 400,
            'neutral_seconds': 800,
            'relaxed_seconds': 600,
            'beta_avg': 10.0,
            'gamma_avg': 15.0,
            'alpha_avg': 20.0,
            'theta_avg': 10.0,
            'neural_state': 'Distracted',
            'signal_integrity': 'Clean',
            'focus_depth': 'Surface Level'
        },
        checkin={
            'session_name': 'Creative Painting Session',
            'subject_task': 'Creative Painting',
            'task_difficulty': 5,
            'estimated_length': '30-60m',
            'session_goal': 'Painting practice',
            'energy_level': 6,
            'mood_emoji': 'Creative',
            'stress_level': 3,
            'time_since_meal': '1-2h',
            'lighting_conditions': 'Bright Light',
            'current_noise': 'Jazz Music',
            'study_method': 'Standard'
        },
        description="Subject Switch Confusion (Subject-Aware)"
    ),
    'consistency': MockSessionData(
        summary={
            'average_focus_score': 0.85,
            'concentrating_seconds': 1800,
            'neutral_seconds': 300,
            'relaxed_seconds': 100,
            'beta_avg': 22.0,
            'gamma_avg': 45.0,
            'alpha_avg': 9.0,
            'theta_avg': 5.0,
            'neural_state': 'Focus',
            'signal_integrity': 'Clean',
            'focus_depth': 'Deep Flow'
        },
        checkin={
            'session_name': 'Reading Session',
            'subject_task': 'Reading',
            'task_difficulty': 5,
            'estimated_length': '30-60m',
            'session_goal': 'Read textbook',
            'energy_level': 8,
            'mood_emoji': 'Focused',
            'stress_level': 3,
            'time_since_meal': '1-2h',
            'lighting_conditions': 'Natural Light',
            'current_noise': 'Jazz Music',
            'study_method': 'Standard'
        },
        description="Consistency Test (Reinforcement)"
    )
}


def run_test(email: str, session_id: str, mock_data: MockSessionData, create_historical: bool = False) -> dict:
    """Run a test scenario with mock data."""

    # Cleanup
    cleanup_session(session_id)
    print("[TEST] Cleaned up SessionSummary and PreSessionCheckIn")

    print(f"\n{'='*80}")
    print(f"--- Testing Session: {session_id} - {mock_data.description}")
    print(f"{'='*80}")

    # Create user profile
    user_profile = create_test_user_profile(email)

    # Create historical data if requested (to trigger Phase 2 logic)
    if create_historical:
        print("[TEST] Creating historical data for Phase 2 logic...")
        create_historical_data(user_profile)

    # Create SessionSummary
    session_summary = create_test_session_summary(user_profile, session_id, mock_data.summary)
    print(f"[TEST] Created SessionSummary: {session_id}")

    # Create PreSessionCheckIn
    pre_check = create_test_pre_session_check_in(user_profile, session_id, mock_data.checkin)
    print(f"[TEST] Created PreSessionCheckIn: {mock_data.checkin.get('subject_task', 'Unknown')} - {mock_data.checkin.get('lighting_conditions', 'Unknown')}")

    # Trigger AI Recommendation and AI Judge output
    print("\n[TEST] Generating AI recommendation...")
    rec_json = generate_recommendation_for_session(email, session_id, mock_data.summary)
    print(f"\n[AI OUTPUT]:\n{rec_json}\n")

   

    return rec_json

def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(description='Test recommendation system with mock data')
    parser.add_argument('--scenario', type=str, help='Test scenario to run (high, low, math, artifact, burnout, switch, consistency)')
    parser.add_argument('--user-email', type=str, default='test@example.com', help='User email')
    parser.add_argument('--phase2', action='store_true', help='Create historical data to trigger Phase 2 logic')

    args = parser.parse_args()
    user_email = args.user_email

    if args.scenario in SCENARIOS:
        session_id = f"SESS_{args.scenario.upper()}_{args.scenario == 'high' and '001' or args.scenario == 'low' and '002' or args.scenario == 'math' and '003' or args.scenario == 'artifact' and '004' or args.scenario == 'burnout' and '005' or args.scenario == 'switch' and '006' or '007'}"
        run_test(user_email, session_id, SCENARIOS[args.scenario], args.phase2)
    else:
        # Run all scenarios
        print("\n" + "="*80)
        print("RUNNING ALL TEST SCENARIOS")
        print("="*80)

        for scenario_name, mock_data in SCENARIOS.items():
            session_id = f"SESS_{scenario_name.upper()}_{list(SCENARIOS.keys()).index(scenario_name) + 1:03d}"
            run_test(user_email, session_id, mock_data)

    print(f"\n{'='*80}")
    print("TEST SUITE COMPLETED")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
