from collections import defaultdict
from email import parser
from email import parser
import json
import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Counter
import time

from ai_validation import judge_response, client
from google.genai import types
from llm_validation import vf

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
    build_historical_context
)



@dataclass
class MockSessionData:
    """Data class for mock session scenarios."""
    summary: dict
    checkin: dict
    description: str


ISSUE_TYPES = [
    "unsupported_historical_claim",
    "unsupported_eeg_claim",
    "contradicted_eeg_value",
    "unsupported_time_of_day_claim",
    "unsupported_user_preference_claim",
    "unsafe_or_impractical_advice",
    "no_issue"
]

# Mock data for various test scenarios
SCENARIOS = {
    # 'high': MockSessionData(
    #     summary={
    #         'average_focus_score': 0.85,
    #         'concentrating_seconds': 1800,
    #         'neutral_seconds': 300,
    #         'relaxed_seconds': 100,
    #         'beta_avg': 24.5,
    #         'gamma_avg': 48.2,
    #         'alpha_avg': 10.1,
    #         'theta_avg': 5.5,
    #         'neural_state': 'Focus',
    #         'signal_integrity': 'Clean',
    #         'focus_depth': 'Deep Flow'
    #     },
    #     checkin={
    #         'session_name': 'High Focus Session',
    #         'subject_task': 'UI Design',
    #         'task_difficulty': 6,
    #         'estimated_length': '30-60m',
    #         'session_goal': 'Complete design task',
    #         'energy_level': 8,
    #         'mood_emoji': 'Focused',
    #         'stress_level': 3,
    #         'time_since_meal': '1-2h',
    #         'lighting_conditions': 'Warm Natural Light',
    #         'current_noise': 'Jazz Music',
    #         'study_method': 'Flow State'
    #     },
    #     description="High Focus/High Gamma (Peak Performance)"
    # ),
    # 'low': MockSessionData(
    #     summary={
    #         'average_focus_score': 0.30,
    #         'concentrating_seconds': 400,
    #         'neutral_seconds': 1200,
    #         'relaxed_seconds': 600,
    #         'beta_avg': 12.5,
    #         'gamma_avg': 15.0,
    #         'alpha_avg': 8.0,
    #         'theta_avg': 18.5,
    #         'neural_state': 'Distracted',
    #         'signal_integrity': 'Clean',
    #         'focus_depth': 'Surface Level'
    #     },
    #     checkin={
    #         'session_name': 'Low Focus Session',
    #         'subject_task': 'Reading',
    #         'task_difficulty': 5,
    #         'estimated_length': '30-60m',
    #         'session_goal': 'Read textbook',
    #         'energy_level': 4,
    #         'mood_emoji': 'Tired',
    #         'stress_level': 6,
    #         'time_since_meal': '2-4h',
    #         'lighting_conditions': 'Warm Light',
    #         'current_noise': 'White Noise',
    #         'study_method': 'Standard'
    #     },
    #     description="Low Focus/High Theta (Distracted)"
    # ),
    # 'math': MockSessionData(
    #     summary={
    #         'average_focus_score': 0.35,
    #         'concentrating_seconds': 500,
    #         'neutral_seconds': 700,
    #         'relaxed_seconds': 600,
    #         'beta_avg': 12.0,
    #         'gamma_avg': 25.0,
    #         'alpha_avg': 15.0,
    #         'theta_avg': 12.0,
    #         'neural_state': 'Distracted',
    #         'signal_integrity': 'Clean',
    #         'focus_depth': 'Surface Level'
    #     },
    #     checkin={
    #         'session_name': 'Math Session',
    #         'subject_task': 'Advanced Mathematics',
    #         'task_difficulty': 8,
    #         'estimated_length': '30-60m',
    #         'session_goal': 'Solve math problems',
    #         'energy_level': 5,
    #         'mood_emoji': 'Neutral',
    #         'stress_level': 5,
    #         'time_since_meal': '1-2h',
    #         'lighting_conditions': 'Dim',
    #         'current_noise': 'Rock Music',
    #         'study_method': 'Standard'
    #     },
    #     description="Math Low Focus (Variable Pivot Test)"
    # ),
    # 'artifact': MockSessionData(
    #     summary={
    #         'average_focus_score': 0.25,
    #         'concentrating_seconds': 300,
    #         'neutral_seconds': 900,
    #         'relaxed_seconds': 600,
    #         'beta_avg': 45.0,
    #         'gamma_avg': 65.0,
    #         'alpha_avg': 8.0,
    #         'theta_avg': 6.0,
    #         'neural_state': 'Anxious',
    #         'signal_integrity': 'Artifact-Heavy',
    #         'focus_depth': 'Surface Level'
    #     },
    #     checkin={
    #         'session_name': 'Restless Session',
    #         'subject_task': 'General Study',
    #         'task_difficulty': 5,
    #         'estimated_length': '30-60m',
    #         'session_goal': 'Study session',
    #         'energy_level': 8,
    #         'mood_emoji': 'Restless',
    #         'stress_level': 7,
    #         'time_since_meal': '1-2h',
    #         'lighting_conditions': 'Bright Light',
    #         'current_noise': 'Silence',
    #         'study_method': 'Standard'
    #     },
    #     description="Artifact Overload (Noise Filtering Validation)"
    # ),
    # 'burnout': MockSessionData(
    #     summary={
    #         'average_focus_score': 0.40,
    #         'concentrating_seconds': 600,
    #         'neutral_seconds': 600,
    #         'relaxed_seconds': 600,
    #         'beta_avg': 15.0,
    #         'gamma_avg': 20.0,
    #         'alpha_avg': 18.0,
    #         'theta_avg': 16.0,
    #         'neural_state': 'Drowsy',
    #         'signal_integrity': 'Clean',
    #         'focus_depth': 'Surface Level'
    #     },
    #     checkin={
    #         'session_name': 'Late Night Coding',
    #         'subject_task': 'Coding',
    #         'task_difficulty': 7,
    #         'estimated_length': '30-60m',
    #         'session_goal': 'Complete coding task',
    #         'energy_level': 3,
    #         'mood_emoji': 'Tired',
    #         'stress_level': 4,
    #         'time_since_meal': '3-4h',
    #         'lighting_conditions': 'Dim',
    #         'current_noise': 'Silence',
    #         'study_method': 'Standard'
    #     },
    #     description="Late Night Burnout (Circadian Rhythm)"
    # ),
    # 'switch': MockSessionData(
    #     summary={
    #         'average_focus_score': 0.30,
    #         'concentrating_seconds': 400,
    #         'neutral_seconds': 800,
    #         'relaxed_seconds': 600,
    #         'beta_avg': 10.0,
    #         'gamma_avg': 15.0,
    #         'alpha_avg': 20.0,
    #         'theta_avg': 10.0,
    #         'neural_state': 'Distracted',
    #         'signal_integrity': 'Clean',
    #         'focus_depth': 'Surface Level'
    #     },
    #     checkin={
    #         'session_name': 'Creative Painting Session',
    #         'subject_task': 'Creative Painting',
    #         'task_difficulty': 5,
    #         'estimated_length': '30-60m',
    #         'session_goal': 'Painting practice',
    #         'energy_level': 6,
    #         'mood_emoji': 'Creative',
    #         'stress_level': 3,
    #         'time_since_meal': '1-2h',
    #         'lighting_conditions': 'Bright Light',
    #         'current_noise': 'Jazz Music',
    #         'study_method': 'Standard'
    #     },
    #     description="Subject Switch Confusion (Subject-Aware)"
    # ),
    # 'consistency': MockSessionData(
    #     summary={
    #         'average_focus_score': 0.85,
    #         'concentrating_seconds': 1800,
    #         'neutral_seconds': 300,
    #         'relaxed_seconds': 100,
    #         'beta_avg': 22.0,
    #         'gamma_avg': 45.0,
    #         'alpha_avg': 9.0,
    #         'theta_avg': 5.0,
    #         'neural_state': 'Focus',
    #         'signal_integrity': 'Clean',
    #         'focus_depth': 'Deep Flow'
    #     },
    #     checkin={
    #         'session_name': 'Reading Session',
    #         'subject_task': 'Reading',
    #         'task_difficulty': 5,
    #         'estimated_length': '30-60m',
    #         'session_goal': 'Read textbook',
    #         'energy_level': 8,
    #         'mood_emoji': 'Focused',
    #         'stress_level': 3,
    #         'time_since_meal': '1-2h',
    #         'lighting_conditions': 'Natural Light',
    #         'current_noise': 'Jazz Music',
    #         'study_method': 'Standard'
    #     },
    #     description="Consistency Test (Reinforcement)"
    # ),
    ######### AI Validation Integration Cases #############################

     'high_focus': MockSessionData(
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
        description="High Focus/High Gamma"
    ),

    'low_focus': MockSessionData(
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
        description="Low Focus/High Theta"
    ),

    'math_difficulty': MockSessionData(
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
        description="Difficult Math With Low Focus"
    ),

    'artifact_heavy': MockSessionData(
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
        description="Artifact Heavy Signal"
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
        description="Late Night Burnout"
    ),

    'creative_switch': MockSessionData(
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
        description="Creative Subject Switch"
    ),

    'reading_success': MockSessionData(
        summary={
            'average_focus_score': 0.78,
            'concentrating_seconds': 1500,
            'neutral_seconds': 450,
            'relaxed_seconds': 250,
            'beta_avg': 20.0,
            'gamma_avg': 34.0,
            'alpha_avg': 11.0,
            'theta_avg': 7.0,
            'neural_state': 'Focused',
            'signal_integrity': 'Clean',
            'focus_depth': 'Moderate Deep Focus'
        },
        checkin={
            'session_name': 'Reading Session',
            'subject_task': 'History Reading',
            'task_difficulty': 4,
            'estimated_length': '30-60m',
            'session_goal': 'Finish one chapter',
            'energy_level': 7,
            'mood_emoji': 'Calm',
            'stress_level': 2,
            'time_since_meal': '1-2h',
            'lighting_conditions': 'Natural Light',
            'current_noise': 'Lo-fi Music',
            'study_method': 'Pomodoro'
        },
        description="Strong Reading Session"
    ),

    'high_stress_good_focus': MockSessionData(
        summary={
            'average_focus_score': 0.72,
            'concentrating_seconds': 1400,
            'neutral_seconds': 500,
            'relaxed_seconds': 200,
            'beta_avg': 28.0,
            'gamma_avg': 42.0,
            'alpha_avg': 7.5,
            'theta_avg': 6.0,
            'neural_state': 'Focused but Tense',
            'signal_integrity': 'Clean',
            'focus_depth': 'Moderate Deep Focus'
        },
        checkin={
            'session_name': 'Exam Prep',
            'subject_task': 'Biology Exam Review',
            'task_difficulty': 8,
            'estimated_length': '60-90m',
            'session_goal': 'Review difficult chapters',
            'energy_level': 7,
            'mood_emoji': 'Stressed',
            'stress_level': 9,
            'time_since_meal': '1-2h',
            'lighting_conditions': 'Bright Light',
            'current_noise': 'Silence',
            'study_method': 'Active Recall'
        },
        description="High Stress But Productive Focus"
    ),

    'low_energy_clean_signal': MockSessionData(
        summary={
            'average_focus_score': 0.48,
            'concentrating_seconds': 700,
            'neutral_seconds': 800,
            'relaxed_seconds': 500,
            'beta_avg': 14.0,
            'gamma_avg': 22.0,
            'alpha_avg': 14.5,
            'theta_avg': 13.5,
            'neural_state': 'Low Energy',
            'signal_integrity': 'Clean',
            'focus_depth': 'Surface Level'
        },
        checkin={
            'session_name': 'Morning Review',
            'subject_task': 'Chemistry',
            'task_difficulty': 6,
            'estimated_length': '30-60m',
            'session_goal': 'Review notes',
            'energy_level': 2,
            'mood_emoji': 'Sleepy',
            'stress_level': 3,
            'time_since_meal': 'Before eating',
            'lighting_conditions': 'Dim',
            'current_noise': 'Silence',
            'study_method': 'Reading Notes'
        },
        description="Low Energy Clean Signal"
    ),

    'easy_task_low_focus': MockSessionData(
        summary={
            'average_focus_score': 0.32,
            'concentrating_seconds': 350,
            'neutral_seconds': 1000,
            'relaxed_seconds': 500,
            'beta_avg': 11.5,
            'gamma_avg': 16.0,
            'alpha_avg': 17.0,
            'theta_avg': 14.0,
            'neural_state': 'Understimulated',
            'signal_integrity': 'Clean',
            'focus_depth': 'Surface Level'
        },
        checkin={
            'session_name': 'Easy Flashcards',
            'subject_task': 'Vocabulary Flashcards',
            'task_difficulty': 2,
            'estimated_length': '15-30m',
            'session_goal': 'Review simple terms',
            'energy_level': 6,
            'mood_emoji': 'Bored',
            'stress_level': 1,
            'time_since_meal': '1-2h',
            'lighting_conditions': 'Natural Light',
            'current_noise': 'Cafe Noise',
            'study_method': 'Flashcards'
        },
        description="Easy Task Causing Boredom"
    ),

    'hard_task_high_focus': MockSessionData(
        summary={
            'average_focus_score': 0.88,
            'concentrating_seconds': 2000,
            'neutral_seconds': 250,
            'relaxed_seconds': 100,
            'beta_avg': 26.0,
            'gamma_avg': 50.0,
            'alpha_avg': 8.5,
            'theta_avg': 5.0,
            'neural_state': 'Deep Focus',
            'signal_integrity': 'Clean',
            'focus_depth': 'Deep Flow'
        },
        checkin={
            'session_name': 'Algorithms Grind',
            'subject_task': 'Algorithms',
            'task_difficulty': 9,
            'estimated_length': '60-90m',
            'session_goal': 'Solve dynamic programming problems',
            'energy_level': 9,
            'mood_emoji': 'Motivated',
            'stress_level': 4,
            'time_since_meal': '1-2h',
            'lighting_conditions': 'Bright Light',
            'current_noise': 'Instrumental Music',
            'study_method': 'Practice Problems'
        },
        description="Hard Task With Deep Focus"
    ),

    'noisy_environment': MockSessionData(
        summary={
            'average_focus_score': 0.42,
            'concentrating_seconds': 600,
            'neutral_seconds': 900,
            'relaxed_seconds': 450,
            'beta_avg': 17.0,
            'gamma_avg': 24.0,
            'alpha_avg': 10.0,
            'theta_avg': 12.0,
            'neural_state': 'Distracted',
            'signal_integrity': 'Clean',
            'focus_depth': 'Surface Level'
        },
        checkin={
            'session_name': 'Cafe Coding',
            'subject_task': 'Web Development',
            'task_difficulty': 6,
            'estimated_length': '30-60m',
            'session_goal': 'Fix frontend bug',
            'energy_level': 6,
            'mood_emoji': 'Neutral',
            'stress_level': 5,
            'time_since_meal': '2-4h',
            'lighting_conditions': 'Natural Light',
            'current_noise': 'Loud Cafe',
            'study_method': 'Debugging'
        },
        description="Noisy Environment Hurt Focus"
    ),

    'calm_relaxed_learning': MockSessionData(
        summary={
            'average_focus_score': 0.62,
            'concentrating_seconds': 1000,
            'neutral_seconds': 650,
            'relaxed_seconds': 350,
            'beta_avg': 16.0,
            'gamma_avg': 28.0,
            'alpha_avg': 18.0,
            'theta_avg': 8.0,
            'neural_state': 'Calm Focus',
            'signal_integrity': 'Clean',
            'focus_depth': 'Moderate Focus'
        },
        checkin={
            'session_name': 'Concept Review',
            'subject_task': 'Psychology',
            'task_difficulty': 4,
            'estimated_length': '30-60m',
            'session_goal': 'Understand core concepts',
            'energy_level': 6,
            'mood_emoji': 'Calm',
            'stress_level': 2,
            'time_since_meal': '1-2h',
            'lighting_conditions': 'Warm Light',
            'current_noise': 'Nature Sounds',
            'study_method': 'Concept Mapping'
        },
        description="Calm Relaxed Learning"
    ),

    'anxious_low_focus': MockSessionData(
        summary={
            'average_focus_score': 0.28,
            'concentrating_seconds': 250,
            'neutral_seconds': 950,
            'relaxed_seconds': 700,
            'beta_avg': 30.0,
            'gamma_avg': 35.0,
            'alpha_avg': 6.5,
            'theta_avg': 11.0,
            'neural_state': 'Anxious',
            'signal_integrity': 'Clean',
            'focus_depth': 'Surface Level'
        },
        checkin={
            'session_name': 'Deadline Work',
            'subject_task': 'Essay Writing',
            'task_difficulty': 7,
            'estimated_length': '60-90m',
            'session_goal': 'Write introduction and outline',
            'energy_level': 5,
            'mood_emoji': 'Anxious',
            'stress_level': 9,
            'time_since_meal': '4h+',
            'lighting_conditions': 'Bright Light',
            'current_noise': 'Silence',
            'study_method': 'Free Writing'
        },
        description="Anxious Low Focus"
    ),

    'post_meal_slump': MockSessionData(
        summary={
            'average_focus_score': 0.38,
            'concentrating_seconds': 450,
            'neutral_seconds': 850,
            'relaxed_seconds': 700,
            'beta_avg': 13.0,
            'gamma_avg': 18.0,
            'alpha_avg': 16.0,
            'theta_avg': 17.0,
            'neural_state': 'Drowsy',
            'signal_integrity': 'Clean',
            'focus_depth': 'Surface Level'
        },
        checkin={
            'session_name': 'After Lunch Study',
            'subject_task': 'Statistics',
            'task_difficulty': 6,
            'estimated_length': '30-60m',
            'session_goal': 'Practice hypothesis testing',
            'energy_level': 3,
            'mood_emoji': 'Tired',
            'stress_level': 4,
            'time_since_meal': 'Less than 1h',
            'lighting_conditions': 'Warm Light',
            'current_noise': 'White Noise',
            'study_method': 'Practice Problems'
        },
        description="Post Meal Slump"
    ),

    'long_session_fatigue': MockSessionData(
        summary={
            'average_focus_score': 0.55,
            'concentrating_seconds': 1200,
            'neutral_seconds': 1100,
            'relaxed_seconds': 900,
            'beta_avg': 18.0,
            'gamma_avg': 26.0,
            'alpha_avg': 13.0,
            'theta_avg': 15.0,
            'neural_state': 'Fatigued Focus',
            'signal_integrity': 'Clean',
            'focus_depth': 'Inconsistent Focus'
        },
        checkin={
            'session_name': 'Long Project Session',
            'subject_task': 'Software Engineering Project',
            'task_difficulty': 7,
            'estimated_length': '90m+',
            'session_goal': 'Implement recommendation validation flow',
            'energy_level': 6,
            'mood_emoji': 'Determined',
            'stress_level': 6,
            'time_since_meal': '2-4h',
            'lighting_conditions': 'Bright Light',
            'current_noise': 'Lo-fi Music',
            'study_method': 'Deep Work'
        },
        description="Long Session Fatigue"
    )
}
######### AI Validation Integration Cases #############################

def build_judge_context(
    user_profile,
    session_summary,
    pre_check,
    mock_data,
    create_historical=False
):
    historical_context = build_historical_context(user_profile)
    #print("\n[TEST] Building context for judge evaluation...", historical_context)
    return {
        "scenario": {
            "description": mock_data.description,
            "phase": "Phase 2" if create_historical else "Phase 1"
        },

        "user_profile": {
            "sound_environment": user_profile.sound_environment,
            "sleep_quality": user_profile.sleep_quality,
            "learning_style": user_profile.learning_style,
            "main_goals": user_profile.main_goals,
        },

        "pre_session": {
            "session_name": pre_check.session_name,
            "subject_task": pre_check.subject_task,
            "task_difficulty": pre_check.task_difficulty,
            "estimated_length": pre_check.estimated_length,
            "session_goal": pre_check.session_goal,
            "energy_level": pre_check.energy_level,
            "mood_emoji": pre_check.mood_emoji,
            "stress_level": pre_check.stress_level,
            "time_since_meal": pre_check.time_since_meal,
            "lighting_conditions": pre_check.lighting_conditions,
            "current_noise": pre_check.current_noise,
            "study_method": pre_check.study_method,
        },

        "eeg_results": {
            "average_focus_score": session_summary.average_focus_score,
            "concentrating_seconds": session_summary.concentrating_seconds,
            "neutral_seconds": session_summary.neutral_seconds,
            "relaxed_seconds": session_summary.relaxed_seconds,
            "beta_avg": session_summary.beta_avg,
            "gamma_avg": session_summary.gamma_avg,
            "alpha_avg": session_summary.alpha_avg,
            "theta_avg": session_summary.theta_avg,
            "neural_state": session_summary.neural_state,
            "signal_integrity": session_summary.signal_integrity,
            "focus_depth": session_summary.focus_depth,
            "total_duration_seconds": session_summary.total_duration_seconds,
        },
        "historical_context": historical_context
    }


def run_test(email: str, session_id: str, mock_data: MockSessionData, create_historical: bool = False, llm_validation: bool = False) -> dict:
    cleanup_session(session_id)

    print(f"\n{'=' * 80}")
    print(f"--- Testing Session: {session_id} - {mock_data.description}")
    print(f"{'=' * 80}")

    user_profile = create_test_user_profile(email)

    if create_historical:
        create_historical_data(user_profile)

    session_summary = create_test_session_summary(
        user_profile,
        session_id,
        mock_data.summary
    )

    pre_check = create_test_pre_session_check_in(
        user_profile,
        session_id,
        mock_data.checkin
    )

    print("\n[TEST] Generating AI recommendation...")
    recommendation = generate_recommendation_for_session(
        email,
        session_id,
        mock_data.summary
    )
    print(f"[TEST] Recommendation: {recommendation}")

    context = build_judge_context(
        user_profile=user_profile,
        session_summary=session_summary,
        pre_check=pre_check,
        mock_data=mock_data,
        create_historical=create_historical
    )

    content= """You are an AI-powered Study Optimization Advisor analyzing EEG brainwave data.

USER PROFILE:
- Sound preference   : {sound}
- Sleep quality      : {sleep}
- Learning style     : {style}
- Study goals        : {goals}
- Subject studying   : {subject}

EEG SESSION RESULTS:
- Avg focus score    : {avg_focus:.2f}
- Concentrating time : {conc} seconds
- Neutral time       : {neut} seconds
- Relaxed/distracted : {relax} seconds
- Session duration   : {duration} mins

BRAINWAVE ANALYSIS:
- Neural State       : {neural_state}
- Signal Integrity   : {signal_integrity}
- Focus Depth        : {focus_depth}
- Beta waves        : {beta_avg:.2f} Hz
- Gamma waves       : {gamma_avg:.2f} Hz
- Alpha waves       : {alpha_avg:.2f} Hz
- Theta waves       : {theta_avg:.2f} Hz

RESPOND WITH:
1. 1-2 line fun personalized recommendation based on their EEG session results
2. Recommended Study Methods (3-4 bullet points)
3. Optimal study environment for this user
4. Tailor study methods specifically for {subject}
""".format(
        sound        = context['user_profile']['sound_environment'],
        sleep        = context['user_profile']['sleep_quality'],
        style        = context['user_profile']['learning_style'],
        goals        = context['user_profile']['main_goals'],
        avg_focus    = float(context['eeg_results']['average_focus_score']),
        conc         = context['eeg_results']['concentrating_seconds'],
        neut         = context['eeg_results']['neutral_seconds'],
        relax        = context['eeg_results']['relaxed_seconds'],
        duration     = round(context['eeg_results']['total_duration_seconds'] / 60, 1),
        neural_state = context['eeg_results']['neural_state'],
        signal_integrity = context['eeg_results']['signal_integrity'],
        focus_depth  = context['eeg_results']['focus_depth'],
        beta_avg     = context['eeg_results']['beta_avg'],
        gamma_avg    = context['eeg_results']['gamma_avg'],
        alpha_avg    = context['eeg_results']['alpha_avg'],
        theta_avg    = context['eeg_results']['theta_avg'],
        subject      = context['pre_session']['subject_task']
    )


    if llm_validation:
        print("\n[TEST] Running LLM validation of recommendation...")
        print(f"[TEST] LLM Validation Result: {vf.validate(content)}")

    return {
        "session_id": session_id,
        "context": context,
        "recommendation": recommendation,
    }


class Orchestrator:
    DEBATE_PROMPT = """
        You are a final consensus judge for an EEG-based study recommendation system.

        You will receive evaluations from 3 AI judges.
        Each judge reviewed 5 different sessions using the same rubric.

        Your job:
        1. Compare the judges' evaluations.
        2. Identify which recommendations were strongest and why.
        3. Identify recurring weaknesses.
        4. Extract reusable feedback rules for future recommendations.
        5. Decide what the recommendation system should do differently next time.

        Return only valid JSON with:
        {
        "best_recommendation_patterns": [],
        "weak_recommendation_patterns": [],
        "recurring_issues": [],
        "future_prompt_rules": [],
        "overall_summary": ""
        }
        """

    def assign_judges(self, cases):
        if len(cases) != 15:
            raise ValueError(f"Expected exactly 15 cases, got {len(cases)}")

        return {
            "Judge1": cases[0:5],
            "Judge2": cases[5:10],
            "Judge3": cases[10:15],
        }

    def run_judge_batch(self, judge_name, assigned_cases, all_judge_results, output_file):
        results = []

        for case in assigned_cases:
            session_id = case["session_id"]

            if self.already_done(all_judge_results, judge_name, session_id):
                print(f"[SKIP] {judge_name} already judged {session_id}")
                continue

            print(f"[ORCH] {judge_name} judging {session_id}")

            judgment = judge_response(
                context=case["context"],
                recommendation=case["recommendation"]
            )

            result = {
                "judge": judge_name,
                "session_id": session_id,
                "judgment": judgment,
            }

            all_judge_results.append(result)
            results.append(result)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_judge_results, f, indent=2, default=str)

            time.sleep(5)

        return results

    def debate_judge_outputs(self, all_judge_results):
        print("\n[ORCH] Compiling judge results for debate...")

        payload = {
            "judge_results": all_judge_results
        }

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{self.DEBATE_PROMPT}\n{json.dumps(payload, indent=2)}",
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json"
            )
        )
        #print(f"\n[ORCH] Debate response: {response.text}")

        return json.loads(response.text)
    
    def load_existing_results(self, path):
        if not os.path.exists(path):
            return []

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


    def already_done(self, existing_results, judge_name, session_id):
        return any(
            r["judge"] == judge_name and r["session_id"] == session_id
            for r in existing_results
        )

    def run_all_judges_on_same_cases(self, cases, output_file="judge_reliability_partial.json"):
        all_results = self.load_existing_results(output_file)

        for judge_name in ["Judge1", "Judge2", "Judge3"]:
            print(f"\n[RELIABILITY] Running {judge_name} on all {len(cases)} cases")

            for case in cases:
                session_id = case["session_id"]

                if self.already_done(all_results, judge_name, session_id):
                    print(f"[SKIP] {judge_name} already judged {session_id}")
                    continue

                print(f"[RELIABILITY] {judge_name} judging {session_id}")

                judgment = judge_response(
                    context=case["context"],
                    recommendation=case["recommendation"]
                )
                
                all_results.append({
                    "judge": judge_name,
                    "session_id": session_id,
                    "judgment": judgment,
                })

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, default=str)

                time.sleep(5)

        return all_results


    def summarize_judge_agreement(self, judge_results):
        """
        Compares whether judges agree on has_hallucination.
        Your judge_response JSON should include has_hallucination.
        """
        grouped = defaultdict(list)

        for result in judge_results:
            grouped[result["session_id"]].append(result)

        agreement_summary = {}

        for session_id, results in grouped.items():
            votes = []

            for result in results:
                judgment = result["judgment"]
                votes.append(bool(judgment.get("has_hallucination", False)))

            vote_counts = Counter(votes)

            agreement_summary[session_id] = {
                "votes": votes,
                "num_judges": len(votes),
                "agree_count": max(vote_counts.values()),
                "agreement_rate": max(vote_counts.values()) / len(votes),
                "majority_has_hallucination": vote_counts[True] > vote_counts[False],
            }

        return agreement_summary
    
    

    def run(self, evaluated_cases,  output_file="judge_results_partial.json"):
        allocations = self.assign_judges(evaluated_cases)

        all_judge_results = self.load_existing_results(output_file)

        for judge_name, cases in allocations.items():
            print(f"\n[ORCH] Running {judge_name} on {len(cases)} sessions")

            self.run_judge_batch(
                judge_name=judge_name,
                assigned_cases=cases,
                all_judge_results=all_judge_results,
                output_file=output_file
            )

        print("\n[ORCH] Running final debate/consensus")

        consensus = self.debate_judge_outputs(all_judge_results)

        return {
            "judge_results": all_judge_results,
            "consensus": consensus,
        }

def load_gold_cases(path):
        with open(path, "r", encoding="utf-8") as f:
            gold_cases = json.load(f)

        return {
            case["session_id"]: case
            for case in gold_cases
        }
def compare_to_gold(judge_results, gold_cases):
    correct = 0
    total = 0

    detailed_results = []

    for result in judge_results:
        session_id = result["session_id"]

        if session_id not in gold_cases:
            continue

        gold = gold_cases[session_id]

        predicted = result["judgment"].get(
            "has_hallucination",
            False
        )

        expected = gold["expected_has_hallucination"]

        is_correct = predicted == expected

        if is_correct:
            correct += 1

        total += 1

        detailed_results.append({
            "session_id": session_id,
            "judge": result["judge"],
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct
        })

    accuracy = correct / total if total else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": detailed_results
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str)
    parser.add_argument("--user-email", type=str, default="test@example.com")
    parser.add_argument("--phase2", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--cases-file", type=str, default="evaluated_cases.json")
    parser.add_argument("--llm_validation", action="store_true")
    parser.add_argument("--reliability", action="store_true")
    parser.add_argument("--reliability-output", type=str, default="judge_reliability.json")
    parser.add_argument("--gold-file", type=str, default="gold_cases.json")

    args = parser.parse_args()
    user_email = args.user_email

    if args.judge_only:
        with open(args.cases_file, "r", encoding="utf-8") as f:
            evaluated_cases = json.load(f)

        print(f"[MAIN] Loaded {len(evaluated_cases)} cases from {args.cases_file}")

        orchestrator = Orchestrator()

        if args.reliability:
            judge_results = orchestrator.run_all_judges_on_same_cases(evaluated_cases)
            agreement = orchestrator.summarize_judge_agreement(judge_results)

            output = {
                "judge_results": judge_results,
                "agreement": agreement,
            }

            with open(args.reliability_output, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, default=str)

            print("\n[RELIABILITY SUMMARY]")
            print(json.dumps(agreement, indent=2))
        
            if args.gold_file:
                gold_cases = load_gold_cases(args.gold_file)
                comparison = compare_to_gold(judge_results, gold_cases)

                print("\n[GOLD COMPARISON]")
                print(json.dumps(comparison, indent=2))

                return

        final_results = orchestrator.run(evaluated_cases)

        print("\n[FINAL CONSENSUS]")
        print(json.dumps(final_results["consensus"], indent=2))


    if args.scenario:
        mock_data = SCENARIOS[args.scenario]
        session_id = f"SESS_{args.scenario.upper()}_001"

        

        result = run_test(
            user_email,
            session_id,
            mock_data,
            args.phase2,
            llm_validation=args.llm_validation
        )

        print(json.dumps(result, indent=2, default=str))
        return

    evaluated_cases = []
    selected_scenarios = list(SCENARIOS.items())[:15]

    for index, (scenario_name, mock_data) in enumerate(selected_scenarios, start=1):
        session_id = f"SESS_{scenario_name.upper()}_{index:03d}"

        result = run_test(
            user_email,
            session_id,
            mock_data,
            args.phase2,
            llm_validation=args.llm_validation
        )

        evaluated_cases.append(result)

    with open(args.cases_file, "w", encoding="utf-8") as f:
        json.dump(evaluated_cases, f, indent=2, default=str)

    print(f"[MAIN] Saved {len(evaluated_cases)} cases to {args.cases_file}")

    if args.generate_only:
        return

    orchestrator = Orchestrator()
    final_results = orchestrator.run(evaluated_cases)

    print("\n[FINAL CONSENSUS]")
    print(json.dumps(final_results["consensus"], indent=2))


if __name__ == "__main__":
    main()