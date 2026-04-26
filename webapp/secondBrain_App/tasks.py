import os
import sys
import time
import csv
from pathlib import Path
from datetime import datetime
from celery import shared_task
from django.conf import settings
import pandas as pd
import numpy as np
from django.core.cache import cache

# Add project root and core_engine to path
ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT
CORE_ENGINE_DIR = ROOT / 'core_engine'
SETUP_DIR = ROOT / 'data_pipeline' / 'setup'

# Add paths to sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CORE_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_ENGINE_DIR))
if str(SETUP_DIR) not in sys.path:
    sys.path.insert(0, str(SETUP_DIR))

# Import ML components
from .services.prediction_service import PredictionService
from .services.EEG_feature_extraction_adv import generate_feature_vectors_from_matrix
from .services.enhanced_feature_extraction import load_preprocessing_artifacts, apply_feature_pipeline

# Import Django models
from secondBrain_App.models import UserProfile, SessionSummary

# Constants
LABEL_MAP = {0: 'relaxed', 1: 'neutral', 2: 'concentrating'}



@shared_task(bind=True)
def run_live_inference(self, user_email, duration_minutes=1):
    """
    Run live EEG inference for specified duration.
    
    Args:
        user_email (str): User's email for identification
        duration_minutes (int): Duration in minutes for recording
    
    Returns:
        dict: Prediction results and session info
    """
    try:
        # Generate unique session info
        from django.utils import timezone
        timestamp = timezone.now().strftime("%Y%m%d_%H%M%S")
        user_prefix = user_email.split('@')[0]
        raw_output_path = ROOT / 'dataset' / 'our_data' / f'{user_prefix}_new' / f'{user_prefix}_{timestamp}_session.csv'
        
        # Ensure output directory exists
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize prediction service
        models_dir = ROOT / 'core_engine' / 'artifacts' 
        prediction_service = PredictionService(models_dir=models_dir, model_name='xgboost')
        
        # Initialize EEG acquirer (non-GUI version)
        from .services.eeg_acquirer import EEGAcquirer
        acq = EEGAcquirer(max_seconds=duration_minutes * 60, sfreq=256)
        
        print(f"Starting EEG inference for {user_email} - Duration: {duration_minutes} minutes")
        print(f"Raw data will be saved to: {raw_output_path}")
        
        # Connect and start acquisition
        acq.connect()
        acq.start()
        
        # Start saving raw data
        acq.start_saving_raw(str(raw_output_path))
        
        # Track predictions
        all_predictions = []
        start_time = time.time()
        duration_seconds = duration_minutes * 60
        
        print("EEG streaming started. Processing predictions...")
        
        stop_key = f"stop eeg task{self.request.id}"
        # Main processing loop
        while time.time() - start_time < duration_seconds:
            # if cache.get(stop_key, False):
            #     print(f"Stop requested for task {self.request.id}")
            #     break
            try:
                # Get buffer data
                rows = acq.get_buffer_copy()
                
                # Minimum samples needed (approximately 1.5 seconds of data)
                min_samples = int(256 * 1.0) 
                if len(rows) < min_samples:
                    time.sleep(0.5)
                    continue
                
                # Process predictions
                result = prediction_service.run(rows, nsamples=150, period=1.0, cols_to_ignore=-1)
                
                if result.get('ok') is False:
                    time.sleep(0.5)
                    continue
                all_predictions.append(result)
                print(f"Prediction update: {result.get('predicted_label')} (confidence: {result.get('confidence'):.2f})")
                
                time.sleep(0.5)  # Update twice per second
                
            except Exception as e:
                print(f"Error during prediction loop: {e}")
                time.sleep(1)
                continue
        
        # Stop acquisition
        acq.stop()
        print("EEG streaming stopped.")
        
        # Aggregate final results
        if all_predictions:
            final_result = aggregate_predictions(all_predictions, user_email, timestamp)
            
            # Save to database
            save_prediction_to_db(final_result, user_email)
            
            return {
                'status': 'completed',
                'session_id': timestamp,
                'user_email': user_email,
                'duration_minutes': duration_minutes,
                'raw_data_path': str(raw_output_path),
                'final_result': final_result
            }
        else:
            return {
                'status': 'no_predictions',
                'session_id': timestamp,
                'user_email': user_email,
                'duration_minutes': duration_minutes,
                'raw_data_path': str(raw_output_path),
                'message': 'No valid predictions were generated'
            }
            
    except Exception as e:
        print(f"Error in run_live_inference: {e}")
        return {
            'status': 'error',
            'session_id': timestamp if 'timestamp' in locals() else 'unknown',
            'user_email': user_email,
            'error': str(e)
        }


def longest_focus_streak(labels):
    max_len = 0
    curr_len = 0

    for label in labels:
        if str(label).strip().lower() == 'concentrating':
            curr_len += 1
            max_len = max(max_len, curr_len)
        else:
            curr_len = 0

    return max_len

def state_switch_count(labels):
    count = 0

    for i in range(1, len(labels)):
        if labels[i - 1] != labels[i]:
            count += 1
    
    return count

def focus_latency(labels):
    latency = 0

    for p in labels:
        if p.strip().lower() != 'concentrating':
            latency += 1
        else:
            break
    
    return latency


def aggregate_predictions(predictions, user_email, session_id):
    """Aggregate multiple prediction results into a single summary."""
    if not predictions:
        return None
    
    # Sum up all metrics
    total_windows = sum(p.get('n_windows', 0) for p in predictions)
    total_seconds = sum(p.get('total_seconds', 0) for p in predictions)
    relaxed_seconds = sum(p.get('relaxed_seconds', 0) for p in predictions)
    neutral_seconds = sum(p.get('neutral_seconds', 0) for p in predictions)
    concentrating_seconds = sum(p.get('concentrating_seconds', 0) for p in predictions)
    
    # Average confidence
    confidences = [p.get('confidence', 0) for p in predictions if p.get('confidence') is not None]
    avg_confidence = np.mean(confidences) if confidences else 0.0
    
    # Determine overall predicted label (most time spent)
    state_times = {
        'relaxed': relaxed_seconds,
        'neutral': neutral_seconds,
        'concentrating': concentrating_seconds
    }
    predicted_label = max(state_times, key=state_times.get)
    window_labels = []
    for p in predictions:
        window_labels.extend(p.get('window_labels', []))
    
    
    
    return {
        'session_id': session_id,
        'user_email': user_email,
        'n_windows': total_windows,
        'total_seconds': total_seconds,
        'relaxed_seconds': relaxed_seconds,
        'neutral_seconds': neutral_seconds,
        'concentrating_seconds': concentrating_seconds,
        'predicted_label': predicted_label,
        'longest_focus_streak': longest_focus_streak(window_labels) * 0.5,
        'focus_latency': focus_latency(window_labels) * 0.5,
        'state_switch_count': state_switch_count(window_labels),
        'confidence': avg_confidence,
        'timestamp': timezone.now().isoformat()
    }


def save_prediction_to_db(result, user_email):
    """Save prediction results to database using SessionSummary."""
    try:
        # Get user profile
        user_profile = UserProfile.objects.get(email=user_email)
        
        # Create or update SessionSummary record
        session_summary, created = SessionSummary.objects.update_or_create(
            session_id=result['session_id'],
            defaults={
                'user': user_profile,
                'csv_file_path': result.get('csv_file_path', ''),
                'start_time': result.get('start_time', timezone.now()),
                'end_time': result.get('end_time', timezone.now()),
                'session_date': result.get('start_time', timezone.now()).date(),
                'total_duration_seconds': result['total_seconds'],
                'average_focus_score': result.get('focus_score', 0.5),
                'peak_focus_score': result.get('peak_focus_score', 0.5),
                'relaxed_seconds': result['relaxed_seconds'],
                'neutral_seconds': result['neutral_seconds'],
                'concentrating_seconds': result['concentrating_seconds'],
                'data_points_count': result['n_windows'],
                'longest_focus_streak': result.get('focus_streak', 0.0),
                'focus_latency': result.get('focus_latency', 0.0),
                'state_switch_count': result.get('state_switch_count', 0),
                'avg_confidence': result.get('confidence', 0.0)
            }
        )
        
        print(f"Saved session summary to database: {session_summary.session_id}")
        return session_summary
        
    except UserProfile.DoesNotExist:
        print(f"User profile not found for email: {user_email}")
        return None
    except Exception as e:
        print(f"Error saving prediction to database: {e}")
        return None


@shared_task
def get_task_status(task_id):
    """Get status of a Celery task."""
    from celery.result import AsyncResult
    task = AsyncResult(task_id)
    return {
        'task_id': task_id,
        'status': task.status,
        'result': task.result if task.ready() else None
    }
