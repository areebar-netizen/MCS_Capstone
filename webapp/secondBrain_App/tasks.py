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
from EEG_feature_extraction_adv import generate_feature_vectors_from_matrix
from enhanced_feature_extraction import load_preprocessing_artifacts, apply_feature_pipeline

# Import Django models
from secondBrain_App.models import UserProfile, Prediction

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_prefix = user_email.split('@')[0]
        raw_output_path = ROOT / 'dataset' / 'our_data' / f'{user_prefix}_new' / f'{user_prefix}_{timestamp}_session.csv'
        
        # Ensure output directory exists
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize prediction service
        models_dir = ROOT / 'core_engine' / 'artifacts' / 'models_out'
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
    
    return {
        'session_id': session_id,
        'user_email': user_email,
        'n_windows': total_windows,
        'total_seconds': total_seconds,
        'relaxed_seconds': relaxed_seconds,
        'neutral_seconds': neutral_seconds,
        'concentrating_seconds': concentrating_seconds,
        'predicted_label': predicted_label,
        'confidence': avg_confidence,
        'timestamp': datetime.now().isoformat()
    }


def save_prediction_to_db(result, user_email):
    """Save prediction results to database."""
    try:
        # Get user profile
        user_profile = UserProfile.objects.get(email=user_email)
        
        # Create prediction record
        prediction = Prediction.objects.create(
            user=user_profile,
            session_id=result['session_id'],
            predicted_label=result['predicted_label'],
            confidence=result['confidence'],
            n_windows=result['n_windows'],
            total_seconds=result['total_seconds'],
            relaxed_seconds=result['relaxed_seconds'],
            neutral_seconds=result['neutral_seconds'],
            concentrating_seconds=result['concentrating_seconds']
        )
        
        print(f"Saved prediction to database: {prediction.prediction_id}")
        return prediction
        
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
