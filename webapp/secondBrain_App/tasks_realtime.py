import os
import sys
import time
import csv
import threading
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
from secondBrain_App.models import UserProfile, SessionSummary

# Constants
LABEL_MAP = {0: 'relaxed', 1: 'neutral', 2: 'concentrating'}
FOCUS_SCORES = {'relaxed': 0.3, 'neutral': 0.6, 'concentrating': 1.0}


class EEGDataStreamer:
    """Handles real-time EEG data streaming and CSV saving"""
    
    def __init__(self, csv_file_path, session_id):
        self.csv_file_path = Path(csv_file_path)
        self.session_id = session_id
        self.csv_file = None
        self.csv_writer = None
        self.is_writing = False
        self.lock = threading.Lock()
        
        # Ensure directory exists
        self.csv_file_path.parent.mkdir(parents=True, exist_ok=True)
        
    def start_recording(self):
        """Initialize CSV file for recording"""
        with self.lock:
            if not self.is_writing:
                self.csv_file = open(self.csv_file_path, 'w', newline='')
                self.csv_writer = csv.writer(self.csv_file)
                
                # Write header
                header = [
                    'timestamp', 'focus_state', 'confidence_score', 'focus_score',
                    'relaxed_prob', 'neutral_prob', 'concentrating_prob'
                ]
                self.csv_writer.writerow(header)
                self.is_writing = True
                
                print(f"Started CSV recording: {self.csv_file_path}")
    
    def write_data_point(self, timestamp, focus_state, confidence, probabilities):
        """Write a single data point to CSV"""
        with self.lock:
            if self.is_writing and self.csv_writer:
                focus_score = FOCUS_SCORES.get(focus_state, 0.5)
                relaxed_prob, neutral_prob, concentrating_prob = probabilities
                
                row = [
                    timestamp,
                    focus_state,
                    round(confidence, 3),
                    round(focus_score, 3),
                    round(relaxed_prob, 3),
                    round(neutral_prob, 3),
                    round(concentrating_prob, 3)
                ]
                self.csv_writer.writerow(row)
    
    def stop_recording(self):
        """Close CSV file"""
        with self.lock:
            if self.csv_file:
                self.csv_file.close()
                self.is_writing = False
                print(f"Stopped CSV recording: {self.csv_file_path}")


def validate_eeg_connection():
    """Check if EEG device is properly connected"""
    try:
        from .services.eeg_acquirer import EEGAcquirer
        
        # Try to initialize EEG acquirer
        acq = EEGAcquirer(max_seconds=5, sfreq=256)
        
        # Try to connect (this will fail if no device)
        acq.connect()
        acq.start()
        
        # Try to get some data (quick test)
        time.sleep(1)
        buffer = acq.get_buffer_copy()
        
        # Clean up
        acq.stop()
        
        if len(buffer) > 10:  # Got some data
            print("EEG device connection validated")
            return True, None
        else:
            return False, "EEG Device not connected. Please check your hardware."
            
    except Exception as e:
        return False, f"EEG Device connection failed: {str(e)}"


@shared_task(bind=True)
def run_live_inference_streaming(self, user_email, duration_minutes=1):
    """
    Real-time EEG inference with per-second focus streaming
    """
    try:
        # Generate unique session info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_prefix = user_email.split('@')[0]
        session_id = f"{user_prefix}_{timestamp}"
        csv_output_path = ROOT / 'dataset' / 'our_data' / f'{user_prefix}_new' / f'{session_id}.csv'
        
        print(f"Starting Real-time EEG Inference")
        print(f"   User: {user_email}")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Session ID: {session_id}")
        print(f"   CSV Path: {csv_output_path}")
        
        # Validate EEG connection first
        is_connected, error_msg = validate_eeg_connection()
        if not is_connected:
            return {
                'status': 'error',
                'session_id': session_id,
                'user_email': user_email,
                'error': error_msg,
                'error_type': 'hardware_connection'
            }
        
        # Initialize data streamer
        streamer = EEGDataStreamer(csv_output_path, session_id)
        streamer.start_recording()
        
        # Initialize prediction service
        models_dir = ROOT / 'core_engine' / 'artifacts' 
        prediction_service = PredictionService(models_dir=models_dir, model_name='xgboost')
        
        # Initialize EEG acquirer
        from .services.eeg_acquirer import EEGAcquirer
        acq = EEGAcquirer(max_seconds=duration_minutes * 60, sfreq=256)
        
        # Connect and start acquisition
        acq.connect()
        acq.start()
        
        # Session tracking
        start_time = datetime.now()
        end_time = start_time + pd.Timedelta(minutes=duration_minutes)
        data_points = []
        focus_scores = []
        
        print(f"Starting real-time inference...")
        print(f"   Start: {start_time.strftime('%H:%M:%S')}")
        print(f"   End: {end_time.strftime('%H:%M:%S')}")
        
        # Main processing loop - run every second
        stop_key = f"stop eeg task{self.request.id}"
        while datetime.now() < end_time:
            # if cache.get(stop_key, False):
            #     print(f"stop requested{self.request.id}")
            #     break
            try:
                # Get buffer data
                rows = acq.get_buffer_copy()
                
                # Need minimum samples for prediction (approximately 1 second)
                min_samples = int(256 * 1.0)
                if len(rows) < min_samples:
                    time.sleep(0.1)
                    continue
                
                # Process prediction
                result = prediction_service.run(rows, nsamples=150, period=1.0, cols_to_ignore=-1)
                
                if result.get('ok'):
                    predicted_label = result.get('predicted_label')
                    confidence = result.get('confidence', 0)
                    
                    # Get probabilities if available
                    probabilities = [0.33, 0.33, 0.34]  # Default fallback
                    if hasattr(prediction_service.predictor.model, 'predict_proba'):
                        try:
                            # This would require model access to get actual probabilities
                            pass
                        except:
                            pass
                    
                    # Calculate focus score
                    focus_score = FOCUS_SCORES.get(predicted_label, 0.5)
                    
                    # Store data point
                    current_time = datetime.now().strftime('%H:%M:%S')
                    data_points.append({
                        'timestamp': current_time,
                        'focus_state': predicted_label,
                        'confidence': confidence,
                        'focus_score': focus_score
                    })
                    focus_scores.append(focus_score)
                    
                    # Write to CSV
                    streamer.write_data_point(current_time, predicted_label, confidence, probabilities)
                    
                    # Broadcast to frontend (could use WebSocket or Redis pub/sub)
                    self.update_state(
                        state='running',
                        session_id=session_id,
                        current_focus=predicted_label,
                        confidence=confidence,
                        focus_score=focus_score,
                        elapsed_seconds=(datetime.now() - start_time).total_seconds()
                    )
                    
                    
                    print(f"   {current_time} | {predicted_label:12} | {confidence:.2f} | {focus_score:.2f}")
                
                time.sleep(1.0)  # Process every second
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(1.0)
                continue
        
        # Stop acquisition
        acq.stop()
        streamer.stop_recording()
        
        # Calculate session summary
        actual_end_time = datetime.now()
        total_duration = (actual_end_time - start_time).total_seconds()
        
        # Calculate statistics
        avg_focus = np.mean(focus_scores) if focus_scores else 0.5
        peak_focus = np.max(focus_scores) if focus_scores else 0.5
        
        # Count time spent in each state
        state_counts = {'relaxed': 0, 'neutral': 0, 'concentrating': 0}
        for point in data_points:
            state = point.get('focus_state', 'neutral')
            if state in state_counts:
                state_counts[state] += 1
        
        # Convert to seconds (assuming 1 point per second)
        relaxed_seconds = state_counts['relaxed']
        neutral_seconds = state_counts['neutral']
        concentrating_seconds = state_counts['concentrating']
        
        # Save session summary to database
        save_session_summary({
            'session_id': session_id,
            'user_email': user_email,
            'task_id': self.request.id,
            'csv_file_path': str(csv_output_path),
            'start_time': start_time,
            'end_time': actual_end_time,
            'total_duration_seconds': total_duration,
            'average_focus_score': avg_focus,
            'peak_focus_score': peak_focus,
            'relaxed_seconds': relaxed_seconds,
            'neutral_seconds': neutral_seconds,
            'concentrating_seconds': concentrating_seconds,
            'data_points_count': len(data_points)
        })
        
        # Final state update
        self.update_state(
            state='completed',
            session_id=session_id,
            final_summary={
                'average_focus_score': avg_focus,
                'peak_focus_score': peak_focus,
                'total_duration_seconds': total_duration,
                'relaxed_seconds': relaxed_seconds,
                'neutral_seconds': neutral_seconds,
                'concentrating_seconds': concentrating_seconds,
                'data_points_count': len(data_points)
            }
        )
        
        return {
            'status': 'completed',
            'session_id': session_id,
            'user_email': user_email,
            'duration_minutes': duration_minutes,
            'csv_file_path': str(csv_output_path),
            'final_summary': {
                'average_focus_score': avg_focus,
                'peak_focus_score': peak_focus,
                'total_duration_seconds': total_duration,
                'relaxed_seconds': relaxed_seconds,
                'neutral_seconds': neutral_seconds,
                'concentrating_seconds': concentrating_seconds,
                'data_points_count': len(data_points)
            }
        }
        
    except Exception as e:
        print(f"Error in real-time inference: {e}")
        return {
            'status': 'error',
            'session_id': session_id if 'session_id' in locals() else 'unknown',
            'user_email': user_email,
            'error': str(e),
            'error_type': 'processing_error'
        }


def save_session_summary(summary_data):
    """Save session summary to database"""
    try:
        user_profile = UserProfile.objects.get(email=summary_data['user_email'])
        
        session_summary = SessionSummary.objects.create(
            session_id=summary_data['session_id'],
            user=user_profile,
            task_id=summary_data.get('task_id'),
            csv_file_path=summary_data['csv_file_path'],
            start_time=summary_data['start_time'],
            end_time=summary_data['end_time'],
            total_duration_seconds=summary_data['total_duration_seconds'],
            average_focus_score=summary_data['average_focus_score'],
            peak_focus_score=summary_data['peak_focus_score'],
            relaxed_seconds=summary_data['relaxed_seconds'],
            neutral_seconds=summary_data['neutral_seconds'],
            concentrating_seconds=summary_data['concentrating_seconds'],
            data_points_count=summary_data['data_points_count']
        )
        
        print(f"Session summary saved to database: {session_summary.session_id}")
        return session_summary
        
    except UserProfile.DoesNotExist:
        print(f"User profile not found: {summary_data['user_email']}")
        return None
    except Exception as e:
        print(f"Error saving session summary: {e}")
        return None


@shared_task(bind=True)
def get_task_status(task_id):
    """Get status of a Celery task"""
    from celery.result import AsyncResult
    task = AsyncResult(task_id)
    return {
        'task_id': task_id,
        'status': task.status,
        'result': task.result if task.ready() else None
    }
