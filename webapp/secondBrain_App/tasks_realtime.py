import os
import sys
import time
import csv
import threading
from pathlib import Path
from datetime import datetime
from celery import shared_task
from django.conf import settings
from django.utils import timezone
import pandas as pd
import numpy as np
from django.core.cache import cache

# Add project root and core_engine to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'core_engine'))

# Import EEG feature extraction
from EEG_feature_extraction_adv import get_raw_band_powers
import math

def scale_power(val):
    """Converts septillions into a 0-100 scale using log"""
    try:
        return min(max(float(math.log10(val + 1) * 3), 0), 100)
    except:
        return 0
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
# from secondBrain_App.models import UserProfile, SessionSummary

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


@shared_task(bind=True)
def run_live_inference_streaming(self, user_email, duration_minutes=1):
    """
    Real-time EEG inference with per-second focus streaming
    """
    all_session_labels = []
    try:
        # CACHE CLEANUP: Remove old cache entries for this user to prevent buildup
        try:
            cache.delete(f"live_eeg_stream_{user_email}")
            cache.delete(f"session_final_result_{user_email}")
            cache.delete(f"live_status_{user_email}")
            cache.delete(f"recommendation_{user_email}")
            print(f"[CACHE] Cleaned old entries for {user_email}")
        except Exception as e:
            print(f"[CACHE WARNING] Failed to clean old entries: {e}")
        # Generate unique session info
        print(f"Starting Real-time EEG Inference")
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
        print(f"[DEBUG 1] Streamer started")
        
        # Initialize prediction service
        print(f"[DEBUG 2] Loading prediction service...")
        models_dir = ROOT / 'core_engine' / 'artifacts'
        print("Loading from {0}".format(models_dir))
        prediction_service = PredictionService(models_dir=models_dir, model_name='xgboost')
        print(f"[DEBUG 3] Prediction service loaded")
        
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
        
        # CACHE OPTIMIZATION: Throttle cache updates to reduce pressure
        last_cache_update = 0
        cache_update_interval = 2  # Update cache every 2 seconds instead of every second
        
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
                
                try:
                    # Process prediction
                    print(f"[DEBUG] Processing prediction with {len(rows)} rows...")
                    result = prediction_service.run(rows, nsamples=150, period=1.0, cols_to_ignore=-1)
                    if result.get('ok'):
                        predicted_label = result.get('predicted_label')
                        confidence = result.get('confidence', 0)
                        
                        # Get window labels and append ONLY NEW labels to session-wide accumulation
                        window_labels = result.get('window_labels', [])
                        # Only add new labels to avoid double-counting overlapping windows
                        current_total = len(all_session_labels)
                        all_session_labels.extend(window_labels)
                        new_labels_count = len(all_session_labels) - current_total
                        
                        # Get probabilities from result (now provided by prediction service)
                        probabilities = result.get('probabilities', [0.33, 0.33, 0.34])
                        
                        # Extract raw band powers for live visualization
                        raw_band_powers = get_raw_band_powers(np.array(rows))
                        
                        # Calculate focus score
                        focus_score = FOCUS_SCORES.get(predicted_label, 0.5)
                        
                        # Store data point
                        current_time = datetime.now().strftime('%H:%M:%S')
                        data_points.append({
                            'timestamp': current_time,
                            'focus_state': predicted_label,
                            'confidence': confidence,
                            'focus_score': focus_score,
                            'window_labels': result.get('window_labels', []),
                            'wave_data': raw_band_powers
                        })
                        focus_scores.append(focus_score)
                        
                        # Write to CSV
                        streamer.write_data_point(current_time, predicted_label, confidence, probabilities)
                        
                        # BROADCAST live data with throttling to reduce cache pressure
                        if result.get('ok'):
                            current_time_sec = time.time()
                            
                            # CACHE THROTTLING: Only update cache every 2 seconds
                            if current_time_sec - last_cache_update >= cache_update_interval:
                                # Normalize those massive septillion values to 0-100 for the UI
                                def normalize(v):
                                    try: return min(max(float(math.log10(v + 1) * 3), 0), 100)
                                    except: return 0

                                # OPTIMIZED: Reduce cache data size and increase timeout
                                live_package = {
                                    'status': 'active',
                                    'state': result.get('predicted_label', 'NEUTRAL').upper(),
                                    'confidence': round(result.get('confidence', 0) * 100, 1),
                                    'waves': {
                                        'delta': normalize(raw_band_powers.get('delta', 0)),
                                        'theta': normalize(raw_band_powers.get('theta', 0)),
                                        'alpha': normalize(raw_band_powers.get('alpha', 0)),
                                        'beta': normalize(raw_band_powers.get('beta', 0)),
                                        'gamma': normalize(raw_band_powers.get('gamma', 0))
                                    },
                                    'last_updated': timezone.now().strftime("%H:%M:%S")
                                }
                                
                                # BROADCAST: Use consistent key with longer timeout
                                cache_key = f"live_eeg_stream_{user_email}"
                                cache.set(cache_key, live_package, timeout=30)  # Increased from 5 to 30 seconds
                                print(f"[BROADCAST] Updated live stream: {live_package.get('state', 'UNKNOWN')} {live_package.get('confidence', 0):.1f}%")
                                
                                last_cache_update = current_time_sec
                        else:
                            print(f"[ERROR] Prediction failed")
                        
                        print(f"   {current_time} | {predicted_label:12} | {confidence:.2f} | {focus_score:.2f}")
                    else:
                        print(f"[ERROR] Prediction returned not ok")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process prediction: {e}")
                    import traceback
                    traceback.print_exc()
                
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
        
        # Calculate statistics - NO DUMMY FALLBACKS
        if not all_session_labels:
            print(f"[ERROR] No data captured during session - all_session_labels is empty!")
            avg_focus = 0.0
            peak_focus = 0.0
        else:
            avg_focus = np.mean(focus_scores) if focus_scores else 0.0
            peak_focus = np.max(focus_scores) if focus_scores else 0.0
        
        # Collect all window labels for proper statistics
        print(f"[DEBUG] Final all_session_labels count: {len(all_session_labels)}")
        print(f"[DEBUG] Sample labels: {all_session_labels[:10] if all_session_labels else 'None'}")
        print(f"[DEBUG] Total data_points collected: {len(data_points)}")
        print(f"[DEBUG] Total focus_scores collected: {len(focus_scores)}")
        
        if len(all_session_labels) == 0:
            print(f"[ERROR] No labels captured! Checking prediction service results...")
            # Let's debug what went wrong
            if len(data_points) == 0:
                print(f"[ERROR] No data points were created - predictions likely failed")
            else:
                print(f"[DEBUG] Data points exist but labels empty - checking window_labels...")
                for i, point in enumerate(data_points[:3]):  # Check first 3 points
                    print(f"[DEBUG] Point {i}: {point}")
        
        # Count time spent in each state based on accumulated session labels (each window = 0.5 seconds)
        state_counts = {'relaxed': 0, 'neutral': 0, 'concentrating': 0}
        for label in all_session_labels:
            if label in state_counts:
                state_counts[label] += 0.5  # Each window represents 0.5 seconds
        
        print(f"[DEBUG] State counts: {state_counts}")
        
        # Calculate statistics based on accumulated session labels - NO DUMMY FALLBACKS
        lfocus_streak = longest_focus_streak(all_session_labels) * 0.5 if all_session_labels else 0.0
        switch_count = state_switch_count(all_session_labels) if all_session_labels else 0
        latency = focus_latency(all_session_labels) * 0.5 if all_session_labels else 0.0

        # Convert to seconds - NO DUMMY FALLBACKS
        relaxed_seconds = state_counts['relaxed']
        neutral_seconds = state_counts['neutral']
        concentrating_seconds = state_counts['concentrating']
        
        # Sanity check: ensure calculated state seconds don't exceed total duration
        calculated_total = relaxed_seconds + neutral_seconds + concentrating_seconds
        if calculated_total > total_duration * 1.1:  # Allow 10% tolerance
            print(f"[WARNING] Calculated state time ({calculated_total}s) exceeds session duration ({total_duration}s)")
            # Scale proportionally to fit total duration
            scale_factor = total_duration / calculated_total
            relaxed_seconds *= scale_factor
            neutral_seconds *= scale_factor
            concentrating_seconds *= scale_factor
            print(f"[DEBUG] Scaled to fit duration - Relaxed: {relaxed_seconds:.1f}, Neutral: {neutral_seconds:.1f}, Concentrating: {concentrating_seconds:.1f}")
        
        print(f"[DEBUG] Final seconds - Relaxed: {relaxed_seconds}, Neutral: {neutral_seconds}, Concentrating: {concentrating_seconds}")
        
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
            'longest_focus_streak': lfocus_streak,
            'state_switch_count': switch_count,
            'focus_latency': latency,
            'avg_confidence': np.mean(focus_scores) if focus_scores else 0.0,
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
                'longest_focus_streak': lfocus_streak,
                'state_switch_count': switch_count,
                'focus_latency': latency,
                'data_points_count': len(data_points)
            }
        )
        
        # Prepare final summary for caching and return
        final_summary = {
            'average_focus_score': avg_focus,
            'peak_focus_score': peak_focus,
            'total_duration_seconds': total_duration,
            'relaxed_seconds': relaxed_seconds,
            'neutral_seconds': neutral_seconds,
            'concentrating_seconds': concentrating_seconds,
            'longest_focus_streak': lfocus_streak,
            'state_switch_count': switch_count,
            'focus_latency': latency,
            'data_points_count': len(data_points)
        }
        
        # Save final result to Django cache for frontend access
        cache.set(f"session_final_result_{user_email}", {
            'status': 'completed',
            'summary': final_summary
        }, timeout=300)
        
        print(f"[DEBUG] Final summary cached for {user_email}: {final_summary}")
        
        return {
            'status': 'completed',
            'session_id': session_id,
            'user_email': user_email,
            'duration_minutes': duration_minutes,
            'csv_file_path': str(csv_output_path),
            'final_summary': final_summary
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

# tasks_realtime.py

def save_session_summary(summary_data):
    """Save session summary to database"""
    from secondBrain_App.models import UserProfile, SessionSummary
    try:
        print(f"[DB] Attempting to save session summary for: {summary_data['session_id']}")
        print(f"[DB] User email: {summary_data['user_email']}")
        print(f"[DB] Data points count: {summary_data['data_points_count']}")
        print(f"[DB] Neutral seconds: {summary_data['neutral_seconds']}")
        
        user_profile = UserProfile.objects.get(email=summary_data['user_email'])
        print(f"[DB] Found user profile: {user_profile.email}")

        session_summary = SessionSummary.objects.create(
            session_id              = summary_data['session_id'],
            user                    = user_profile,
            task_id                 = summary_data.get('task_id'),
            csv_file_path           = summary_data['csv_file_path'],
            start_time              = summary_data['start_time'],
            end_time                = summary_data['end_time'],
            session_date            = summary_data['start_time'].date(),
            total_duration_seconds  = summary_data['total_duration_seconds'],
            average_focus_score     = summary_data['average_focus_score'],
            peak_focus_score        = summary_data['peak_focus_score'],
            relaxed_seconds         = summary_data['relaxed_seconds'],
            neutral_seconds         = summary_data['neutral_seconds'],
            concentrating_seconds   = summary_data['concentrating_seconds'],
            longest_focus_streak    = summary_data.get('longest_focus_streak', 0.0),
            focus_latency           = summary_data.get('focus_latency', 0.0),
            state_switch_count      = summary_data.get('state_switch_count', 0),
            avg_confidence         = summary_data.get('avg_confidence', 0.0),
            data_points_count       = summary_data['data_points_count']
        )

        print(f"[SESSION] Summary saved successfully: {session_summary.session_id}")
        print(f"[SESSION] SessionSummary ID: {session_summary.id}")

        # ---- STEP 1: Update MySQL user_summary table ----──
        try:
            from core_engine.recommendation.user_summary import main as update_summary
            update_summary(user_id=summary_data['user_email'])
            print(f"[SUMMARY] user_summary table updated")
        except Exception as e:
            print(f"[SUMMARY ERROR] {e}")
        
        # ---- CRITICAL: Ensure UI gets completion signal even if summary fails ----
        try:
            from django.core.cache import cache
            cache.set(f"live_status_{summary_data['user_email']}", "completed", timeout=300)
            print(f"[STATUS] Live status set to 'completed' for {summary_data['user_email']}")
        except Exception as e:
            print(f"[STATUS ERROR] Failed to set live status: {e}")

        # ── STEP 2: Generate recommendation using updated summary ──
        try:
            from core_engine.recommendation import generate_recommendation_for_session
            recommendation_text = generate_recommendation_for_session(
                user_email    = summary_data['user_email'],
                session_id    = summary_data['session_id'],
                final_summary = {
                    'average_focus_score'  : summary_data['average_focus_score'],
                    'concentrating_seconds': summary_data['concentrating_seconds'],
                    'neutral_seconds'      : summary_data['neutral_seconds'],
                    'relaxed_seconds'      : summary_data['relaxed_seconds']
                }
            )
            print(f"[RECOMMENDATION] Generated successfully")

            from secondBrain_App.models import Recommendation
            Recommendation.objects.create(
                user=user_profile,
                session_id=summary_data['session_id'],
                message=recommendation_text
            )
            # ----------------------------------------

            print(f"[RECOMMENDATION] Generated and Saved to DB successfully")

            # Save recommendation text to cache so view can retrieve it
            from django.core.cache import cache
            cache_key = f"recommendation_{summary_data['user_email']}"
            cache.set(cache_key, {
                'text'      : recommendation_text,
                'session_id': summary_data['session_id']
            }, timeout=3600)  # store for 1 hour

        except Exception as e:
            print(f"[RECOMMENDATION ERROR] {e}")

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