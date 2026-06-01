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
                
                # Write header with raw EEG data columns
                header = [
                    'timestamp', 'focus_state', 'confidence_score', 'focus_score',
                    'relaxed_prob', 'neutral_prob', 'concentrating_prob',
                    'TP9', 'AF7', 'AF8', 'TP10', 'Right_AUX'
                ]
                self.csv_writer.writerow(header)
                self.is_writing = True
    def write_data_point(self, timestamp, focus_state, confidence, probabilities, raw_eeg=None):
        """Write a single data point to CSV"""
        with self.lock:
            if self.is_writing and self.csv_writer:
                focus_score = FOCUS_SCORES.get(focus_state, 0.5)
                relaxed_prob, neutral_prob, concentrating_prob = probabilities
                
                # Extract raw EEG values if provided (take average of last sample)
                if raw_eeg is not None and len(raw_eeg) > 0:
                    # Get the last row of EEG data
                    last_sample = raw_eeg[-1]
                    # EEG data format: [timestamp, TP9, AF7, AF8, TP10, Right_AUX]
                    tp9 = last_sample[1] if len(last_sample) > 1 else 0
                    af7 = last_sample[2] if len(last_sample) > 2 else 0
                    af8 = last_sample[3] if len(last_sample) > 3 else 0
                    tp10 = last_sample[4] if len(last_sample) > 4 else 0
                    right_aux = last_sample[5] if len(last_sample) > 5 else 0
                else:
                    tp9, af7, af8, tp10, right_aux = 0, 0, 0, 0, 0
                
                row = [
                    timestamp,
                    focus_state,
                    round(confidence, 3),
                    round(focus_score, 3),
                    round(relaxed_prob, 3),
                    round(neutral_prob, 3),
                    round(concentrating_prob, 3),
                    round(tp9, 3),
                    round(af7, 3),
                    round(af8, 3),
                    round(tp10, 3),
                    round(right_aux, 3)
                ]
                self.csv_writer.writerow(row)
    
    def stop_recording(self):
        """Close CSV file"""
        with self.lock:
            if self.csv_file:
                self.csv_file.close()
                self.is_writing = False


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
def run_live_inference_streaming(self, user_email, duration_minutes=1, session_id=None, from_expo = False):
    """
    Real-time EEG inference with per-second focus streaming
    """
    # Initialize state_counts to ensure it's available in all code paths
    state_counts = {'relaxed': 0, 'neutral': 0, 'concentrating': 0}
    
    all_session_labels = []
    try:
        # CACHE CLEANUP: Remove old cache entries for this user to prevent buildup
        try:
            cache.delete(f"live_eeg_stream_{user_email}")
            cache.delete(f"session_final_result_{user_email}")
            cache.delete(f"live_status_{user_email}")
            cache.delete(f"recommendation_{user_email}")
        except Exception as e:
            pass
        
        # Set initial cache immediately to show brainwave box right away
        from django.utils import timezone
        initial_package = {
            'status': 'active',
            'state': 'INITIALIZING',
            'confidence': 0.0,
            'waves': {'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0, 'gamma': 0},
            'last_updated': timezone.localtime(timezone.now()).strftime("%H:%M:%S")
        }
        cache_key = f"live_eeg_stream_{user_email}"
        cache.set(cache_key, initial_package, timeout=30)
        print(f"[INIT] Set initial cache package immediately to show brainwave box")
        
        # Generate unique session info if not provided
        user_prefix = user_email.split('@')[0]
        if not session_id:
            timestamp = timezone.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"{user_prefix}_{timestamp}"
        csv_output_path = ROOT / 'dataset' / 'our_data' / f'{user_prefix}_new' / f'{session_id}.csv'
        
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
        start_time = timezone.now()
        end_time = start_time + pd.Timedelta(minutes=duration_minutes)
        data_points = []
        focus_scores = []
        confidence_scores = []
        
        # Main processing loop - run every second
        stop_key = f"stop eeg task{self.request.id}"
        
        # CACHE OPTIMIZATION: Throttle cache updates to reduce pressure
        last_cache_update = 0
        cache_update_interval = 1  # Update cache every 1 second for immediate feedback
        
        # Initialize cache immediately when session starts
        initial_cache_set = False
        
        while timezone.now() < end_time:
            # Check for stop signal from session
            try:
                stop_signal = cache.get(f"stop_eeg_task_{self.request.id}")
                if stop_signal:
                    print(f"[STOP] Stop signal received for task {self.request.id}")
                    break
            except Exception as e:
                print(f"[ERROR] Failed to check stop signal: {e}")
            
            try:
                # Get buffer data
                rows = acq.get_buffer_copy()[-1536:]
                

                # Track EEG buffer size
                if len(rows) == 0:
                    print(f"[DEBUG] Empty EEG buffer, waiting for data...")
                
                # Need minimum samples for prediction (approximately 1 second)
                min_samples = int(256 * 1.0)
                if len(rows) < min_samples:
                    print("[WAIT] Not enough EEG samples yet")
                    time.sleep(1)
                    continue

                
                try:
                    # Process prediction
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
                        
                        # Reduce brainwave debug logging
                        if len(data_points) % 10 == 0:  # Log every 10th data point
                            print(f"[DEBUG] Brainwave: delta:{raw_band_powers.get('delta', 0):.1f} theta:{raw_band_powers.get('theta', 0):.1f} alpha:{raw_band_powers.get('alpha', 0):.1f}")
                        
                        # Calculate focus score
                        focus_score = FOCUS_SCORES.get(predicted_label, 0.5)
                        
                        # Store data point
                        current_time = timezone.now().strftime('%H:%M:%S')
                        data_points.append({
                            'timestamp': current_time,
                            'focus_state': predicted_label,
                            'confidence': confidence,
                            'focus_score': focus_score,
                            'window_labels': result.get('window_labels', []),
                            'wave_data': raw_band_powers
                        })
                        focus_scores.append(focus_score)
                        confidence_scores.append(confidence)
                        
                        # Write to CSV with raw EEG data
                        streamer.write_data_point(current_time, predicted_label, confidence, probabilities, rows)
                        
                        # BROADCAST live data with throttling to reduce cache pressure
                        if result.get('ok'):
                            current_time_sec = time.time()
                            
                            # CACHE THROTTLING: Update immediately on first prediction, then every second
                            should_update_cache = (current_time_sec - last_cache_update >= cache_update_interval) or (last_cache_update == 0)
                            
                            if should_update_cache:
                                # Normalize brainwave values using fixed scale to prevent over-smoothing
                                def normalize_with_fixed_scale(v, band_name):
                                    # Updated ranges based on actual Muse output observed in logs
                                    fixed_scales = {
                                        'delta': (0, 5000),      # Delta: 0-5000 uV²
                                        'theta': (0, 500),       # Theta: 0-500 uV²
                                        'alpha': (0, 2000),      # Alpha: 0-2000 uV²
                                        'beta': (0, 10000),      # Beta: 0-10000 uV² (was hitting 9000+)
                                        'gamma': (0, 3000)       # Gamma: 0-3000 uV²
                                    }
                                    try:
                                        min_val, max_val = fixed_scales.get(band_name, (0, 50000))
                                        if max_val == min_val:
                                            return 0
                                        
                                        # CLIP: Ensure raw value is within valid range before scaling
                                        clipped_val = max(min_val, min(v, max_val))
                                        
                                        # Linear normalization: (clipped / max) * 100
                                        scaled = (clipped_val / max_val) * 100
                                        
                                        return round(max(0, min(100, scaled)), 1)
                                    except Exception as e:
                                        print(f"[SCALE ERROR] {band_name}: {e}")
                                        return 0

                                # Get current brainwave values
                                current_waves = {
                                    'delta': raw_band_powers.get('delta', 0),
                                    'theta': raw_band_powers.get('theta', 0),
                                    'alpha': raw_band_powers.get('alpha', 0),
                                    'beta': raw_band_powers.get('beta', 0),
                                    'gamma': raw_band_powers.get('gamma', 0)
                                }
                                
                                # Debug: Track if values are actually changing
                                print(f"[DEBUG] Raw brainwave values at {current_time}: {current_waves}")
                                
                                # Check if values are the same as previous (detect reuse)
                                if not hasattr(run_live_inference_streaming, '_last_waves'):
                                    run_live_inference_streaming._last_waves = {}
                                
                                waves_changed = False
                                for band, value in current_waves.items():
                                    if band not in run_live_inference_streaming._last_waves or abs(value - run_live_inference_streaming._last_waves[band]) > 0.01:
                                        waves_changed = True
                                        break
                                
                                if not waves_changed:
                                    print(f"[WARNING] Brainwave values haven't changed since last update!")
                                else:
                                    print(f"[INFO] Brainwave values changed - new data detected")
                                
                                run_live_inference_streaming._last_waves = current_waves.copy()
                                
                                # OPTIMIZED: Minimal payload for reduced cache pressure and faster UI updates
                                live_package = {
                                    'status': 'active',
                                    'state': result.get('predicted_label', 'NEUTRAL').upper(),
                                    'confidence': round(result.get('confidence', 0) * 100, 1),
                                    'waves': {
                                        'delta': normalize_with_fixed_scale(current_waves['delta'], 'delta'),
                                        'theta': normalize_with_fixed_scale(current_waves['theta'], 'theta'),
                                        'alpha': normalize_with_fixed_scale(current_waves['alpha'], 'alpha'),
                                        'beta': normalize_with_fixed_scale(current_waves['beta'], 'beta'),
                                        'gamma': normalize_with_fixed_scale(current_waves['gamma'], 'gamma')
                                    },
                                    'last_updated': timezone.localtime(timezone.now()).strftime("%H:%M:%S")
                                }
                                
                                # Debug: Show scaling results
                                print(f"[SCALING] Raw→Scaled: delta:{current_waves['delta']:.1f}→{live_package['waves']['delta']:.1f}% | "
                                      f"theta:{current_waves['theta']:.1f}→{live_package['waves']['theta']:.1f}% | "
                                      f"alpha:{current_waves['alpha']:.1f}→{live_package['waves']['alpha']:.1f}% | "
                                      f"beta:{current_waves['beta']:.1f}→{live_package['waves']['beta']:.1f}% | "
                                      f"gamma:{current_waves['gamma']:.1f}→{live_package['waves']['gamma']:.1f}%")
                                
                                # BROADCAST: Use consistent key with longer timeout
                                cache_key = f"live_eeg_stream_{user_email}"
                                cache.set(cache_key, live_package, timeout=30)  # Increased from 5 to 30 seconds
                                print(f"[BROADCAST] Updated live stream: {live_package.get('state', 'UNKNOWN')} {live_package.get('confidence', 0):.1f}%")
                                
                                last_cache_update = current_time_sec
                        else:
                            print(f"[ERROR] Prediction failed")
                        
                        print(f"   {current_time} | {predicted_label:12} | {confidence:.2f} | {focus_score:.2f}")
                    else:
                        print(f"[ERROR] Prediction returned not ok", result)
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process prediction: {e}")
                    import traceback
                    traceback.print_exc()
                
                time.sleep(1.0)  # Process every second
                
            except Exception as e:
                time.sleep(1.0)
                continue
        
        # Stop acquisition
        acq.stop()
        streamer.stop_recording()
        
        # Calculate session summary
        actual_end_time = timezone.now()
        total_duration = (actual_end_time - start_time).total_seconds()
        
        # Initialize state_counts to ensure it's available in all code paths
        state_counts = {'relaxed': 0, 'neutral': 0, 'concentrating': 0}
        for label in all_session_labels:
            if label in state_counts:
                state_counts[label] += 0.5  # Each window represents 0.5 seconds
        
        
        # Calculate statistics - NO DUMMY FALLBACKS
        if not all_session_labels:
            avg_focus = 0.0
            peak_focus = 0.0
        else:
            # Calculate weighted Average Focus Score based on state values
            # State values: Concentrating=10, Neutral=5, Relaxed=2
            relaxed_seconds = state_counts.get('relaxed', 0)
            neutral_seconds = state_counts.get('neutral', 0) 
            concentrating_seconds = state_counts.get('concentrating', 0)
            
            total_time = relaxed_seconds + neutral_seconds + concentrating_seconds
            if total_time > 0:
                # Weighted calculation: (state_seconds × state_value) summed, then divided by total_time
                total_focus_points = (relaxed_seconds * 2) + (neutral_seconds * 5) + (concentrating_seconds * 10)
                avg_focus = (total_focus_points / total_time)
        
        # Count time spent in each state based on accumulated session labels (each window = 0.5 seconds)
        state_counts = {'relaxed': 0, 'neutral': 0, 'concentrating': 0}
        for label in all_session_labels:
            if label in state_counts:
                state_counts[label] += 0.5  # Each window represents 0.5 seconds
        
        # Calculate statistics based on accumulated session labels - NO DUMMY FALLBACKS
        
        # Peak Focus Score: Longest continuous concentration streak relative to ideal performance
        longest_streak_seconds = longest_focus_streak(all_session_labels) * 0.5 if all_session_labels else 0.0
        total_session_seconds = total_duration
        
        # Scientific 1-10 scale based on streak quality:
        # 10 = 30+ second continuous concentration (excellent deep work)
        # 8 = 20-30 seconds (very good)
        # 6 = 10-20 seconds (good)
        # 4 = 5-10 seconds (fair)
        # 2 = <5 seconds (poor)
        if longest_streak_seconds >= 30:
            peak_focus_score = 10.0
        elif longest_streak_seconds >= 20:
            peak_focus_score = 8.0
        elif longest_streak_seconds >= 10:
            peak_focus_score = 6.0
        elif longest_streak_seconds >= 5:
            peak_focus_score = 4.0
        else:
            peak_focus_score = 2.0
        
        # Update peak_focus to use the new calculation
        peak_focus = peak_focus_score
        lfocus_streak = longest_streak_seconds
        
        # Collect all window labels for proper statistics
        
        # Debug state switch calculation
        switch_count = state_switch_count(all_session_labels) if all_session_labels else 0
        
        # Debug focus latency calculation
        latency_windows = focus_latency(all_session_labels) if all_session_labels else 0
        latency = latency_windows * 0.5

        # Convert to seconds - NO DUMMY FALLBACKS
        relaxed_seconds = state_counts.get('relaxed', 0)
        neutral_seconds = state_counts.get('neutral', 0)
        concentrating_seconds = state_counts.get('concentrating', 0)
        
        # Sanity check: ensure calculated state seconds don't exceed total duration
        calculated_total = relaxed_seconds + neutral_seconds + concentrating_seconds
        if calculated_total > total_duration * 1.1:  # Allow 10% tolerance
            # Scale proportionally to fit total duration
            scale_factor = total_duration / calculated_total
            relaxed_seconds *= scale_factor
            neutral_seconds *= scale_factor
            concentrating_seconds *= scale_factor
        
        # Final seconds calculated
        
        # Save session summary to database
        if(not from_expo):
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
                'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
                'data_points_count': len(data_points),
                'data_points': data_points
            })
        else:
            print("[FROM EXPO] no summary and recommendations saved")
            
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
            'average_focus_score': float(avg_focus),
            'peak_focus_score': float(peak_focus),
            'total_duration_seconds': float(total_duration),
            'relaxed_seconds': float(relaxed_seconds),
            'neutral_seconds': float(neutral_seconds),
            'concentrating_seconds': float(concentrating_seconds),
            'longest_focus_streak': float(lfocus_streak),
            'state_switch_count': int(switch_count),
            'focus_latency': float(latency),
            'data_points_count': int(len(data_points))
        }
        
        # Save final result to Django cache for frontend access
        cache.set(f"session_final_result_{user_email}", {
            'status': 'completed',
            'summary': final_summary
        }, timeout=300)
        
        # Final summary cached
        
        return {
            'status': 'completed',
            'session_id': session_id,
            'user_email': user_email,
            'duration_minutes': duration_minutes,
            'csv_file_path': str(csv_output_path),
            'final_summary': final_summary
        }
        
    except Exception as e:
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
    print(f"[DEBUG] save_session_summary called for session: {summary_data['session_id']}")
    print(f"[DEBUG] User: {summary_data['user_email']}")
    print(f"[DEBUG] Duration: {summary_data['total_duration_seconds']} seconds")
    
    from secondBrain_App.models import UserProfile, SessionSummary
    try:
        user_profile = UserProfile.objects.get(email=summary_data['user_email'])

        # ── STEP 1: Retrieve wave averages from cache (pre-calculated by live_predict) ──
        # live_predict.py calculates filtered wave averages during the session and stores them in cache
        # This ensures we use the same filtered values that were used during live prediction
        
        beta_avg = 0
        gamma_avg = 0
        alpha_avg = 0
        theta_avg = 0
        
        try:
            from django.core.cache import cache
            wave_averages_cache_key = f"wave_averages_{summary_data['user_email']}"
            cached_wave_averages = cache.get(wave_averages_cache_key)
            
            if cached_wave_averages:
                beta_avg = cached_wave_averages.get('beta_avg', 0)
                gamma_avg = cached_wave_averages.get('gamma_avg', 0)
                alpha_avg = cached_wave_averages.get('alpha_avg', 0)
                theta_avg = cached_wave_averages.get('theta_avg', 0)
                print(f"[WAVE AVERAGES] Retrieved from cache - Beta: {beta_avg:.2f}Hz, Gamma: {gamma_avg:.2f}Hz, Alpha: {alpha_avg:.2f}Hz, Theta: {theta_avg:.2f}Hz")
            else:
                # Fallback: Calculate from data_points if cache not available
                print(f"[WAVE AVERAGES] Cache miss, calculating from data_points")
                data_points = summary_data.get('data_points', [])
                beta_values = []
                gamma_values = []
                alpha_values = []
                theta_values = []
                
                for point in data_points:
                    wave_data = point.get('wave_data', {})
                    if wave_data:
                        beta_values.append(wave_data.get('beta', 0))
                        gamma_values.append(wave_data.get('gamma', 0))
                        alpha_values.append(wave_data.get('alpha', 0))
                        theta_values.append(wave_data.get('theta', 0))
                
                beta_avg = np.mean(beta_values) if beta_values else 0
                gamma_avg = np.mean(gamma_values) if gamma_values else 0
                alpha_avg = np.mean(alpha_values) if alpha_values else 0
                theta_avg = np.mean(theta_values) if theta_values else 0
                print(f"[WAVE AVERAGES] Calculated from data_points - Beta: {beta_avg:.2f}Hz, Gamma: {gamma_avg:.2f}Hz, Alpha: {alpha_avg:.2f}Hz, Theta: {theta_avg:.2f}Hz")
        except Exception as e:
            print(f"[ERROR] Failed to retrieve wave averages from cache: {e}")
            # Fallback to calculation from data_points
            data_points = summary_data.get('data_points', [])
            beta_values = []
            gamma_values = []
            alpha_values = []
            theta_values = []
            
            for point in data_points:
                wave_data = point.get('wave_data', {})
                if wave_data:
                    beta_values.append(wave_data.get('beta', 0))
                    gamma_values.append(wave_data.get('gamma', 0))
                    alpha_values.append(wave_data.get('alpha', 0))
                    theta_values.append(wave_data.get('theta', 0))
            
            beta_avg = np.mean(beta_values) if beta_values else 0
            gamma_avg = np.mean(gamma_values) if gamma_values else 0
            alpha_avg = np.mean(alpha_values) if alpha_values else 0
            theta_avg = np.mean(theta_values) if theta_values else 0
            print(f"[WAVE AVERAGES] Fallback calculation - Beta: {beta_avg:.2f}Hz, Gamma: {gamma_avg:.2f}Hz, Alpha: {alpha_avg:.2f}Hz, Theta: {theta_avg:.2f}Hz")
        
        # Calculate high-level string inferences
        inferences = calculate_inferences(beta_avg, gamma_avg, alpha_avg, theta_avg)
        print(f"[INFERENCE] Neural State: {inferences['neural_state']}, Signal Integrity: {inferences['signal_integrity']}, Focus Depth: {inferences['focus_depth']}")
        
        # Add wave averages and inferences to summary_data
        summary_data['beta_avg'] = beta_avg
        summary_data['gamma_avg'] = gamma_avg
        summary_data['alpha_avg'] = alpha_avg
        summary_data['theta_avg'] = theta_avg
        summary_data['neural_state'] = inferences['neural_state']
        summary_data['signal_integrity'] = inferences['signal_integrity']
        summary_data['focus_depth'] = inferences['focus_depth']

        print(f"[DEBUG] Creating SessionSummary for session: {summary_data['session_id']}")
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
            data_points_count       = summary_data['data_points_count'],
            longest_focus_streak    = summary_data['longest_focus_streak'],
            focus_latency           = summary_data['focus_latency'],
            state_switch_count      = summary_data['state_switch_count'],
            # Wave averages (filtered)
            beta_avg                = summary_data.get('beta_avg', 0),
            gamma_avg               = summary_data.get('gamma_avg', 0),
            alpha_avg               = summary_data.get('alpha_avg', 0),
            theta_avg               = summary_data.get('theta_avg', 0),
            # High-level inferences
            neural_state            = summary_data.get('neural_state', 'Unknown'),
            signal_integrity        = summary_data.get('signal_integrity', 'Unknown'),
            focus_depth             = summary_data.get('focus_depth', 'Unknown')
        )
        print(f"[DEBUG] SessionSummary created successfully with ID: {session_summary.id}")
        
        # Update user summary (if table exists)
        # try:
        #     from core_engine.recommendation.user_summary import update_summary
        #     update_summary(user_id=summary_data['user_email'])
        # except Exception as e:
        #     print(f"[UPDATE SUMMARY ERROR] {e}")
        
        # Update live status cache
        try:
            cache.set(f"live_status_{summary_data['user_email']}", "completed", timeout=300)
            
            # Clean up live brainwave cache
            cache.delete(f"live_eeg_stream_{summary_data['user_email']}")
            
        except Exception as e:
            pass

        # ── STEP 2: Generate recommendation using updated summary with inferences ──
        try:
            print(f"[RECOMMENDATION] Starting recommendation generation...")
            from core_engine.recommendation import generate_recommendation_for_session
            print(f"[RECOMMENDATION] Import successful")
            recommendation_text = generate_recommendation_for_session(
                user_email    = summary_data['user_email'],
                session_id    = summary_data['session_id'],
                final_summary = {
                    'average_focus_score'  : summary_data['average_focus_score'],
                    'concentrating_seconds': summary_data['concentrating_seconds'],
                    'neutral_seconds'      : summary_data['neutral_seconds'],
                    'relaxed_seconds'      : summary_data['relaxed_seconds'],
                    'beta_avg'             : summary_data.get('beta_avg', 0),
                    'gamma_avg'            : summary_data.get('gamma_avg', 0),
                    'alpha_avg'            : summary_data.get('alpha_avg', 0),
                    'theta_avg'            : summary_data.get('theta_avg', 0),
                    'neural_state'         : summary_data.get('neural_state', 'Unknown'),
                    'signal_integrity'     : summary_data.get('signal_integrity', 'Unknown'),
                    'focus_depth'          : summary_data.get('focus_depth', 'Unknown')
                }
            )
            print(f"[RECOMMENDATION] Generated successfully")
            print(f"[RECOMMENDATION] Text value: '{recommendation_text}'")
            print(f"[RECOMMENDATION] Type: {type(recommendation_text)}")

            subject = 'General'
            try:
                from secondBrain_App.models import PreSessionCheckIn
                print(f"[RECOMMENDATION DEBUG] Looking for PreSessionCheckIn with session_id: {summary_data['session_id']}")
                print(f"[RECOMMENDATION DEBUG] User: {summary_data['user_email']}")
                
                checkin = PreSessionCheckIn.objects.filter(
                    user=user_profile,
                    session_id=summary_data['session_id']
                ).order_by('-created_at').first()
                
                print(f"[RECOMMENDATION DEBUG] Found PreSessionCheckIn: {checkin is not None}")
                if checkin:
                    # Use custom value if subject is "Other"
                    if checkin.subject_task == 'Other' and checkin.subject_other_value:
                        subject = checkin.subject_other_value
                    else:
                        subject = checkin.subject_task or 'General'
                    print(f"[RECOMMENDATION DEBUG] Subject from PreSessionCheckIn: {subject}")
                else:
                    print(f"[RECOMMENDATION DEBUG] No PreSessionCheckIn found for session {summary_data['session_id']}")
            except Exception as e:
                print(f"[RECOMMENDATION] Could not fetch subject: {e}")

            # Parse sections from recommendation text
            from core_engine.recommendation import parse_recommendation_sections
            sections = parse_recommendation_sections(recommendation_text)

            from secondBrain_App.models import Recommendation
            import uuid
            Recommendation.objects.create(
                user=user_profile,
                session=session_summary,
                inference_id=str(uuid.uuid4()),
                recommendation_category='general',
                stimulus_name='study_tip',
                trigger_reason='session_end',
                message=recommendation_text,
                subject=subject,                                                        # ← new
                personalized_recommendation=sections['personalized_recommendation'],    # ← new
                recommended_study_methods=sections['recommended_study_methods'],        # ← new
                optimal_study_environment=sections['optimal_study_environment'],        # ← new
                what_to_avoid=sections['what_to_avoid']                                # ← new
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
            print(f"[RECOMMENDATION] Saved to cache key: {cache_key}")

            verify = cache.get(cache_key)
            print(f"[RECOMMENDATION] Cache verify read back: {verify}")


        except Exception as e:
            import traceback
            print(f"[RECOMMENDATION ERROR] {e}")
            print(traceback.format_exc())

        # ── STEP 3: Save recommendation to database (if recommendation engine is available) ──
        try:
            from core_engine.recommendation import save_recommendation
            
            # save_recommendation(
            #     user_email=summary_data['user_email'],
            #     session_id=summary_data['session_id'],
            #     inference_id=summary_data.get('task_id', 'unknown'),
            #     category='general',
            #     stimulus='study_tip',
            #     trigger='session_end',
            #     message=recommendation_text
            # )
        
        except Exception as e:
            import traceback
            print(f"[SAVE RECOMMENDATION ERROR] {e}")
            print(traceback.format_exc())

        return session_summary

    except UserProfile.DoesNotExist:
        print(f"User profile not found: {summary_data['user_email']}")
        return None
    except Exception as e:
        print(f"Error saving session summary: {e}")
        return None


def notch_filter(signal, freq=50, fs=256, Q=30):
    """Apply notch filter to remove power line noise"""
    try:
        b, a = iirnotch(freq/(fs/2), Q)
        return filtfilt(b, a, signal)
    except Exception as e:
        print(f"[FILTER ERROR] notch_filter failed: {e}")
        return signal


def calculate_inferences(beta_avg, gamma_avg, alpha_avg, theta_avg):
    """
    Convert raw wave power averages into high-level string inferences.
    
    Args:
        beta_avg: Average Beta wave power (Hz)
        gamma_avg: Average Gamma wave power (Hz)
        alpha_avg: Average Alpha wave power (Hz)
        theta_avg: Average Theta wave power (Hz)
    
    Returns:
        dict: {
            'neural_state': str,
            'signal_integrity': str,
            'focus_depth': str
        }
    """
    # Data Guard: Check for artifact-heavy signals
    max_wave = max(beta_avg, gamma_avg, alpha_avg, theta_avg)
    if max_wave > 500:
        return {
            'neural_state': 'Unknown',
            'signal_integrity': 'Poor',
            'focus_depth': 'Unknown'
        }
    
    # Determine Signal Integrity
    if max_wave > 100:
        signal_integrity = 'Artifact-Heavy'
    elif max_wave > 50:
        signal_integrity = 'Clean'
    else:
        signal_integrity = 'Clean'
    
    # Determine Neural State based on wave patterns
    # High Beta (15-30Hz) + High Gamma = Focus
    # High Alpha + High Theta = Drowsy
    # High Beta (>30Hz) = Anxious
    # High Theta = Distracted
    
    if beta_avg > 30:
        neural_state = 'Anxious'
    elif alpha_avg > 15 and theta_avg > 12:
        neural_state = 'Drowsy'
    elif theta_avg > 15:
        neural_state = 'Distracted'
    elif beta_avg >= 15 and beta_avg <= 30 and gamma_avg > 30:
        neural_state = 'Focus'
    else:
        neural_state = 'Neutral'
    
    # Determine Focus Depth
    if neural_state == 'Focus' and gamma_avg > 40:
        focus_depth = 'Deep Flow'
    elif neural_state == 'Focus':
        focus_depth = 'Light Focus'
    else:
        focus_depth = 'Surface Level'
    
    return {
        'neural_state': neural_state,
        'signal_integrity': signal_integrity,
        'focus_depth': focus_depth
    }


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