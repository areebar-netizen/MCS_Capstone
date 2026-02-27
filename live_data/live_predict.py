#!/usr/bin/env python3
"""
Live prediction pipeline:

- Modes:
  1. EEG (--eeg): Connects to Muse headset via LSL, denoises data, predicts, and shows live UI.
  2. CSV (--csv-dir): Processes static CSV files for testing/validation.

Usage:
    python code/live_predict.py --eeg --models models_out --model xgboost
    python code/live_predict.py --csv-dir dataset/test --models models_out
"""

from pathlib import Path
import sys
import time
import warnings
import os
import csv
import threading
from collections import deque
from typing import List, Tuple, Optional, Union 

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt, iirnotch

# LSL for live acquisition
try:
    from pylsl import resolve_byprop, StreamInlet
except ImportError:
    print("Warning: pylsl not installed. EEG mode will not work.")
    StreamInlet = None

# GUI
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / 'code'
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# Import custom modules
from EEG_feature_extraction_adv import generate_feature_vectors_from_matrix
from enhanced_feature_extraction import load_preprocessing_artifacts, apply_feature_pipeline
import joblib

# Attempt to import Stream.py if it exists
try:
    from Stream import Stream
except ImportError:
    Stream = None

# Constants
LABEL_MAP = {0: 'relaxed', 1: 'neutral', 2: 'concentrating'}
LABEL_COLORS = {
    'relaxed': (100, 180, 255),
    'neutral': (200, 200, 200),
    'concentrating': (120, 255, 120)
}

# -----------------------------------------------------------------------------
# Denoiser Class 
# -----------------------------------------------------------------------------
class Denoiser:
    """
    EEG denoiser: applies bandpass + notch filtering per channel.
    Operates on streaming rows (timestamp + channels).
    """

    def __init__(self, fs=256, lowcut=1, highcut=45, notch_freq=50, order=4, Q=30):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.order = order
        self.Q = Q

    def _bandpass_filter(self, signal):
        nyq = 0.5 * self.fs
        b, a = butter(
            self.order,
            [self.lowcut / nyq, self.highcut / nyq],
            btype="band"
        )
        return filtfilt(b, a, signal)

    def _notch_filter(self, signal):
        b, a = iirnotch(self.notch_freq / (self.fs / 2), self.Q)
        return filtfilt(b, a, signal)

    def process(self, rows: List[List[float]]) -> List[List[float]]:
        if not rows or len(rows) < 2:
            return rows

        arr = np.asarray(rows, dtype=float)
        timestamps = arr[:, 0]
        signals = arr[:, 1:]  # shape: (N, n_channels)

        filtered = np.zeros_like(signals)

        for ch in range(signals.shape[1]):
            sig = signals[:, ch]
            sig = self._bandpass_filter(sig)
            sig = self._notch_filter(sig)
            filtered[:, ch] = sig

        out = np.column_stack([timestamps, filtered])
        return out.tolist()

# -----------------------------------------------------------------------------
# Helper Classes
# -----------------------------------------------------------------------------

class EEGAcquirer:
    """Handles LSL connection and buffering."""
    def __init__(self, max_seconds: int = 30, sfreq: int = 256):
        self.inlet = None
        self.buffer = deque()
        self.max_len = max_seconds * sfreq
        self._running = False
        self._thread = None
        
        # For raw data saving
        self._raw_fh = None
        self._raw_writer = None

    def connect(self, timeout: float = 5.0):
        if StreamInlet is None:
            raise RuntimeError("pylsl not installed.")
        
        # Use Stream.py logic if available, otherwise direct resolve
        if Stream:
            s = Stream()
            self.inlet = s.connect_to_eeg_stream()
        else:
            stream = resolve_byprop('type', 'EEG', timeout=timeout)
            if not stream:
                raise RuntimeError('No EEG stream found')
            self.inlet = StreamInlet(stream[0])
        return True

    def start_saving_raw(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        need_header = not os.path.exists(path) or os.path.getsize(path) == 0
        
        fh = open(path, mode='a', newline='')
        self._raw_writer = csv.writer(fh)
        if need_header:
            # Default Muse header
            header = ['timestamps', 'TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
            self._raw_writer.writerow(header)
        
        self._raw_fh = fh
        print(f"Logging raw data to {path}")

    def start(self):
        if self.inlet is None:
            raise RuntimeError('Not connected')
        if self._running:
            return
        self._running = True

        def run():
            while self._running:
                try:
                    sample, timestamp = self.inlet.pull_sample(timeout=1.0)
                except Exception:
                    sample, timestamp = None, None
                
                if sample is None:
                    continue
                
                row = [float(timestamp)] + [float(x) for x in sample]
                self.buffer.append(row)
                
                if len(self.buffer) > self.max_len:
                    self.buffer.popleft()
                
                # Save raw if enabled
                if self._raw_writer:
                    try:
                        self._raw_writer.writerow(row)
                        self._raw_fh.flush()
                    except Exception:
                        pass

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.1)
        if self._raw_fh:
            self._raw_fh.close()

    def get_buffer_copy(self):
        return list(self.buffer)


def load_model(models_dir: Path, model_name: str):
    """Loads model, feature selector, and enhanced preprocessing artifacts."""
    mapping = {
        'random_forest': 'random_forest.joblib',
        'xgboost': 'xgboost.joblib',
        'stacked_model': 'stacked_model.joblib'
    }
    
    model_file = Path(models_dir) / mapping.get(model_name, f"{model_name}.joblib")
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")
    
    model = joblib.load(model_file)
    
    # Load standard feature selector (used if enhanced artifacts missing)
    selector_path = Path(models_dir) / 'feature_selector.joblib'
    selector = joblib.load(selector_path) if selector_path.exists() else None
    
    # Load enhanced artifacts
    try:
        scaler, feature_info = load_preprocessing_artifacts('preprocessing_artifacts')
    except Exception:
        scaler, feature_info = None, None
    
    return model, selector, (scaler, feature_info)


class Predictor:
    def __init__(self, models_dir: Union[str, Path], model_name: str = 'stacked_model'):
        self.model, self.feature_selector, self.preprocessing_artifacts = load_model(models_dir, model_name)
        self.last_prediction = None
        self.last_confidence = 0.0

    def predict_from_rows(self, rows: List[List[float]], nsamples: int = 150, period: float = 1.0, cols_to_ignore: int = -1) -> Tuple[np.ndarray, int, float]:
        """Process raw rows, extract features, and predict."""
        if not rows or len(rows) < 2:
            return None, 0, 0.0

        arr = np.array(rows, dtype=float)
        if arr.size == 0:
            return None, 0, 0.0

        # 1. Feature Extraction
        try:
            vectors, _ = generate_feature_vectors_from_matrix(
                arr, 
                nsamples=nsamples, 
                period=period,
                state=None,
                remove_redundant=True,
                cols_to_ignore=cols_to_ignore
            )
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None, 0, 0.0

        if vectors is None or len(vectors) == 0:
            return None, 0, 0.0

        X = np.asarray(vectors, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 2. Apply Enhanced Preprocessing (Fix for the accuracy issue)
        scaler, feature_info = self.preprocessing_artifacts
        if scaler is not None and feature_info is not None:
            try:
                # This uses the corrected apply_feature_pipeline with indices
                X = apply_feature_pipeline(X, scaler, feature_info)
            except Exception as e:
                print(f"Warning: Enhanced pipeline failed: {e}")
        
        # 3. Apply Standard Feature Selector (if no enhanced artifacts)
        elif self.feature_selector is not None:
            try:
                X = self.feature_selector.transform(X)
            except Exception as e:
                print(f"Feature selection error: {e}")

        # 4. Predict
        try:
            preds = self.model.predict(X)
            confidence = 0.0
            if hasattr(self.model, "predict_proba"):
                try:
                    probas = self.model.predict_proba(X)
                    confidence = np.max(probas, axis=1).mean()
                except Exception:
                    confidence = 0.5
            else:
                confidence = 0.5
            return preds, X.shape[0], confidence
        except Exception as e:
            print(f"Prediction failed: {e}")
            return None, 0, 0.0


class Visualizer(QtWidgets.QWidget):
    """Live UI for predictions."""
    def __init__(self, preds=None, step_seconds: float = 0.5):
        super().__init__()
        self.preds = np.asarray(preds, dtype=int) if preds is not None else np.array([], dtype=int)
        self.n = len(self.preds)
        self.step = step_seconds
        self.idx = 0
        self.prediction_count = 0
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.init_ui()
        self.update_display()

    def init_ui(self):
        self.setWindowTitle('Live Prediction Visualizer')
        layout = QtWidgets.QVBoxLayout()

        # Main Label
        self.label = QtWidgets.QLabel('', alignment=QtCore.Qt.AlignCenter)
        font = self.label.font()
        font.setPointSize(28)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setFixedHeight(80)
        layout.addWidget(self.label)

        # Warning Label
        self.warning_label = QtWidgets.QLabel('', alignment=QtCore.Qt.AlignCenter)
        self.warning_label.setStyleSheet('color: red;')
        self.warning_label.setFixedHeight(40)
        layout.addWidget(self.warning_label)

        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setFixedHeight(120)
        self.plot_widget.hideAxis('left')
        self.plot_widget.showGrid(x=True, y=False)
        self.image = pg.ImageItem()
        self.plot_widget.addItem(self.image)
        self.vline = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=2))
        self.plot_widget.addItem(self.vline)
        self.plot_widget.setYRange(0, 1)
        layout.addWidget(self.plot_widget)

        # Controls
        btn_layout = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton('Play')
        self.play_btn.clicked.connect(self.toggle_play)
        btn_layout.addWidget(self.play_btn)
        
        self.step_fwd_btn = QtWidgets.QPushButton('Next')
        self.step_fwd_btn.clicked.connect(self.next_frame)
        btn_layout.addWidget(self.step_fwd_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.plot_widget.setXRange(-1, max(10, self.n))

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText('Play')
        else:
            self.timer.start(int(self.step * 1000))
            self.play_btn.setText('Pause')

    def next_frame(self):
        if self.idx < self.n - 1:
            self.idx += 1
            self.update_display()
        else:
            self.timer.stop()
            self.play_btn.setText('Play')

    def update_display(self):
        if self.n == 0:
            self.label.setText('Waiting for data...')
            return
            
        p = int(self.preds[self.idx])
        name = LABEL_MAP.get(p, str(p))
        color = LABEL_COLORS.get(name, (220, 220, 220))
        r, g, b = color
        
        self.label.setText(f"{self.prediction_count}: {name}")
        self.label.setStyleSheet(f'background-color: rgb({r},{g},{b});')
        self.vline.setPos(self.idx)
        self.plot_widget.setXRange(max(0, self.idx - 10), self.idx + 5)

    def set_preds(self, preds, warning_msg=None):
        if preds is not None:
            # Append new predictions to existing ones for continuous timeline
            new_preds = np.asarray(preds, dtype=int)
            if len(self.preds) == 0:
                self.preds = new_preds
            else:
                self.preds = np.concatenate([self.preds, new_preds])
            self.n = len(self.preds)
            self.prediction_count += 1
        else:
            self.n = len(self.preds)
        
        self.idx = max(0, self.n - 1) # Jump to latest
        
        if warning_msg:
            self.warning_label.setText(warning_msg)
        else:
            self.warning_label.clear()
            
        # Build image bar
        if self.n == 0:
            img = np.zeros((1, 1, 3), dtype=np.uint8)
        else:
            img = np.zeros((1, self.n, 3), dtype=np.uint8)
            for i, p in enumerate(self.preds):
                name = LABEL_MAP.get(int(p), str(p))
                color = LABEL_COLORS.get(name, (220, 220, 220))
                img[0, i, :] = np.array(color, dtype=np.uint8)
        
        self.image.setImage(img)
        self.plot_widget.setXRange(-1, max(10, self.n))
        
        # Update display last to ensure text color matches timeline
        self.update_display()


# -----------------------------------------------------------------------------
# Main Execution Logic
# -----------------------------------------------------------------------------

def process_csv_mode(args):
    """Process static CSV files."""
    print(f"Processing CSV files in: {args.csv_dir}")
    
    den = Denoiser()
    pred = Predictor(args.models, args.model)
    
    csv_dir = Path(args.csv_dir)
    csv_files = sorted(csv_dir.glob('*.csv'))
    
    if not csv_files:
        print("No CSV files found.")
        return

    print(f"Found {len(csv_files)} files.")
    
    # Initialize summary rows for CSV output
    summary_rows = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            rows = df.values.tolist()
            
            # Denoise
            rows = den.process(rows)
            
            # Predict
            preds, n_windows, confidence = pred.predict_from_rows(rows, nsamples=args.nsamples, period=args.period)
            
            if preds is not None:
                counts = {lbl: 0 for lbl in LABEL_MAP.values()}
                for p in preds:
                    name = LABEL_MAP.get(int(p), str(p))
                    counts[name] = counts.get(name, 0) + 1
                
                # Calculate durations (0.5s per window)
                durations = {k: v * 0.5 for k, v in counts.items()}
                total_seconds = n_windows * 0.5
                
                # Get predicted label (most common)
                predicted_label = max(counts, key=counts.get)
                
                print(f"{csv_file.name}: {n_windows} windows")
                for lbl, count in counts.items():
                    print(f"  {lbl}: {count}")
                
                # Create result dict like predict_test.py
                result = {
                    'filename': csv_file.name,
                    'n_windows': n_windows,
                    'total_seconds': total_seconds,
                    'relaxed_seconds': durations.get('relaxed', 0),
                    'neutral_seconds': durations.get('neutral', 0),
                    'concentrating_seconds': durations.get('concentrating', 0),
                    'predicted_label': predicted_label,
                    'confidence': confidence
                }
                print(f"  Full result: {result}")
                
                # Add to summary rows for CSV output
                summary_rows.append([
                    csv_file.name,
                    n_windows,
                    total_seconds,
                    durations.get('relaxed', 0),
                    durations.get('neutral', 0),
                    durations.get('concentrating', 0),
                    predicted_label,
                    confidence
                ])
            else:
                print(f"{csv_file.name}: No valid windows.")
                
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
    
    # Write summary CSV if requested
    if args.summary_out and summary_rows:
        try:
            with open(args.summary_out, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'n_windows', 'total_seconds', 
                               'relaxed_seconds', 'neutral_seconds', 'concentrating_seconds',
                               'predicted_label', 'confidence'])
                writer.writerows(summary_rows)
            print(f"Summary written to {args.summary_out}")
        except Exception as e:
            print(f"Error writing summary file: {e}")


def run_eeg_mode(args):
    """Run live EEG acquisition and prediction."""
    print("Starting EEG Mode...")
    
    app = QtWidgets.QApplication([])
    viz = Visualizer()
    viz.resize(900, 300)
    viz.show()

    den = Denoiser()
    pred = Predictor(args.models, args.model)
    
    # Connect to LSL
    acq = EEGAcquirer()
    
    # Setup duration timer if specified
    duration_timer = None
    if args.duration is not None:
        duration_seconds = args.duration * 60  # Convert minutes to seconds
        print(f"Recording will stop after {args.duration} minutes ({duration_seconds} seconds)")
        
        def stop_recording():
            print(f"\nRecording duration of {args.duration} minutes reached. Stopping...")
            viz.set_preds([], "Recording completed")
            app.quit()
        
        duration_timer = QtCore.QTimer()
        duration_timer.timeout.connect(stop_recording)
        duration_timer.start(int(duration_seconds * 1000))  # Convert to milliseconds
    
    try:
        print("Connecting to EEG stream...")
        acq.connect()
        acq.start()
        
        if args.raw_out:
            acq.start_saving_raw(args.raw_out)
            
        print("Connected. Streaming data...")
        
    except Exception as e:
        viz.set_preds([], f"Connection Failed: {e}")
        app.exec_()
        return

    # Polling function
    def poll_and_predict():
        rows = acq.get_buffer_copy()
        
        # Minimum samples needed (approx 1.5s)
        min_samples = int(256 * args.period) 
        if len(rows) < min_samples:
            return

        # Denoise
        rows = den.process(rows)
        
        # Predict
        preds, n_windows, confidence = pred.predict_from_rows(rows, nsamples=args.nsamples, period=args.period)
        
        if preds is not None:
            viz.set_preds(preds)

    timer = QtCore.QTimer()
    timer.timeout.connect(poll_and_predict)
    timer.start(500)  # Update twice per second

    app.exec_()
    acq.stop()


def main():
    import argparse
    p = argparse.ArgumentParser(description='EEG Prediction Pipeline')
    
    # Mode Selection
    mode_group = p.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--eeg', action='store_true', help='Use live EEG input')
    mode_group.add_argument('--csv-dir', type=str, help='Process CSV files in directory')
    
    # Common arguments
    p.add_argument('--models', default='models_out', help='Directory containing models')
    p.add_argument('--model', default='xgboost', choices=['random_forest', 'xgboost', 'stacked_model'])
    p.add_argument('--period', type=float, default=1.0, help='Window size in seconds')
    p.add_argument('--nsamples', type=int, default=150, help='Samples per window')
    
    # EEG specific
    p.add_argument('--raw-out', type=str, default=None, help='Path to save raw EEG CSV')
    p.add_argument('--summary-out', type=str, default=None, help='Path to save prediction summary CSV')
    p.add_argument('--duration', type=float, default=None, help='Recording duration in minutes (if not specified, runs indefinitely)')
    
    args = p.parse_args()
    
    if args.eeg:
        run_eeg_mode(args)
    elif args.csv_dir:
        process_csv_mode(args)

if __name__ == '__main__':
    main()