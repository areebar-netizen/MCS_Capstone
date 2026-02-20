#!/usr/bin/env python3
"""
Live prediction pipeline:

- EEGAcquirer: connects to an LSL EEG stream and buffers samples
- Denoiser: placeholder class (user will implement later)
- Predictor: loads a saved model and uses the existing feature extractor to
  produce per-window predictions from the buffered samples

Run from the repository root so relative imports to the `code/` folder work.
"""
from pathlib import Path
import sys
import time
import warnings
import webbrowser
import os
import csv
import threading
from collections import deque
from typing import List, Tuple, Optional, Union 

import numpy as np

# LSL for live acquisition
from pylsl import resolve_byprop, StreamInlet
from scipy.signal import butter, filtfilt, iirnotch

# GUI
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / 'code'
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from EEG_feature_extraction_adv import generate_feature_vectors_from_samples, generate_feature_vectors_from_matrix
from enhanced_feature_extraction import load_preprocessing_artifacts, apply_feature_pipeline
import joblib


LABEL_MAP = {
    0: 'relaxed',
    1: 'neutral',
    2: 'concentrating'
}

LABEL_COLORS = {
    'relaxed': (100, 180, 255),
    'neutral': (200, 200, 200),
    'concentrating': (120, 255, 120)
}


class EEGAcquirer:
    """Connect to an LSL EEG stream and keep a rolling buffer of samples.

    Each sample is stored as a list: [timestamp, ch1, ch2, ...]
    """
    def __init__(self, max_seconds: int = 30, sfreq: int = 256):
        self.inlet: Optional[StreamInlet] = None
        self.channels = []
        self.units = []
        self.buffer = deque()
        self.max_len = max_seconds * sfreq
        self._thread = None
        self._running = False
        self._raw_file = None
        self._raw_writer = None
        self._raw_fh = None
        self._raw_writer = None

    #  searches for an eeg stream
    def connect(self, timeout: float = 5.0):
        stream = resolve_byprop('type', 'EEG', timeout=timeout)
        if not stream:
            raise RuntimeError('No EEG stream found')
        self.inlet = StreamInlet(stream[0])
        self.channels, self.units = self.get_meta()
        return self.inlet

    def start_saving_raw(self, path: str):
        """ Opens a CSV file in append mode, writes header if file is new, then starts a background thread (_save_raw_loop) that writes one sample per second (most recent from buffer) to the file. This is a second raw logging mechanism distinct from enable_raw_logging.
            In practice, enable_raw_logging is unused; start_saving_raw is the working one.
        """
        if self.inlet is None:
            raise RuntimeError('Not connected')
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # determine whether to write header
        need_header = True
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                need_header = False
        except Exception as e:
            print(f"Warning: Could not check existing file: {e}")
            
        try:
            fh = open(path, mode='a', newline='')
            writer = csv.writer(fh)
            
            if need_header:
                header = ['timestamps', 'TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
                writer.writerow(header)
                fh.flush()
                
            self._raw_fh = fh
            self._raw_writer = writer
            print(f"Started saving raw data to: {os.path.abspath(path)}")
            
            # Start a thread to continuously save raw data
            self._raw_running = True
            self._raw_thread = threading.Thread(target=self._save_raw_loop, daemon=True)
            self._raw_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting raw data logging: {e}")
            if 'fh' in locals():
                fh.close()
            return False
            
    def _save_raw_loop(self):
        """Background thread to save raw data continuously."""
        last_save_time = time.time()
        while getattr(self, '_raw_running', False):
            try:
                current_time = time.time()
                if current_time - last_save_time >= 1.0:  # Save every second
                    if self.buffer and self._raw_writer is not None and self._raw_fh is not None:
                        # Get the most recent sample
                        sample = self.buffer[-1]
                        # Write timestamp and channel data
                        self._raw_writer.writerow(sample)
                        self._raw_fh.flush()
                        last_save_time = current_time
                time.sleep(0.1)  # Small sleep to prevent busy waiting
            except Exception as e:
                print(f"Error in raw data saving thread: {e}")
                time.sleep(1.0)
                
    def stop_saving_raw(self):
        """Stop the raw data saving thread."""
        self._raw_running = False
        if hasattr(self, '_raw_thread'):
            self._raw_thread.join(timeout=1.0)
        if self._raw_fh is not None:
            try:
                self._raw_fh.close()
            except Exception:
                pass

    def get_meta(self) -> Tuple[List[str], List[str]]:
        """ Walks through the stream’s channels structure to extract labels and units for every channel."""
        info = self.inlet.info()
        channels = []
        units = []
        ch = info.desc().child('channels').child('channel')
        for i in range(info.channel_count()):
            channels.append(ch.child_value('label'))
            units.append(ch.child_value('unit'))
            ch = ch.next_sibling()
        return channels, units


    def start(self):
        """ Starts a background thread that continuously calls pull_sample(timeout=1.0).
            Each sample (timestamp + channel voltages) is appended to the deque; if length exceeds max_len, oldest is popped.
            If raw logging is enabled, the sample is also written to CSV and flushed. 
        """

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
                # write raw sample if enabled (single write)
                if self._raw_writer is not None and self._raw_fh is not None:
                    try:
                        self._raw_writer.writerow(row)
                        try:
                            self._raw_fh.flush()
                        except Exception:
                            pass
                    except Exception:
                        # ignore I/O errors to avoid stopping acquisition
                        pass

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        """ Stops the background thread and closes the raw file if open. """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.1)
        if self._raw_fh is not None:
            try:
                self._raw_fh.close()
            except Exception:
                pass

    def get_buffer_copy(self) -> List[List[float]]:
        return list(self.buffer)

    def enable_raw_logging(self, path: str):
        """Enable appending raw LSL rows to CSV file at `path`.

        The header row will be written after connection when channel labels are
        available (call this after `connect()` returns)."""
        try:
            f = open(path, mode='a', newline='')
            writer = csv.writer(f)
            # If file is empty, write header if channels are known
            try:
                f.seek(0, os.SEEK_END)
                if f.tell() == 0 and self.channels:
                    header = ['timestamps'] + [str(c) for c in self.channels]
                    writer.writerow(header)
                    f.flush()
            except Exception:
                pass
            # keep names consistent with start_saving_raw
            self._raw_fh = f
            self._raw_writer = writer
        except Exception as e:
            print('Failed to enable raw logging:', e)


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
        """ Designs a Butterworth bandpass (1–45 Hz) using butter and applies filtfilt (zero‑phase) to the entire signal array.
            Note: filtfilt requires the full signal, which is acceptable for window‑wise processing.
        """
        nyq = 0.5 * self.fs
        b, a = butter(
            self.order,
            [self.lowcut / nyq, self.highcut / nyq],
            btype="band"
        )
        return filtfilt(b, a, signal)

    def _notch_filter(self, signal):
        """ Designs a 50 Hz notch filter using iirnotch and applies filtfilt. """
        b, a = iirnotch(self.notch_freq / (self.fs / 2), self.Q)
        return filtfilt(b, a, signal)

    def process(self, rows: List[List[float]]) -> List[List[float]]:
        """
        rows format:
        [
          [timestamp, ch1, ch2, ch3, ch4],
          ...
        ]
        """
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



def load_model(models_dir: Path, model_name: str):
    """ Loads a trained model from a joblib file. """
    mapping = {
        'random_forest': 'random_forest.joblib',
        'xgboost': 'xgboost.joblib',
        'stacked_model': 'stacked_model.joblib'
    }
    if model_name not in mapping:
        raise ValueError(f'unknown model {model_name}')
    model_file = Path(models_dir) / mapping[model_name]
    if not model_file.exists():
        raise FileNotFoundError(f'Model file not found: {model_file}')
    model = joblib.load(model_file)
    
    # Load feature selector
    feature_selector_file = Path(models_dir) / 'feature_selector.joblib'
    feature_selector = None
    if feature_selector_file.exists():
        feature_selector = joblib.load(feature_selector_file)
    
    # Load preprocessing artifacts for enhanced features
    try:
        scaler, feature_info = load_preprocessing_artifacts('preprocessing_artifacts')
        has_enhanced_preprocessing = True
    except Exception as e:
        print(f"Warning: Could not load preprocessing artifacts: {e}")
        scaler = None
        feature_info = None
        has_enhanced_preprocessing = False
    
    return model, feature_selector, (scaler, feature_info, has_enhanced_preprocessing)


class Predictor:
    def __init__(self, models_dir: Union[str, Path], model_name: str = 'stacked_model'):
        """Initialize predictor with a trained model."""
        self.model, self.feature_selector, self.preprocessing_artifacts = load_model(models_dir, model_name)
        self.last_prediction = None
        self.last_confidence = 0.0
        self.prediction_history = []
        self.confidence_history = []
        self.last_update_time = time.time()

    def predict_from_rows(self, rows: List[List[float]], channels: List[str], units: List[str],
                         nsamples: int = 150, period: float = 1.0, cols_to_ignore: int = -1) -> Tuple[np.ndarray, int]:
        """Generate predictions from a list of sample rows.
        
        Args:
            rows: List of sample rows, where each row is a list of floats
            channels: List of channel names
            units: List of units for each channel
            nsamples: Number of samples per window
            period: Window size in seconds
            cols_to_ignore: Number of columns to ignore from end of each row
        
        Returns:
            Tuple of (predictions, number_of_windows)
        """
        if not rows or len(rows) < 2:
            return None, 0

        # Convert rows to numpy array
        arr = np.array(rows, dtype=float)
        
        # If no samples after trimming, return early
        if arr.size == 0:
            return None, 0

        # Apply feature extraction
        try:
            # Try in-memory feature extraction first
            vectors, _ = generate_feature_vectors_from_matrix(
                arr, 
                nsamples=nsamples, 
                period=period,
                state=None,
                remove_redundant=True,
                cols_to_ignore=cols_to_ignore
            )
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return None, 0

        if vectors is None or len(vectors) == 0:
            return None, 0

        X = np.asarray(vectors, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Apply enhanced preprocessing if available
        if (self.preprocessing_artifacts is not None and 
            len(self.preprocessing_artifacts) >= 3 and 
            self.preprocessing_artifacts[2]):  # has_enhanced_preprocessing
            
            try:
                scaler, feature_info = self.preprocessing_artifacts[0], self.preprocessing_artifacts[1]
                X_processed = apply_feature_pipeline(X, scaler, feature_info)
                
                if X_processed is not None:
                    X = X_processed
                else:
                    print("Warning: Enhanced preprocessing failed, using raw features")
            except Exception as e:
                print(f"Warning: Enhanced preprocessing failed: {e}")
        
        # Apply feature selection if available
        if self.feature_selector is not None:
            try:
                X = self.feature_selector.transform(X)
            except Exception as e:
                print(f"Warning: Feature selection failed: {e}")

        # Get predictions
        try:
            preds = self.model.predict(X)
            return preds, X.shape[0]
        except Exception as e:
            print(f"Prediction failed: {e}")
            return None, 0

    def predict(self, sample_matrix: np.ndarray) -> Tuple[int, float]:
        """Generate a prediction from a pre-processed feature matrix."""
        if len(sample_matrix) == 0:
            return self.last_prediction, self.last_confidence
            
        # Apply feature selection if available
        if self.feature_selector is not None:
            try:
                sample_matrix = self.feature_selector.transform(sample_matrix)
            except Exception as e:
                print(f"Warning: Feature selection failed: {e}")

        # Get predictions
        predictions = self.model.predict(sample_matrix)
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(sample_matrix)
            confidence = np.max(probas, axis=1).mean()
        else:
            # For models without predict_proba, use the most common prediction
            confidence = np.mean(predictions == stats.mode(predictions, keepdims=True)[0])
            
        predicted_class = int(stats.mode(predictions, keepdims=True)[0][0])
        
        # Update state
        current_time = time.time()
        self.last_update_time = current_time
        
        self.prediction_history.append(predicted_class)
        self.confidence_history.append(confidence)
        
        # Keep history of last 10 predictions
        if len(self.prediction_history) > 10:
            self.prediction_history.pop(0)
            self.confidence_history.pop(0)
            
        # Only update if we have enough history or confidence is high
        if len(self.prediction_history) >= 3:
            # Use the most common prediction in the history
            predicted_class = int(stats.mode(self.prediction_history, keepdims=True)[0][0])
            confidence = np.mean(self.confidence_history)
            
        self.last_prediction = predicted_class
        self.last_confidence = confidence
        
        return predicted_class, confidence


class Visualizer(QtWidgets.QWidget):
    """Adapted visualizer that can be updated with new predictions live."""
    def __init__(self, preds=None, step_seconds: float = 0.5):
        """ Stores predictions array, step size, index. 
            Sets up the UI: a main QLabel, a warning label, a PlotWidget with an ImageItem for the coloured bar, a red vertical line, and play/pause/step buttons.
        """
        super().__init__()
        self.preds = np.asarray(preds, dtype=int) if preds is not None else np.array([], dtype=int)
        self.n = len(self.preds)
        self.step = step_seconds
        self.idx = 0
        self.prediction_count = 0  # Add a counter for total predictions
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.init_ui()
        self.update_display()

    def init_ui(self):
        """ Builds ui layout """
        self.setWindowTitle('Live Prediction Visualizer')
        layout = QtWidgets.QVBoxLayout()

        # Main prediction label
        self.label = QtWidgets.QLabel('', alignment=QtCore.Qt.AlignCenter)
        font = self.label.font()
        font.setPointSize(28)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setFixedHeight(80)
        layout.addWidget(self.label)
        
        # Warning label for device connection issues
        self.warning_label = QtWidgets.QLabel('', alignment=QtCore.Qt.AlignCenter)
        warning_font = self.warning_label.font()
        warning_font.setPointSize(16)
        warning_font.setBold(True)
        self.warning_label.setFont(warning_font)
        self.warning_label.setStyleSheet('color: red;')
        self.warning_label.setFixedHeight(40)
        layout.addWidget(self.warning_label)

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

        btn_layout = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton('Play')
        self.play_btn.clicked.connect(self.toggle_play)
        btn_layout.addWidget(self.play_btn)
        self.step_back_btn = QtWidgets.QPushButton('Step <-')
        self.step_back_btn.clicked.connect(self.step_back)
        btn_layout.addWidget(self.step_back_btn)
        self.step_fwd_btn = QtWidgets.QPushButton('-> Step')
        self.step_fwd_btn.clicked.connect(self.next_frame)
        btn_layout.addWidget(self.step_fwd_btn)
        btn_layout.addStretch()
        self.step_label = QtWidgets.QLabel('step (s):')
        btn_layout.addWidget(self.step_label)
        self.step_spin = QtWidgets.QDoubleSpinBox()
        self.step_spin.setSingleStep(0.1)
        self.step_spin.setDecimals(2)
        self.step_spin.setValue(self.step)
        self.step_spin.valueChanged.connect(self.change_step)
        btn_layout.addWidget(self.step_spin)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.plot_widget.setXRange(-1, max(10, self.n))

    def change_step(self, val):
        """ Updates step size and timer interval """
        self.step = float(val)
        if self.timer.isActive():
            self.timer.setInterval(int(self.step * 1000))

    def toggle_play(self):
        """ Toggles play/pause timer """
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText('Play')
        else:
            self.timer.start(int(self.step * 1000))
            self.play_btn.setText('Pause')

    def step_back(self):
        """ move current index """
        self.idx = max(0, self.idx - 1)
        self.update_display()

    def next_frame(self):
        if self.idx < self.n - 1:
            self.idx += 1
            self.update_display()
        else:
            self.timer.stop()
            self.play_btn.setText('Play')

    def update_display(self):
        """ Updates display with current prediction """
        if self.n == 0:
            self.label.setText('No windows')
            return
        p = int(self.preds[self.idx])
        name = LABEL_MAP.get(p, str(p))
        color = LABEL_COLORS.get(name, (220, 220, 220))
        r, g, b = color
        # Show prediction count and current state
        self.label.setText(f"{self.prediction_count}: {name}")
        self.label.setStyleSheet(f'background-color: rgb({r},{g},{b});')
        self.vline.setPos(self.idx)
        self.plot_widget.setXRange(max(0, self.idx - 10), self.idx + 5)

    def set_preds(self, preds, warning_msg=None):
        """Update predictions and increment the prediction counter."""
        self.preds = np.asarray(preds, dtype=int)
        self.n = len(self.preds)
        # Update the display with the current prediction count
        if hasattr(self, 'prediction_count'):
            self.prediction_count += len(preds)
        else:
            self.prediction_count = len(preds)
        
        # Only increment counter if we have new predictions
        if self.n > 0:
            self.prediction_count += 1
            
        self.idx = min(self.idx, self.n - 1) if self.n > 0 else 0
        
        # Update warning label if message is provided
        if warning_msg:
            self.warning_label.setText(warning_msg)
        else:
            self.warning_label.clear()
            
        self.update_display()

        # build image: 1 x n x 3
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
        self.idx = min(self.idx, max(0, self.n - 1))
        self.update_display()


def process_csv_file(pred, den, csv_path, nsamples=150, period=1.0, cols_to_ignore=-1, step_seconds=0.5):
    """Process a single CSV file and return predictions."""
    import pandas as pd
    from pathlib import Path
    
    print(f"Processing {Path(csv_path).name}...")
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None, 0
    
    # Convert to list of rows format expected by the predictor
    rows = df.values.tolist()
    
    # Process the data through denoiser
    denoised_rows = den.process(rows)
    
    # Get predictions
    preds, n_windows = pred.predict_from_rows(
        denoised_rows, 
        channels=df.columns[1:].tolist(),  # Skip timestamp column
        units=['uV'] * (len(df.columns) - 1),  # Assume all channels are in uV
        nsamples=nsamples,
        period=period,
        cols_to_ignore=cols_to_ignore
    )
    
    if preds is None or n_windows == 0:
        print(f"  No predictions generated for {Path(csv_path).name}")
        return None, 0
    
    # Summarize predictions
    counts = {lbl: 0 for lbl in LABEL_MAP.values()}
    for p in preds:
        name = LABEL_MAP.get(int(p), str(p))
        counts[name] = counts.get(name, 0) + 1
    
    durations = {k: v * step_seconds for k, v in counts.items()}
    total_seconds = n_windows * step_seconds
    
    print(f"  windows: {n_windows}, total seconds (approx): {total_seconds:.2f}")
    for lbl in LABEL_MAP.values():
        print(f"  {lbl}: {durations.get(lbl, 0):.2f}s", end="")
    print()
    
    return preds, n_windows

def main():
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description='EEG prediction pipeline with support for live EEG and CSV processing')
    
    # Mode selection
    mode_group = p.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--eeg', action='store_true', help='Use live EEG input')
    mode_group.add_argument('--csv-dir', type=str, help='Process all CSV files in the specified directory')
    
    # Common arguments
    p.add_argument('--models', default='models_out', help='Directory containing model files')
    p.add_argument('--model', default='stacked_model', choices=['random_forest', 'xgboost', 'stacked_model'],
                  help='Which model to use for predictions')
    p.add_argument('--period', type=float, default=1.0,
                  help='Window size in seconds for feature extraction')
    p.add_argument('--nsamples', type=int, default=150,
                  help='Number of samples per window')
    p.add_argument('--cols-to-ignore', type=int, default=-1,
                  help='Number of columns to ignore from the end of input')
    p.add_argument('--summary-out', type=str, default=None,
                  help='Path to write prediction summary CSV')
    
    # EEG-specific arguments
    eeg_group = p.add_argument_group('EEG mode options')
    eeg_group.add_argument('--min-buffer-sec', type=float, default=1.5,
                         help='Minimum seconds of buffered data before prediction (EEG mode only)')
    eeg_group.add_argument('--mock', action='store_true',
                          help='Run in mock mode with test data (EEG mode only)')
    eeg_group.add_argument('--replay', type=str, default=None,
                          help='Path to CSV to replay in mock mode (EEG mode only)')
    eeg_group.add_argument('--raw-out', type=str, default=None,
                          help='Path to write raw EEG samples as CSV (EEG mode only)')
    eeg_group.add_argument('--auto-stream', action='store_true',
                          help='Automatically launch muselsl streamer (EEG mode only)')
    eeg_group.add_argument('--muselsl-args', type=str, default='',
                          help='Additional args to pass to muselsl stream (quoted, EEG mode only)')
    
    args = p.parse_args()

    # Initialize denoiser and predictor
    den = Denoiser()
    pred = Predictor(models_dir=args.models, model_name=args.model)
    
    # Process in CSV mode if requested
    if args.csv_dir:
        csv_dir = Path(args.csv_dir)
        if not csv_dir.is_dir():
            print(f"Error: {csv_dir} is not a valid directory")
            return 1
            
        # Process all CSV files in the directory
        csv_files = list(csv_dir.glob('*.csv'))
        if not csv_files:
            print(f"No CSV files found in {csv_dir}")
            return 0
            
        summary_rows = []
        
        for csv_file in sorted(csv_files):
            preds, n_windows = process_csv_file(
                pred, den, csv_file,
                nsamples=args.nsamples,
                period=args.period,
                cols_to_ignore=args.cols_to_ignore
            )
            
            if preds is not None and n_windows > 0:
                # Calculate durations for summary
                counts = {lbl: 0 for lbl in LABEL_MAP.values()}
                for p in preds:
                    name = LABEL_MAP.get(int(p), str(p))
                    counts[name] = counts.get(name, 0) + 1
                
                durations = {k: v * 0.5 for k, v in counts.items()}  # 0.5s per prediction
                summary_rows.append([
                    csv_file.name,  # filename
                    n_windows,  # n_windows
                    n_windows * 0.5,  # total_seconds
                    durations.get('relaxed', 0),
                    durations.get('neutral', 0),
                    durations.get('concentrating', 0)
                ])
        
        # Write summary if requested
        if args.summary_out and summary_rows:
            with open(args.summary_out, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'n_windows', 'total_seconds', 
                               'relaxed_seconds', 'neutral_seconds', 'concentrating_seconds'])
                writer.writerows(summary_rows)
            print(f"Summary written to {args.summary_out}")
            
        return 0
    
    # Otherwise, continue with EEG processing
    # ... (rest of the original EEG processing code)

    app = QtWidgets.QApplication([])
    viz = Visualizer()
    viz.resize(900, 300)
    viz.show()

    if args.mock or args.replay is not None:
        # load a CSV from disk and use it as the buffer
        replay_file = getattr(args, 'replay', None)
        if replay_file is None:
            # default test file (10sec has enough data for multiple windows)
            replay_file = str(ROOT / 'dataset' / 'test' / '10sec.csv')
        # load numeric CSV into memory using numpy (skip header row)
        try:
            data = np.genfromtxt(replay_file, delimiter=',', skip_header=1)
        except Exception as e:
            print('Failed to load replay file:', e)
            return

        if data.size == 0:
            print('Replay file contains no numeric data:', replay_file)
            return

        # Use the entire replay file immediately (matches code/predict_test.py behavior)
        rows = data.tolist()
        rows = den.process(rows)
        preds, _ = pred.predict_from_rows(rows, ['CH'] * (data.shape[1] - 1), ['u'] * (data.shape[1] - 1),
                                           nsamples=args.nsamples, period=args.period,
                                           cols_to_ignore=args.cols_to_ignore)
        if preds is not None:
            viz.set_preds(preds)
        else:
            print('No feature windows generated from replay; try a different file or adjust parameters')

        # Start a timer to refresh periodically if desired (re-run same data)
        def mock_tick():
            # re-run prediction on same data (useful for GUI activity)
            preds2, _ = pred.predict_from_rows(rows, ['CH'] * (data.shape[1] - 1), ['u'] * (data.shape[1] - 1),
                                               nsamples=args.nsamples, period=args.period,
                                               cols_to_ignore=args.cols_to_ignore)
            if preds2 is not None:
                viz.set_preds(preds2)

        mock_timer = QtCore.QTimer()
        mock_timer.timeout.connect(mock_tick)
        mock_timer.start(500)

        app.exec_()
        return

    # live mode
    acq = EEGAcquirer()
    muselsl_proc = None
    # optionally start muselsl streamer automatically
    if getattr(args, 'auto_stream', False):
        import shlex
        import subprocess
        try:
            cmd = [sys.executable, '-m', 'muselsl', 'stream']
            extra = args.muselsl_args.strip()
            if extra:
                cmd += shlex.split(extra)
            # run in background, discard output to avoid clutter
            muselsl_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print('Started muselsl streamer (pid:', muselsl_proc.pid, ')')
            # give it a moment to initialize
            time.sleep(2.0)
        except Exception as e:
            print('Failed to start muselsl automatically:', e)
        else:
            # register cleanup to terminate muselsl on exit
            import atexit

            def _terminate_muselsl(proc=muselsl_proc):
                try:
                    if proc is None:
                        return
                    # poll to see if it's still running
                    if proc.poll() is None:
                        try:
                            proc.terminate()
                            proc.wait(timeout=2.0)
                        except Exception:
                            try:
                                proc.kill()
                            except Exception:
                                pass
                except Exception:
                    pass

            atexit.register(_terminate_muselsl)
            # also try to hook into Qt app quit if available
            try:
                if 'app' in locals() and app is not None:
                    app.aboutToQuit.connect(_terminate_muselsl)
            except Exception:
                pass
    print('Connecting to EEG...')
    try:
        acq.connect()
    except RuntimeError:
        print('No EEG stream found. If you do not have a headset connected, run with --mock or --replay <file>')
        return
    acq.start()

    # start saving raw if requested
    if getattr(args, 'raw_out', None):
        try:
            acq.start_saving_raw(args.raw_out)
            print(f'Writing raw EEG samples to {args.raw_out}')
        except Exception as e:
            print('Failed to start raw saving:', e)

    # Initialize prediction counter for the visualizer
    prediction_counter = 0
    if getattr(args, 'summary_out', None) and os.path.exists(args.summary_out):
        try:
            with open(args.summary_out, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                last_line = None
                for last_line in reader:
                    pass
                if last_line:
                    prediction_counter = int(last_line[0]) + 1
                    print(f"Resuming from prediction count: {prediction_counter}")
        except Exception as e:
            print(f"Error reading summary file: {e}")
            prediction_counter = 0

    def estimate_sfreq(rows):
        # Estimate sampling frequency from recent timestamps (median dt)
        try:
            if len(rows) < 2:
                return 256.0
            ts = np.array([r[0] for r in rows[-min(len(rows), 256):]])
            diffs = np.diff(ts)
            diffs = diffs[diffs > 0]
            if len(diffs) == 0:
                return 256.0
            median_dt = float(np.median(diffs))
            if median_dt <= 0:
                return 256.0
            return 1.0 / median_dt
        except Exception:
            return 256.0


    def poll_and_predict():
        try:
            # Get the latest samples from the buffer
            rows = acq.get_buffer_copy()
            if not rows or len(rows) < 2:  # Need at least 2 samples to estimate sample rate
                return
                
            # Process the samples through the denoiser
            rows = den.process(rows)
            
            # Estimate sample rate from timestamps
            timestamps = [float(row[0]) for row in rows]
            if len(timestamps) > 1:
                sample_rates = 1.0 / np.diff(timestamps)
                sample_rate = np.median(sample_rates)
            else:
                sample_rate = 256.0  # Default fallback
            
            # Only process if we have enough samples
            min_samples = int(sample_rate * args.period)  # At least one window worth of data
            if len(rows) < min_samples:
                print(f"Waiting for more data: {len(rows)}/{min_samples} samples")
                return
                
            # Make predictions
            preds, n_windows = pred.predict_from_rows(
                rows, 
                channels=acq.channels, 
                units=acq.units,
                nsamples=args.nsamples, 
                period=args.period,
                cols_to_ignore=args.cols_to_ignore
            )
            
            # Update the visualization
            if preds is not None:
                viz.set_preds(preds)
                
                # Write predictions to summary file
                if getattr(args, 'summary_out', None):
                    try:
                        # Use current time as the timestamp for all predictions in this batch
                        current_time = time.time()
                        
                        # Ensure file exists and has header
                        need_header = not os.path.exists(args.summary_out) or os.path.getsize(args.summary_out) == 0
                        
                        with open(args.summary_out, 'a', newline='') as f:
                            writer = csv.writer(f)
                            if need_header:
                                writer.writerow(['window_index', 'timestamp', 'prediction', 'label'])
                            
                            # Get the next window index from the file if it exists
                            next_window_idx = 0
                            if not need_header:
                                with open(args.summary_out, 'r') as f_read:
                                    reader = csv.reader(f_read)
                                    next(reader, None)  # Skip header
                                    for row in reader:
                                        next_window_idx = max(next_window_idx, int(row[0]) + 1)
                            
                            # Write new predictions with sequential window indices and current timestamp
                            for i in range(len(preds)):
                                pred_label = int(preds[i])
                                label_name = LABEL_MAP.get(pred_label, str(pred_label))
                                writer.writerow([
                                    next_window_idx + i,
                                    current_time + (i * 0.5),  # Add 0.5s between predictions
                                    pred_label, 
                                    label_name
                                ])
                            
                            # Update the visualizer with the current prediction count
                            viz.prediction_count = next_window_idx + len(preds)
                        
                    except Exception as e:
                        print(f"Error writing to summary file: {e}")
                        
            else:
                print(f"No predictions generated. buffer_rows={len(rows)}")
                if len(rows) > 0:
                    print(f"First row sample: {rows[0][:5]}...")
                    
        except Exception as e:
            print(f"Error in poll_and_predict: {e}")
            import traceback
            traceback.print_exc()

    timer = QtCore.QTimer()
    timer.timeout.connect(poll_and_predict)
    timer.start(500)  # poll twice a second

    app.exec_()


if __name__ == '__main__':
    main()
