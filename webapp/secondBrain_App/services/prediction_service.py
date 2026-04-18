from pathlib import Path
import sys
import time
from unittest import result
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



# Import custom modules
from .EEG_feature_extraction_adv import generate_feature_vectors_from_matrix
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../core_engine'))
from enhanced_feature_extraction import (load_preprocessing_artifacts, apply_feature_pipeline)
import joblib

# Attempt to import Stream.py if it exists


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


def load_model(models_dir: Path, model_name: str):
    """Loads model, feature selector, and enhanced preprocessing artifacts."""
    mapping = {
        'random_forest': 'random_forest.joblib',
        'xgboost': 'xgboost.joblib',
        'stacked_model': 'stacked_model.joblib'
    }
    
    model_file = Path(models_dir) / mapping.get(model_name, f"{model_name}.joblib")
    print(f"--- ATTEMPTING TO LOAD: {model_file} ---")
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")
    
    model = joblib.load(model_file)
    print(f"{model_file} loaded---")
    
    # Load standard feature selector (used if enhanced artifacts missing)
    selector_path = Path(models_dir) / 'feature_selector.joblib'
    selector = joblib.load(selector_path) if selector_path.exists() else None
    
    # Load enhanced artifacts from correct location
    try:
        # Look for preprocessing artifacts in core_engine/artifacts/preprocessing_artifacts/
        base_dir = Path(models_dir).parent.parent / 'core_engine' / 'artifacts' / 'preprocessing_artifacts'
        print(f"--- LOOKING FOR ARTIFACTS IN: {base_dir} ---")
        scaler, feature_info = load_preprocessing_artifacts(base_dir)
        
        # Validate that we have the required selected_indices
        if feature_info is not None and 'selected_indices' not in feature_info:
            raise ValueError("selected_indices not found in feature_info.pkl - artifacts may be corrupted")
        
        print(f"✅ Scaler loaded. Expected input features: {scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 'unknown'}")
        print(f"✅ Feature info indices count: {len(feature_info.get('selected_indices', []))} (Best Features confirmed)")
        
    except Exception as e:
        print(f"❌ Failed to load enhanced artifacts: {e}")
        scaler, feature_info = None, None
    
    return model, selector, (scaler, feature_info)


class Predictor:
    def __init__(self, models_dir: Union[str, Path], model_name: str = 'stacked_model'):
        print("Predictor class initilaising")
        self.model, self.feature_selector, self.preprocessing_artifacts = load_model(models_dir, model_name)
        print("models loaded")
        self.last_prediction = None
        self.last_confidence = 0.0
        print("Predictor class initilaised")

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
            print("generate_feature_vectors_from_matrix done")
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
                print("apply_feature_pipeline done")
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
            print("inside preditct #4")
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




# -----------------------------------------------------------------------------
# Prediction Class
# -----------------------------------------------------------------------------

class PredictionService:
    def __init__(self, models_dir: Union[str, Path], model_name: str = 'stacked_model'):
        print("Initializing PredictionService...")
        self.denoiser = Denoiser()
        print("Denoiser loaded.")
        self.predictor = Predictor(models_dir, model_name)
        print("Predictor loaded.")

    def run(self, rows: List[List[float]], nsamples: int = 150, period: float=1.0, cols_to_ignore: int = -1):
        print("Running Predservice")
        rows = self.denoiser.process(rows)
        print(f"Input rows size: {len(rows)}")
        preds, n_windows, confidence = self.predictor.predict_from_rows(rows, nsamples = nsamples, period = period, cols_to_ignore=cols_to_ignore)
        if preds is None:
            return {
                'ok': False,
                'message': 'No valid windows for prediction.'
            }
        
        counts = {lbl: 0 for lbl in LABEL_MAP.values()}
        for p in preds:
            name = LABEL_MAP.get(int(p), str(p))
            counts[name] = counts.get(name, 0) + 1
        
        # Calculate durations (0.5s per window)
        durations = {k: v * 0.5 for k, v in counts.items()}
        total_seconds = n_windows * 0.5
        
        # Get predicted label (most common)
        predicted_label = max(counts, key=counts.get)

        window_labels = [LABEL_MAP.get(int(p), str(p)) for p in preds]
        
        # Convert predictions to label names for window_labels
        window_labels = [LABEL_MAP.get(int(p), str(p)) for p in preds]
        
        # Create result dict like predict_test.py
        result = {
            'n_windows': int(n_windows),
            'total_seconds': float(total_seconds),
            'relaxed_seconds': float(durations.get('relaxed', 0)),
            'neutral_seconds': float(durations.get('neutral', 0)),
            'concentrating_seconds': float(durations.get('concentrating', 0)),
            'predicted_label': str(predicted_label),
            'confidence': float(confidence),
            'window_labels': window_labels,  # Add individual window labels
        }
        print(f"  Full result: {result}")

        return result
            

# def process_csv_mode(args):
#     """Process static CSV files."""
#     print(f"Processing CSV files in: {args.csv_dir}")
    
#     den = Denoiser()
#     pred = Predictor(args.models, args.model)
    
#     csv_dir = Path(args.csv_dir)
#     csv_files = sorted(csv_dir.glob('*.csv'))
    
#     if not csv_files:
#         print("No CSV files found.")
#         return

#     print(f"Found {len(csv_files)} files.")
    
#     # Initialize summary rows for CSV output
#     summary_rows = []

#     for csv_file in csv_files:
#         try:
#             df = pd.read_csv(csv_file)
#             rows = df.values.tolist()
            
#             # Denoise
#             rows = den.process(rows)
            
#             # Predict
#             preds, n_windows, confidence = pred.predict_from_rows(rows, nsamples=args.nsamples, period=args.period)
            
#             if preds is not None:
#                 counts = {lbl: 0 for lbl in LABEL_MAP.values()}
#                 for p in preds:
#                     name = LABEL_MAP.get(int(p), str(p))
#                     counts[name] = counts.get(name, 0) + 1
                
#                 # Calculate durations (0.5s per window)
#                 durations = {k: v * 0.5 for k, v in counts.items()}
#                 total_seconds = n_windows * 0.5
                
#                 # Get predicted label (most common)
#                 predicted_label = max(counts, key=counts.get)
                
#                 print(f"{csv_file.name}: {n_windows} windows")
#                 for lbl, count in counts.items():
#                     print(f"  {lbl}: {count}")
                
#                 # Create result dict like predict_test.py
#                 result = {
#                     'filename': csv_file.name,
#                     'n_windows': n_windows,
#                     'total_seconds': total_seconds,
#                     'relaxed_seconds': durations.get('relaxed', 0),
#                     'neutral_seconds': durations.get('neutral', 0),
#                     'concentrating_seconds': durations.get('concentrating', 0),
#                     'predicted_label': predicted_label,
#                     'confidence': confidence
#                 }
#                 print(f"  Full result: {result}")
                
#                 # Add to summary rows for CSV output
#                 summary_rows.append([
#                     csv_file.name,
#                     n_windows,
#                     total_seconds,
#                     durations.get('relaxed', 0),
#                     durations.get('neutral', 0),
#                     durations.get('concentrating', 0),
#                     predicted_label,
#                     confidence
#                 ])
#             else:
#                 print(f"{csv_file.name}: No valid windows.")
                
#         except Exception as e:
#             print(f"Error processing {csv_file.name}: {e}")
    
#     # Write summary CSV if requested
#     if args.summary_out and summary_rows:
#         try:
#             with open(args.summary_out, 'w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(['filename', 'n_windows', 'total_seconds', 
#                                'relaxed_seconds', 'neutral_seconds', 'concentrating_seconds',
#                                'predicted_label', 'confidence'])
#                 writer.writerows(summary_rows)
#             print(f"Summary written to {args.summary_out}")
#         except Exception as e:
#             print(f"Error writing summary file: {e}")

