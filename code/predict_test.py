#!/usr/bin/env python3
"""
Predict labels for files in dataset/test using saved models in models_out/.
Mimics the behavior of live_data/live_predict.py for consistent results.

Usage:
    python3 code/predict_test.py [--models models_out] [--testdir dataset/test] [--model random_forest|xgboost|stacked_model]

Output:
    - Prints per-file summary (counts and seconds per label)
    - Writes `prediction_summary.csv` in the current directory with a row per file
"""

import argparse
import csv
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats

# Import enhanced feature extraction functions
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from enhanced_feature_extraction import load_preprocessing_artifacts, apply_feature_pipeline
from EEG_feature_extraction_adv import generate_feature_vectors_from_matrix

# Constants
LABEL_MAP = {0: 'relaxed', 1: 'neutral', 2: 'concentrating'}
CSV_HEADER = ['filename', 'n_windows', 'total_seconds', 
              'relaxed_seconds', 'neutral_seconds', 'concentrating_seconds',
              'predicted_label', 'confidence']


def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description='Predict labels for EEG data files.')
    parser.add_argument('--models', default='models_out', 
                       help='directory containing model joblib files')
    parser.add_argument('--testdir', default='dataset/test', 
                       help='directory with test CSV files')
    parser.add_argument('--model', default='xgboost', 
                       choices=['random_forest', 'xgboost', 'stacked_model'],
                       help='which saved model to use')
    parser.add_argument('--out', default='prediction_summary.csv', 
                       help='CSV file to write summary results')
    return parser.parse_args()


def load_model_artifacts(models_dir: Path, model_name: str) -> Tuple[Any, Any, Any, int]:
    """Load model and feature selector with error handling."""
    model_path = models_dir / f"{model_name}.joblib"
    selector_path = models_dir / "feature_selector.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    model = joblib.load(model_path)
    selector = joblib.load(selector_path) if selector_path.exists() else None
    
    # Load preprocessing artifacts for enhanced features
    try:
        scaler, feature_info = load_preprocessing_artifacts('preprocessing_artifacts')
    except Exception as e:
        print(f"Warning: Could not load preprocessing artifacts: {e}")
        scaler = None
        feature_info = None
    
    # Get the expected number of features from the model
    expected_features = getattr(model, 'n_features_in_', None)
    if expected_features is None and selector is not None:
        expected_features = selector.n_features_in_
    
    if expected_features is None:
        raise ValueError("Could not determine expected number of features")
    
    print(f"Loaded {model_name} with {expected_features} features")
    return model, selector, expected_features, (scaler, feature_info)


def process_file(model: Any, file_path: Path, selector: Any, expected_features: int, preprocessing_artifacts: Any) -> Optional[Dict]:
    """Process a single file and return prediction results."""
    print(f"Processing {file_path.name}...")
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return None
    
    if len(df) < 2:
        print(f"  Not enough samples in {file_path}")
        return None
    
    # Convert to numpy array (skip timestamp column if it exists)
    data = df.values
    if 'time' in df.columns or 'timestamp' in df.columns:
        data = df.drop(columns=['time', 'timestamp'], errors='ignore').values
    
    # Use enhanced feature extraction pipeline if artifacts are available
    if preprocessing_artifacts[0] is not None and preprocessing_artifacts[1] is not None:
        try:
            # Apply enhanced feature extraction pipeline
            # Generate raw features first
            vectors, _ = generate_feature_vectors_from_matrix(
                data,
                nsamples=150,
                period=1.0,
                state=None,
                remove_redundant=True,
                cols_to_ignore=-1
            )
            
            if vectors is None or len(vectors) == 0:
                print(f"  No feature vectors generated for {file_path.name}")
                return None
            
            # Apply enhanced preprocessing (scaling + feature selection)
            scaler, feature_info = preprocessing_artifacts
            X_processed = apply_feature_pipeline(vectors, scaler, feature_info)
            
            if X_processed is None:
                print(f"  Enhanced preprocessing failed for {file_path.name}")
                return None
                
            X = X_processed
            
        except Exception as e:
            print(f"  Enhanced feature extraction failed: {e}")
            return None
    else:
        # Fallback to old method if no preprocessing artifacts
        try:
            vectors, _ = generate_feature_vectors_from_matrix(
                data,
                nsamples=150,
                period=1.0,
                state=None,
                remove_redundant=True,
                cols_to_ignore=-1
            )
        except Exception as e:
            print(f"  Feature extraction failed: {e}")
            return None
        
        if vectors is None or len(vectors) == 0:
            print(f"  No feature vectors generated for {file_path.name}")
            return None
        
        X = np.asarray(vectors, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Apply feature selection if available
        if selector is not None:
            try:
                X = selector.transform(X)
            except Exception as e:
                print(f"  Feature selection failed: {e}")
                return None
    
    # Get predictions
    try:
        predictions = model.predict(X)
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X)
            confidences = np.max(probas, axis=1)
        else:
            confidences = np.ones(len(predictions))
    except Exception as e:
        print(f"  Prediction failed: {e}")
        return None
    
    # Calculate statistics
    n_windows = len(predictions)
    total_seconds = n_windows * 0.5  # 0.5s per window
    
    # Count predictions per class
    counts = {label: 0 for label in LABEL_MAP.values()}
    for p in predictions:
        name = LABEL_MAP.get(int(p), str(p))
        counts[name] = counts.get(name, 0) + 1
    
    # Calculate durations and confidence
    durations = {k: v * 0.5 for k, v in counts.items()}  # 0.5s per window
    avg_confidence = float(np.mean(confidences)) if len(confidences) > 0 else 0.0
    
    # Get most common prediction
    if len(predictions) > 0:
        predicted_label_idx = int(stats.mode(predictions, keepdims=True)[0][0])
        predicted_label = LABEL_MAP.get(predicted_label_idx, 'unknown')
    else:
        predicted_label = 'unknown'
    
    return {
        'filename': file_path.name,
        'n_windows': n_windows,
        'total_seconds': total_seconds,
        'relaxed_seconds': durations.get('relaxed', 0),
        'neutral_seconds': durations.get('neutral', 0),
        'concentrating_seconds': durations.get('concentrating', 0),
        'predicted_label': predicted_label,
        'confidence': avg_confidence
    }


def save_results(results: List[Dict], output_file: str) -> None:
    """Save prediction results to CSV file."""
    if not results:
        print("No results to save")
        return
        
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_file}")


def main() -> int:
    """Main prediction pipeline."""
    args = parse_arguments()
    test_dir = Path(args.testdir)
    models_dir = Path(args.models)
    
    try:
        # Load model and get expected feature count
        model, selector, expected_features, preprocessing_artifacts = load_model_artifacts(models_dir, args.model)
        results = []
        
        # Process each CSV file in test directory
        for file_path in sorted(test_dir.glob('*.csv')):
            if file_path.is_file():
                result = process_file(model, file_path, selector, expected_features, preprocessing_artifacts)
                if result:
                    results.append(result)
                    print(f"  Processed {file_path.name}: {result}")
        
        # Save results if any files were processed
        if results:
            save_results(results, args.out)
            return 0
        else:
            print("No valid files processed")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())