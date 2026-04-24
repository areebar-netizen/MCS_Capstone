"""
Prediction Service Optimizer
Provides performance optimizations and edge case handling for EEG predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import logging
from typing import List, Tuple, Optional, Dict, Any
from functools import lru_cache
import threading
import warnings

logger = logging.getLogger(__name__)

class PredictionOptimizer:
    """Optimizes prediction performance and handles edge cases."""
    
    def __init__(self):
        self.prediction_cache = {}
        self.cache_lock = threading.Lock()
        self.performance_stats = {
            'total_predictions': 0,
            'avg_prediction_time': 0.0,
            'failed_predictions': 0,
            'cache_hits': 0
        }
    
    def validate_input_data(self, rows: List[List[float]]) -> Tuple[bool, str]:
        """
        Validate input EEG data for common edge cases.
        
        Args:
            rows: Raw EEG data rows
            
        Returns:
            (is_valid, error_message)
        """
        if not rows:
            return False, "Empty input data"
        
        if len(rows) < 2:
            return False, "Insufficient data (need at least 2 rows)"
        
        # Convert to numpy array for validation
        try:
            arr = np.array(rows, dtype=float)
        except (ValueError, TypeError) as e:
            return False, f"Invalid data format: {e}"
        
        # Check for NaN or infinite values
        if np.any(np.isnan(arr)):
            return False, "Data contains NaN values"
        
        if np.any(np.isinf(arr)):
            return False, "Data contains infinite values"
        
        # Check data range (EEG values should be reasonable)
        data_values = arr[:, 1:]  # Skip timestamp column
        if np.max(np.abs(data_values)) > 1e6:  # 1 million uV is unreasonable
            return False, f"Data values out of reasonable range (max: {np.max(np.abs(data_values)):.2e})"
        
        # Check timestamp consistency
        timestamps = arr[:, 0]
        if len(timestamps) > 1:
            time_diffs = np.diff(timestamps)
            if np.any(time_diffs <= 0):
                return False, "Timestamps are not monotonically increasing"
            
            # Check for large gaps in data (more than 1 second)
            if np.any(time_diffs > 1.0):
                logger.warning(f"Large gaps detected in EEG data (max gap: {np.max(time_diffs):.2f}s)")
        
        return True, ""
    
    def preprocess_input_data(self, rows: List[List[float]], target_length: Optional[int] = None) -> List[List[float]]:
        """
        Preprocess input data to handle edge cases and optimize performance.
        
        Args:
            rows: Raw EEG data rows
            target_length: Optional target length for data alignment
            
        Returns:
            Preprocessed data rows
        """
        if not rows:
            return rows
        
        arr = np.array(rows, dtype=float)
        
        # Handle data length alignment
        if target_length and len(arr) > target_length:
            # Take the most recent data
            arr = arr[-target_length:]
            logger.debug(f"Trimmed data to {target_length} rows")
        elif target_length and len(arr) < target_length:
            # Pad with repeated last row if needed (better than interpolation for real-time)
            last_row = arr[-1:]
            padding_needed = target_length - len(arr)
            padding = np.repeat(last_row, padding_needed, axis=0)
            arr = np.vstack([arr, padding])
            logger.debug(f"Padded data to {target_length} rows")
        
        # Remove duplicate timestamps (common in streaming data)
        timestamps = arr[:, 0]
        unique_mask = np.concatenate([[True], np.diff(timestamps) > 1e-6])  # 1 microsecond tolerance
        if not np.all(unique_mask):
            arr = arr[unique_mask]
            logger.debug(f"Removed {np.sum(~unique_mask)} duplicate rows")
        
        return arr.tolist()
    
    def optimize_feature_extraction(self, rows: List[List[float]], nsamples: int = 150, period: float = 1.0) -> Dict[str, Any]:
        """
        Optimize feature extraction parameters based on input data.
        
        Args:
            rows: EEG data rows
            nsamples: Number of samples for feature extraction
            period: Time period for each window
            
        Returns:
            Optimization suggestions
        """
        if not rows:
            return {}
        
        arr = np.array(rows, dtype=float)
        total_duration = arr[-1, 0] - arr[0, 0] if len(arr) > 1 else 0
        sampling_rate = len(arr) / total_duration if total_duration > 0 else 256  # Default to 256 Hz
        
        optimizations = {
            'sampling_rate': sampling_rate,
            'total_duration': total_duration,
            'recommended_nsamples': nsamples,
            'recommended_period': period,
            'max_windows': max(1, int(total_duration / period))
        }
        
        # Adjust parameters based on data characteristics
        if total_duration < period:
            optimizations['recommended_period'] = total_duration * 0.8  # Use 80% of available data
            optimizations['max_windows'] = 1
        
        if sampling_rate < 100:  # Low sampling rate
            optimizations['recommended_nsamples'] = min(nsamples, int(sampling_rate * period * 0.5))
        
        return optimizations
    
    @lru_cache(maxsize=128)
    def get_cached_feature_vector(self, data_hash: int, nsamples: int, period: float) -> Optional[np.ndarray]:
        """
        Cache feature vectors to avoid recomputation for identical data.
        
        Args:
            data_hash: Hash of input data
            nsamples: Number of samples
            period: Time period
            
        Returns:
            Cached feature vector or None
        """
        with self.cache_lock:
            cache_key = (data_hash, nsamples, period)
            if cache_key in self.prediction_cache:
                self.performance_stats['cache_hits'] += 1
                return self.prediction_cache[cache_key]
            return None
    
    def cache_feature_vector(self, data_hash: int, nsamples: int, period: float, features: np.ndarray):
        """
        Cache computed feature vectors.
        
        Args:
            data_hash: Hash of input data
            nsamples: Number of samples
            period: Time period
            features: Feature vector to cache
        """
        with self.cache_lock:
            cache_key = (data_hash, nsamples, period)
            # Limit cache size to prevent memory issues
            if len(self.prediction_cache) < 100:
                self.prediction_cache[cache_key] = features.copy()
    
    def compute_data_hash(self, rows: List[List[float]]) -> int:
        """
        Compute a hash of input data for caching purposes.
        
        Args:
            rows: EEG data rows
            
        Returns:
            Hash value
        """
        if not rows:
            return 0
        
        # Use a subset of data for hashing (first, middle, last rows)
        arr = np.array(rows, dtype=float)
        if len(arr) <= 10:
            sample_data = arr
        else:
            sample_data = arr[[0, len(arr)//2, -1]]
        
        return hash(sample_data.tobytes())
    
    def update_performance_stats(self, prediction_time: float, success: bool = True):
        """
        Update performance statistics.
        
        Args:
            prediction_time: Time taken for prediction
            success: Whether prediction was successful
        """
        self.performance_stats['total_predictions'] += 1
        
        if success:
            # Update running average
            current_avg = self.performance_stats['avg_prediction_time']
            total = self.performance_stats['total_predictions']
            new_avg = (current_avg * (total - 1) + prediction_time) / total
            self.performance_stats['avg_prediction_time'] = new_avg
        else:
            self.performance_stats['failed_predictions'] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.
        
        Returns:
            Performance statistics
        """
        stats = self.performance_stats.copy()
        
        if stats['total_predictions'] > 0:
            stats['success_rate'] = (stats['total_predictions'] - stats['failed_predictions']) / stats['total_predictions']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_predictions']
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        stats['cache_size'] = len(self.prediction_cache)
        
        return stats
    
    def clear_cache(self):
        """Clear the prediction cache."""
        with self.cache_lock:
            self.prediction_cache.clear()
            logger.info("Prediction cache cleared")

# Global optimizer instance
prediction_optimizer = PredictionOptimizer()

def optimize_prediction_pipeline(rows: List[List[float]], nsamples: int = 150, period: float = 1.0) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Optimize the prediction pipeline with validation and preprocessing.
    
    Args:
        rows: Raw EEG data rows
        nsamples: Number of samples for feature extraction
        period: Time period for each window
        
    Returns:
        (optimized_rows, optimization_info)
    """
    optimizer = prediction_optimizer
    
    # Validate input data
    is_valid, error_msg = optimizer.validate_input_data(rows)
    if not is_valid:
        raise ValueError(f"Invalid input data: {error_msg}")
    
    # Get optimization suggestions
    optimizations = optimizer.optimize_feature_extraction(rows, nsamples, period)
    
    # Preprocess data
    optimized_rows = optimizer.preprocess_input_data(
        rows, 
        target_length=optimizations.get('recommended_nsamples')
    )
    
    # Add performance info
    optimizations['data_hash'] = optimizer.compute_data_hash(optimized_rows)
    optimizations['original_rows'] = len(rows)
    optimizations['optimized_rows'] = len(optimized_rows)
    
    return optimized_rows, optimizations
