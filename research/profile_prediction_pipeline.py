#!/usr/bin/env python3
"""
Comprehensive Performance Profiling for EEG Prediction Pipeline
Profiles the complete pipeline to identify bottlenecks and optimize performance.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Any, Tuple
import cProfile
import pstats
import io
from contextlib import contextmanager

# Setup Django
BASE_DIR = Path(__file__).resolve().parent.parent  # Go up to project root
sys.path.append(str(BASE_DIR / 'webapp'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'secondBrain.settings')

import django
django.setup()

from secondBrain_App.services.prediction_service import PredictionService
from secondBrain_App.services.prediction_optimizer import prediction_optimizer
from secondBrain_App.services.model_cache import model_cache
from secondBrain_App.utils.cache_manager import cache_manager

class PipelineProfiler:
    """Comprehensive profiler for the EEG prediction pipeline."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.results = {
            'model_loading': {},
            'prediction_pipeline': {},
            'cache_performance': {},
            'memory_usage': {},
            'bottlenecks': []
        }
    
    @contextmanager
    def timer(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.results['memory_usage'][operation_name] = {
                'duration': duration,
                'memory_before': start_memory,
                'memory_after': end_memory,
                'memory_delta': memory_delta
            }
            
            print(f"{operation_name}: {duration:.3f}s, Memory: {memory_delta:+.1f}MB")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _generate_test_data(self, duration_seconds: int = 10, sampling_rate: int = 256) -> List[List[float]]:
        """Generate realistic EEG test data."""
        n_samples = duration_seconds * sampling_rate
        timestamps = np.linspace(0, duration_seconds, n_samples)
        
        # Generate realistic EEG-like signals with different frequencies
        channels = 4  # Typical 4-channel EEG
        data = np.zeros((n_samples, channels + 1))  # +1 for timestamp
        
        data[:, 0] = timestamps
        
        # Generate different brainwave patterns for each channel
        for ch in range(channels):
            # Mix of different frequencies (alpha, beta, theta, delta waves)
            signal = (
                10 * np.sin(2 * np.pi * 10 * timestamps) +  # Alpha (10 Hz)
                5 * np.sin(2 * np.pi * 20 * timestamps) +    # Beta (20 Hz)
                3 * np.sin(2 * np.pi * 5 * timestamps) +     # Theta (5 Hz)
                2 * np.sin(2 * np.pi * 2 * timestamps) +     # Delta (2 Hz)
                np.random.normal(0, 2, n_samples)             # Noise
            )
            data[:, ch + 1] = signal
        
        return data.tolist()
    
    def profile_model_loading(self):
        """Profile model loading performance."""
        print("\n" + "="*60)
        print("PROFILING MODEL LOADING")
        print("="*60)
        
        models_to_test = ['random_forest', 'xgboost', 'stacked_model']
        
        for model_name in models_to_test:
            print(f"\nTesting {model_name}:")
            
            # Clear cache first
            model_cache.clear_cache()
            
            with self.timer(f"load_{model_name}_first"):
                try:
                    model_data = model_cache.get_model(self.models_dir, model_name)
                    self.results['model_loading'][f"{model_name}_first"] = {
                        'success': True,
                        'load_time': model_data['load_time']
                    }
                except Exception as e:
                    self.results['model_loading'][f"{model_name}_first"] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Test cache hit
            with self.timer(f"load_{model_name}_cached"):
                try:
                    model_data = model_cache.get_model(self.models_dir, model_name)
                    self.results['model_loading'][f"{model_name}_cached"] = {
                        'success': True,
                        'cache_hit': True
                    }
                except Exception as e:
                    self.results['model_loading'][f"{model_name}_cached"] = {
                        'success': False,
                        'error': str(e)
                    }
    
    def profile_prediction_pipeline(self):
        """Profile the complete prediction pipeline."""
        print("\n" + "="*60)
        print("PROFILING PREDICTION PIPELINE")
        print("="*60)
        
        # Initialize prediction service
        with self.timer("init_prediction_service"):
            try:
                prediction_service = PredictionService(models_dir=self.models_dir, model_name='xgboost')
                print("Prediction service initialized successfully")
            except Exception as e:
                print(f"Failed to initialize prediction service: {e}")
                return
        
        # Test with different data sizes
        test_durations = [5, 10, 30]  # seconds
        
        for duration in test_durations:
            print(f"\nTesting with {duration}s of data:")
            test_data = self._generate_test_data(duration)
            
            # Profile complete pipeline
            with self.timer(f"predict_{duration}s_complete"):
                try:
                    result = prediction_service.run(test_data, nsamples=150, period=1.0, cols_to_ignore=-1)
                    
                    if result.get('ok'):
                        self.results['prediction_pipeline'][f"{duration}s"] = {
                            'success': True,
                            'n_windows': result.get('n_windows'),
                            'confidence': result.get('confidence'),
                            'predicted_label': result.get('predicted_label')
                        }
                    else:
                        self.results['prediction_pipeline'][f"{duration}s"] = {
                            'success': False,
                            'error': result.get('message', 'Unknown error')
                        }
                        
                except Exception as e:
                    self.results['prediction_pipeline'][f"{duration}s"] = {
                        'success': False,
                        'error': str(e)
                    }
    
    def profile_cache_performance(self):
        """Profile cache performance."""
        print("\n" + "="*60)
        print("PROFILING CACHE PERFORMANCE")
        print("="*60)
        
        # Test Redis cache performance
        test_data = {
            'status': 'active',
            'state': 'CONCENTRATING',
            'confidence': 85.5,
            'waves': {
                'delta': 23.4, 'theta': 18.7, 'alpha': 45.2,
                'beta': 67.8, 'gamma': 34.1
            }
        }
        
        user_email = 'profile_test@example.com'
        iterations = 100
        
        # Test write performance
        write_times = []
        for i in range(iterations):
            start = time.time()
            success = cache_manager.set_live_eeg_data(user_email, test_data)
            end = time.time()
            write_times.append(end - start)
        
        # Test read performance
        read_times = []
        for i in range(iterations):
            start = time.time()
            data = cache_manager.get_live_eeg_data(user_email)
            end = time.time()
            read_times.append(end - start)
        
        self.results['cache_performance'] = {
            'write_avg_ms': np.mean(write_times) * 1000,
            'write_max_ms': np.max(write_times) * 1000,
            'read_avg_ms': np.mean(read_times) * 1000,
            'read_max_ms': np.max(read_times) * 1000,
            'write_ops_per_sec': 1000 / (np.mean(write_times) * 1000),
            'read_ops_per_sec': 1000 / (np.mean(read_times) * 1000)
        }
        
        print(f"Cache Write: {self.results['cache_performance']['write_avg_ms']:.3f}ms avg, "
              f"{self.results['cache_performance']['write_ops_per_sec']:.0f} ops/sec")
        print(f"Cache Read:  {self.results['cache_performance']['read_avg_ms']:.3f}ms avg, "
              f"{self.results['cache_performance']['read_ops_per_sec']:.0f} ops/sec")
        
        # Cleanup
        cache_manager.cleanup_user_cache(user_email)
    
    def profile_with_cprofile(self):
        """Profile using cProfile for detailed analysis."""
        print("\n" + "="*60)
        print("DETAILED PROFILING WITH CPROFILE")
        print("="*60)
        
        # Create a test function to profile
        def test_prediction():
            prediction_service = PredictionService(models_dir=self.models_dir, model_name='xgboost')
            test_data = self._generate_test_data(10)  # 10 seconds of data
            result = prediction_service.run(test_data, nsamples=150, period=1.0, cols_to_ignore=-1)
            return result
        
        # Profile the function
        pr = cProfile.Profile()
        pr.enable()
        
        try:
            result = test_prediction()
            print(f"Test prediction completed: {result.get('ok', False)}")
        except Exception as e:
            print(f"Test prediction failed: {e}")
        
        pr.disable()
        
        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        profile_output = s.getvalue()
        
        # Save detailed profile
        with open('prediction_profile_detailed.txt', 'w') as f:
            f.write(profile_output)
        
        print("Detailed profile saved to 'prediction_profile_detailed.txt'")
        
        # Extract bottleneck information
        lines = profile_output.split('\n')
        for line in lines[:15]:  # Top 15 functions
            if 'function calls' in line.lower() or 'ncalls' in line.lower():
                continue
            if line.strip() and not line.startswith(' '):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        cum_time = float(parts[3])
                        if cum_time > 0.1:  # Functions taking more than 0.1s
                            func_name = ' '.join(parts[5:])
                            self.results['bottlenecks'].append({
                                'function': func_name,
                                'cumulative_time': cum_time,
                                'percentage': cum_time / float(parts[3]) * 100 if parts[3] != '0.000' else 0
                            })
                    except (ValueError, IndexError):
                        continue
    
    def generate_report(self):
        """Generate comprehensive performance report."""
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        # Model loading performance
        print("\n1. MODEL LOADING PERFORMANCE:")
        for model, data in self.results['model_loading'].items():
            if data.get('success'):
                load_time = data.get('load_time', 0)
                cache_hit = data.get('cache_hit', False)
                cache_indicator = " (CACHED)" if cache_hit else ""
                print(f"  {model}: {load_time:.3f}s{cache_indicator}")
            else:
                print(f"  {model}: FAILED - {data.get('error', 'Unknown error')}")
        
        # Prediction pipeline performance
        print("\n2. PREDICTION PIPELINE PERFORMANCE:")
        for duration, data in self.results['prediction_pipeline'].items():
            if data.get('success'):
                print(f"  {duration}: SUCCESS - {data.get('n_windows')} windows, "
                      f"confidence: {data.get('confidence', 0):.3f}")
            else:
                print(f"  {duration}: FAILED - {data.get('error', 'Unknown error')}")
        
        # Cache performance
        cache_perf = self.results['cache_performance']
        if cache_perf:
            print("\n3. CACHE PERFORMANCE:")
            print(f"  Write: {cache_perf['write_avg_ms']:.3f}ms avg, {cache_perf['write_ops_per_sec']:.0f} ops/sec")
            print(f"  Read:  {cache_perf['read_avg_ms']:.3f}ms avg, {cache_perf['read_ops_per_sec']:.0f} ops/sec")
        
        # Bottlenecks
        if self.results['bottlenecks']:
            print("\n4. PERFORMANCE BOTTLENECKS:")
            for i, bottleneck in enumerate(self.results['bottlenecks'][:5], 1):
                print(f"  {i}. {bottleneck['function']}: {bottleneck['cumulative_time']:.3f}s")
        
        # Memory usage
        print("\n5. MEMORY USAGE:")
        for operation, data in self.results['memory_usage'].items():
            print(f"  {operation}: {data['duration']:.3f}s, {data['memory_delta']:+.1f}MB")
        
        # Recommendations
        print("\n6. OPTIMIZATION RECOMMENDATIONS:")
        recommendations = []
        
        # Check model loading times
        model_times = [data.get('load_time', 0) for data in self.results['model_loading'].values() 
                      if data.get('success') and not data.get('cache_hit')]
        if model_times and max(model_times) > 2.0:
            recommendations.append("Consider model optimization (quantization, pruning) for faster loading")
        
        # Check cache performance
        if cache_perf and cache_perf['write_avg_ms'] > 1.0:
            recommendations.append("Cache write performance is slow - consider Redis optimization")
        
        # Check bottlenecks
        if self.results['bottlenecks']:
            top_bottleneck = max(self.results['bottlenecks'], key=lambda x: x['cumulative_time'])
            recommendations.append(f"Focus optimization on: {top_bottleneck['function']}")
        
        if not recommendations:
            recommendations.append("Performance looks good - no major bottlenecks detected")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Save full report
        report_file = 'performance_report.json'
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nFull report saved to '{report_file}'")

def main():
    """Main profiling function."""
    print("Starting comprehensive EEG prediction pipeline profiling...")
    
    # Setup
    models_dir = Path(__file__).parent / 'core_engine' / 'artifacts'
    
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        print("Please ensure models are available before running profiling")
        return
    
    profiler = PipelineProfiler(models_dir)
    
    try:
        # Run all profiling tests
        profiler.profile_model_loading()
        profiler.profile_prediction_pipeline()
        profiler.profile_cache_performance()
        profiler.profile_with_cprofile()
        
        # Generate comprehensive report
        profiler.generate_report()
        
        print("\n" + "="*60)
        print("PROFILING COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"Profiling failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
