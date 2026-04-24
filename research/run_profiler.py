#!/usr/bin/env python3
"""
Standalone profiler runner script.
Usage: python run_profiler.py [--models-dir core_engine/artifacts] [--quick]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from profile_prediction_pipeline import main
except ImportError as e:
    print(f"Error importing profiler: {e}")
    print("Make sure profile_prediction_pipeline.py is in the research directory")
    sys.exit(1)

def main_wrapper():
    """Main wrapper with argument parsing."""
    parser = argparse.ArgumentParser(description='Profile EEG Prediction Pipeline')
    parser.add_argument(
        '--models-dir',
        type=str,
        default='core_engine/artifacts',
        help='Path to models directory (default: core_engine/artifacts)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick profiling (skip detailed cProfile)'
    )
    
    args = parser.parse_args()
    
    # Override models directory if specified
    if hasattr(args, 'models_dir'):
        import profile_prediction_pipeline
        original_main = profile_prediction_pipeline.main
        
        def patched_main():
            # Temporarily modify the models_dir in the profiler
            models_dir = Path(args.models_dir)
            if not models_dir.is_absolute():
                models_dir = PROJECT_ROOT / models_dir
            
            # This is a bit of a hack, but it allows us to override the models dir
            # without modifying the original profiler code too much
            profiler = profile_prediction_pipeline.PipelineProfiler(models_dir)
            
            try:
                # Run profiling tests
                profiler.profile_model_loading()
                profiler.profile_prediction_pipeline()
                profiler.profile_cache_performance()
                
                if not args.quick:
                    profiler.profile_with_cprofile()
                
                # Generate report
                profiler.generate_report()
                
            except Exception as e:
                print(f"Profiling failed: {e}")
                import traceback
                traceback.print_exc()
        
        profile_prediction_pipeline.main = patched_main
    
    # Run the main profiler
    try:
        main()
    except Exception as e:
        print(f"Profiler execution failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main_wrapper()
