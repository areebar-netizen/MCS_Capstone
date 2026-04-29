from django.core.management.base import BaseCommand
from django.conf import settings
import sys
from pathlib import Path

# Add research directory to path for profiler import
RESEARCH_DIR = Path(__file__).resolve().parent.parent.parent.parent / 'research'
sys.path.insert(0, str(RESEARCH_DIR))

try:
    from profile_prediction_pipeline import PipelineProfiler
except ImportError:
    print("Error: profile_prediction_pipeline.py not found in research directory")
    sys.exit(1)

class Command(BaseCommand):
    help = 'Run comprehensive performance profiling of the EEG prediction pipeline'

    def add_arguments(self, parser):
        parser.add_argument(
            '--models-dir',
            type=str,
            default='core_engine/artifacts',
            help='Path to models directory (default: core_engine/artifacts)'
        )
        parser.add_argument(
            '--output-dir',
            type=str,
            default='.',
            help='Output directory for reports (default: current directory)'
        )
        parser.add_argument(
            '--quick',
            action='store_true',
            help='Run quick profiling (skip detailed cProfile)'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting EEG Prediction Pipeline Profiler...'))
        
        # Setup paths
        models_dir = Path(options['models_dir'])
        if not models_dir.is_absolute():
            # Relative to project root
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            models_dir = project_root / models_dir
        
        output_dir = Path(options['output_dir'])
        
        self.stdout.write(f"Models directory: {models_dir}")
        self.stdout.write(f"Output directory: {output_dir}")
        
        if not models_dir.exists():
            self.stdout.write(self.style.ERROR(f"Models directory not found: {models_dir}"))
            return
        
        # Initialize profiler
        try:
            profiler = PipelineProfiler(models_dir)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to initialize profiler: {e}"))
            return
        
        try:
            # Run profiling tests
            self.stdout.write(self.style.WARNING('Profiling Model Loading...'))
            profiler.profile_model_loading()
            
            self.stdout.write(self.style.WARNING('Profiling Prediction Pipeline...'))
            profiler.profile_prediction_pipeline()
            
            self.stdout.write(self.style.WARNING('Profiling Cache Performance...'))
            profiler.profile_cache_performance()
            
            if not options['quick']:
                self.stdout.write(self.style.WARNING('Running Detailed Profiling (cProfile)...'))
                profiler.profile_with_cprofile()
            
            # Generate report
            self.stdout.write(self.style.WARNING('Generating Performance Report...'))
            profiler.generate_report()
            
            self.stdout.write(self.style.SUCCESS('Profiling completed successfully!'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Profiling failed: {e}"))
            import traceback
            traceback.print_exc()
            return
