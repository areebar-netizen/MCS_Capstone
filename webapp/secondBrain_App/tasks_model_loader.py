"""
Celery tasks for model loading and Redis cache management.
Preloads models into Redis for fast access by workers.
"""

from celery import shared_task
from pathlib import Path
import logging
from django.conf import settings

from .services.redis_model_loader import redis_model_loader

logger = logging.getLogger(__name__)

@shared_task
def preload_models_to_redis():
    """Preload all models and artifacts into Redis cache."""
    try:
        # Get models directory from settings or default
        models_dir = getattr(settings, 'MODELS_DIR', None)
        if not models_dir:
            # Default to core_engine/artifacts
            from pathlib import Path
            import os
            base_dir = Path(settings.BASE_DIR).parent
            models_dir = base_dir / 'core_engine' / 'artifacts'
        
        models_dir = Path(models_dir)
        
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        logger.info(f"Starting model preload from {models_dir}")
        
        # Preload all models and artifacts
        results = redis_model_loader.preload_all_models(models_dir)
        
        logger.info(f"Model preload results: {results}")
        
        return {
            'status': 'completed',
            'results': results,
            'total_loaded': results['total_success']
        }
        
    except Exception as e:
        logger.error(f"Model preload failed: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }

@shared_task
def clear_model_cache():
    """Clear all models from Redis cache."""
    try:
        success = redis_model_loader.clear_redis_cache()
        
        if success:
            logger.info("Model cache cleared successfully")
            return {'status': 'completed', 'message': 'Cache cleared'}
        else:
            logger.error("Failed to clear model cache")
            return {'status': 'failed', 'message': 'Failed to clear cache'}
            
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        return {'status': 'failed', 'error': str(e)}

@shared_task
def check_model_cache():
    """Check if models are loaded in Redis cache."""
    try:
        from .services.redis_model_loader import get_model_from_redis, get_artifacts_from_redis
        
        # Check each model
        model_status = {}
        for model_name in ['random_forest', 'xgboost', 'stacked_model']:
            model = get_model_from_redis(model_name)
            model_status[model_name] = model is not None
        
        # Check artifacts
        artifacts = get_artifacts_from_redis()
        artifact_status = {
            'feature_selector': artifacts.get('feature_selector') is not None,
            'feature_scaler': artifacts.get('feature_scaler') is not None,
            'feature_info': artifacts.get('feature_info') is not None
        }
        
        total_cached = sum(model_status.values()) + sum(artifact_status.values())
        total_expected = len(model_status) + len(artifact_status)
        
        return {
            'status': 'completed',
            'models': model_status,
            'artifacts': artifact_status,
            'total_cached': total_cached,
            'total_expected': total_expected,
            'cache_ready': total_cached == total_expected
        }
        
    except Exception as e:
        logger.error(f"Cache check failed: {e}")
        return {'status': 'failed', 'error': str(e)}
