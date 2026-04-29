"""
Redis-based Model Loader for Celery Workers
Loads models once and stores in Redis for fast access across workers.
"""

import os
import pickle
import logging
from pathlib import Path
from django.core.cache import cache
import joblib

logger = logging.getLogger(__name__)

class RedisModelLoader:
    """Load and cache models in Redis for Celery workers."""
    
    def __init__(self):
        self.model_cache_keys = {
            'random_forest': 'model_random_forest',
            'xgboost': 'model_xgboost', 
            'stacked_model': 'model_stacked_model'
        }
        self.artifact_cache_keys = {
            'feature_selector': 'artifact_feature_selector',
            'feature_scaler': 'artifact_feature_scaler',
            'feature_info': 'artifact_feature_info'
        }
    
    def load_model_to_redis(self, models_dir: Path, model_name: str) -> bool:
        """Load model from disk and store in Redis."""
        try:
            model_files = {
                'random_forest': 'random_forest.joblib',
                'xgboost': 'xgboost.joblib',
                'stacked_model': 'stacked_model.joblib'
            }
            
            model_file = models_dir / model_files.get(model_name, f"{model_name}.joblib")
            
            if not model_file.exists():
                raise FileNotFoundError(f"Model not found: {model_file}")
            
            # Set threading environment for safe loading
            old_omp = os.environ.get('OMP_NUM_THREADS')
            os.environ['OMP_NUM_THREADS'] = '1'
            
            try:
                model = joblib.load(model_file)
                logger.info(f"Loaded model: {model_file}")
            finally:
                if old_omp is not None:
                    os.environ['OMP_NUM_THREADS'] = old_omp
                else:
                    os.environ.pop('OMP_NUM_THREADS', None)
            
            # Store in Redis (24 hour timeout)
            cache_key = self.model_cache_keys[model_name]
            success = cache.set(cache_key, pickle.dumps(model), timeout=86400)
            
            if success:
                logger.info(f"Model {model_name} cached in Redis")
                return True
            else:
                logger.error(f"Failed to cache model {model_name} in Redis")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model {model_name} to Redis: {e}")
            return False
    
    def load_artifacts_to_redis(self, models_dir: Path) -> bool:
        """Load preprocessing artifacts to Redis."""
        try:
            success_count = 0
            
            # Load feature selector
            selector_path = models_dir / 'feature_selector.joblib'
            if selector_path.exists():
                selector = joblib.load(selector_path)
                cache_key = self.artifact_cache_keys['feature_selector']
                if cache.set(cache_key, pickle.dumps(selector), timeout=86400):
                    success_count += 1
                    logger.info("Feature selector cached in Redis")
            
            # Load enhanced preprocessing artifacts
            try:
                base_dir = models_dir.parent.parent / 'core_engine' / 'artifacts' / 'preprocessing_artifacts'
                from enhanced_feature_extraction import load_preprocessing_artifacts
                scaler, feature_info = load_preprocessing_artifacts(base_dir)
                
                if scaler is not None:
                    cache_key = self.artifact_cache_keys['feature_scaler']
                    if cache.set(cache_key, pickle.dumps(scaler), timeout=86400):
                        success_count += 1
                        logger.info("Feature scaler cached in Redis")
                
                if feature_info is not None:
                    cache_key = self.artifact_cache_keys['feature_info']
                    if cache.set(cache_key, pickle.dumps(feature_info), timeout=86400):
                        success_count += 1
                        logger.info("Feature info cached in Redis")
                        
            except Exception as e:
                logger.warning(f"Failed to load enhanced artifacts: {e}")
            
            logger.info(f"Loaded {success_count} artifacts to Redis")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error loading artifacts to Redis: {e}")
            return False
    
    def get_model_from_redis(self, model_name: str):
        """Get model from Redis cache."""
        try:
            cache_key = self.model_cache_keys.get(model_name)
            if not cache_key:
                logger.error(f"Unknown model name: {model_name}")
                return None
            
            model_data = cache.get(cache_key)
            if model_data is None:
                logger.warning(f"Model {model_name} not found in Redis cache")
                return None
            
            return pickle.loads(model_data)
            
        except Exception as e:
            logger.error(f"Error getting model {model_name} from Redis: {e}")
            return None
    
    def get_artifacts_from_redis(self):
        """Get all artifacts from Redis cache."""
        try:
            artifacts = {}
            
            for name, cache_key in self.artifact_cache_keys.items():
                data = cache.get(cache_key)
                if data is not None:
                    artifacts[name] = pickle.loads(data)
                else:
                    logger.warning(f"Artifact {name} not found in Redis cache")
            
            return artifacts
            
        except Exception as e:
            logger.error(f"Error getting artifacts from Redis: {e}")
            return {}
    
    def preload_all_models(self, models_dir: Path) -> dict:
        """Preload all models and artifacts to Redis."""
        results = {
            'models': {},
            'artifacts': False,
            'total_success': 0
        }
        
        logger.info(f"Preloading models from {models_dir}")
        
        # Load models
        for model_name in self.model_cache_keys.keys():
            success = self.load_model_to_redis(models_dir, model_name)
            results['models'][model_name] = success
            if success:
                results['total_success'] += 1
        
        # Load artifacts
        results['artifacts'] = self.load_artifacts_to_redis(models_dir)
        if results['artifacts']:
            results['total_success'] += 1
        
        logger.info(f"Preloading complete: {results['total_success']}/{len(self.model_cache_keys) + 1} items loaded")
        return results
    
    def clear_redis_cache(self) -> bool:
        """Clear all models and artifacts from Redis."""
        try:
            all_keys = list(self.model_cache_keys.values()) + list(self.artifact_cache_keys.values())
            deleted_count = 0
            
            for cache_key in all_keys:
                if cache.delete(cache_key):
                    deleted_count += 1
            
            logger.info(f"Cleared {deleted_count} items from Redis cache")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
            return False

# Global instance
redis_model_loader = RedisModelLoader()

def get_model_from_redis(model_name: str):
    """Convenience function to get model from Redis."""
    return redis_model_loader.get_model_from_redis(model_name)

def get_artifacts_from_redis():
    """Convenience function to get artifacts from Redis."""
    return redis_model_loader.get_artifacts_from_redis()
