import threading
import time
from uuid import uuid4
from .eeg_acquirer import EEGAcquirer
from django.conf import settings
from secondBrain_App.models import UserProfile, Recommendation
from secondBrain_App.services.prediction_service import PredictionService
from pathlib import Path

MODEL_SERVICE = PredictionService(models_dir=Path(settings.BASE_DIR.parent)/ 'models_out', model_name='xgboost')

class EEGService:
    """ Manages Live EEG Streaming and buffering as a background process"""

    def __init__(self, max_seconds: int = 30, sfreq: int = 256):
        self.acquirer = EEGAcquirer(max_seconds, sfreq)
        self._thread = None
        self.running = False
        self.session_id = None
        self.user_email = None
        self.lock = threading.Lock()

    def start(self, user_email: str):
        with self.lock:
            if self.running:
                return {
                    'ok': False,
                    'message': 'EEG streaming already running'
                }
            self.user_email = user_email
            self.session_id = str(uuid4())
            self.acquirer.connect()
            self.acquirer.start()
            self.running = True

            return {
                'ok': True,
                'message': 'EEG streaming started',
                'session_id': self.session_id
            }
    
    def stop(self):
        with self.lock:
            if not self.running:
                return {
                    'ok': False,
                    'message': 'EEG stream not running'
                }
        rows = self.acquirer.get_buffer_copy()
        self.acquirer.stop()

        
        session_id = self.session_id

        self.acquirer = None
        self.user_email = None
        self.session_id = None
        self.running = False

        def status(self):
            with self.lock:
                return {
                    'ok': True,
                    'running': self.running,
                    'session_id': self.session_id
                }

        return {
            'ok': True,
            'message': 'EEG streaming stopped',
            'session_id': session_id,
            'rows': rows
        }
        


       


