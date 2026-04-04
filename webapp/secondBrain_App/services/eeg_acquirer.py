from pathlib import Path
import sys
import time
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
from Stream import Stream
from pylsl import StreamInlet, resolve_byprop


class EEGAcquirer:
    """Handles LSL connection and buffering."""
    def __init__(self, max_seconds: int = 30, sfreq: int = 256):
        self.inlet = None
        self.buffer = deque()
        self.max_len = max_seconds * sfreq
        self._running = False
        self._thread = None
        
        # For raw data saving
        self._raw_fh = None
        self._raw_writer = None

    def connect(self, timeout: float = 5.0):
        if StreamInlet is None:
            raise RuntimeError("pylsl not installed.")
        
        # Use Stream.py logic if available, otherwise direct resolve
        if Stream:
            s = Stream()
            self.inlet = s.connect_to_eeg_stream()
        else:
            stream = resolve_byprop('type', 'EEG', timeout=timeout)
            if not stream:
                raise RuntimeError('No EEG stream found')
            self.inlet = StreamInlet(stream[0])
        return True

    def start_saving_raw(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        need_header = not os.path.exists(path) or os.path.getsize(path) == 0
        
        fh = open(path, mode='a', newline='')
        self._raw_writer = csv.writer(fh)
        if need_header:
            # Default Muse header
            header = ['timestamps', 'TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
            self._raw_writer.writerow(header)
        
        self._raw_fh = fh
        print(f"Logging raw data to {path}")

    def start(self):
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
                
                # Save raw if enabled
                if self._raw_writer:
                    try:
                        self._raw_writer.writerow(row)
                        self._raw_fh.flush()
                    except Exception:
                        pass

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.1)
        if self._raw_fh:
            self._raw_fh.close()

    def get_buffer_copy(self):
        return list(self.buffer)