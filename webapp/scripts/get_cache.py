import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE','secondBrain.settings')
django.setup()
from django.core.cache import cache
keys = ['session_final_result_areebaarashid31@gmail.com','wave_averages_areebaarashid31@gmail.com','live_eeg_stream_areebaarashid31@gmail.com']
for k in keys:
    print(k, '->', cache.get(k))
