from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('calendar/', views.calendar_view, name='calendar'),
    path('onboarding/', views.onboarding_view, name='onboarding'),
    path('send-otp/', views.send_otp, name='send_otp'),
    path('verify-otp/', views.verify_otp, name='verify_otp'),
    path('', views.email_entry, name='home'),  # Default to email entry
    path('api/predict/', views.prediction_view, name='prediction_view'),
    path('api/calendar-data/', views.calendar_api_data, name='calendar_api_data'),
    path('api/study-time-data/', views.study_time_api_data, name='study_time_api_data'),
    path('upload_csv/', views.upload_csv_view, name='upload_csv_views'),
    path('start_eeg/', views.start_live_eeg_view, name='start_eeg'),
    path('stop_eeg/', views.stop_live_eeg_view, name='stop_eeg'),
    path('eeg_status/', views.eeg_task_status_view, name='eeg_status'),
    
    # New real-time endpoints
    path('start_realtime_eeg/', views.start_realtime_eeg_view, name='start_realtime_eeg'),
    path('stop_realtime_eeg/', views.stop_realtime_eeg_view, name='stop_realtime_eeg'),
    path('realtime_eeg_status/', views.get_realtime_eeg_status_view, name='realtime_eeg_status'),
    path('get-latest-eeg-state/', views.get_latest_eeg_state_view, name='get_latest_eeg_state'),
    path('test-cache/', views.test_cache_view, name='test_cache'),
    path('end_session/', views.end_session, name='end_session'),
    path('recommendation/', views.recommendation_view, name='recommendation'),
    path('focus-history/', views.focus_track_history, name='focus_track_history'),
    path('api/presession-checkin/', views.presession_checkin_view, name='presession_checkin'),
    path('api/recommendation-feedback/', views.recommendation_feedback_view, name='recommendation_feedback'),
]
