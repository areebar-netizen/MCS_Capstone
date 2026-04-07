from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('onboarding/', views.onboarding_view, name='onboarding'),
    path('send-otp/', views.send_otp, name='send_otp'),
    path('verify-otp/', views.verify_otp, name='verify_otp'),
    path('', views.email_entry, name='home'),  # Default to email entry
    path('api/predict/', views.prediction_view, name='prediction_view'),
    path('upload_csv/', views.upload_csv_view, name='upload_csv_views'),
    path('start_eeg/', views.start_live_eeg_view, name='start_eeg'),
    path('stop_eeg/', views.stop_live_eeg_view, name='stop_eeg'),
    path('eeg_status/', views.eeg_task_status_view, name='eeg_status'),
    
    # New real-time endpoints
    path('start_realtime_eeg/', views.start_realtime_eeg_view, name='start_realtime_eeg'),
    path('stop_realtime_eeg/', views.stop_realtime_eeg_view, name='stop_realtime_eeg'),
    path('realtime_eeg_status/', views.get_realtime_eeg_status_view, name='realtime_eeg_status'),
    path('end_session/', views.end_session, name='end_session'),
]
