from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('onboarding/', views.onboarding_view, name='onboarding'),
    path('', views.onboarding_view, name='home'),  # Default to onboarding
]
