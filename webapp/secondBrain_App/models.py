from django.db import models
import uuid
from datetime import datetime, timedelta
import random
from django.utils import timezone

class EmailOTP(models.Model):
    email = models.EmailField()
    otp_code = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def is_valid(self):
        """Check if OTP is still valid (5 minutes expiry)"""
        now = timezone.now()
        expiry_time = self.created_at + timedelta(minutes=5)
        return now < expiry_time
    
    class Meta:
        db_table = 'email_otp'

class UserProfile(models.Model):
    # Use Email as the primary unique identifier for validation
    email = models.EmailField(primary_key=True)
    name = models.CharField(max_length=255)
    age = models.IntegerField()
    academic_level = models.CharField(max_length=100)
    
    # Section 2: Rhythms
    alert_time = models.CharField(max_length=100)
    sleep_hours = models.FloatField()
    sleep_quality = models.CharField(max_length=100)
    
    # Section 3: Caffeine
    consumes_caffeine = models.BooleanField(default=False)
    caffeine_types = models.TextField(blank=True, help_text="Comma separated list")
    caffeine_servings = models.IntegerField(default=0)
    caffeine_timing = models.CharField(max_length=100, blank=True)
    
    # Section 4: Styles
    learning_style = models.CharField(max_length=100)
    study_subjects = models.TextField(blank=True)
    
    # Section 5: Habits
    session_length = models.CharField(max_length=100)
    takes_breaks = models.CharField(max_length=100)
    study_time_of_day = models.CharField(max_length=100)
    procrastination_level = models.IntegerField()
    
    # Section 6 & 7: Environment & Distractions
    study_location = models.TextField(blank=True)
    sound_environment = models.CharField(max_length=100)
    lighting_preference = models.CharField(max_length=100)
    phone_location = models.CharField(max_length=100)
    distractions = models.TextField(blank=True)
    
    # Section 8 & 9: Lifestyle & Health
    exercise_frequency = models.CharField(max_length=100)
    eating_timing = models.CharField(max_length=100)
    health_conditions = models.TextField(blank=True)
    
    # Section 10: Goals
    main_goals = models.TextField(blank=True)
    study_effectiveness = models.IntegerField()

    class Meta:
        db_table = 'user_profile'

class Recommendation(models.Model):
    recommendation_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    session_id = models.CharField(max_length=100)
    inference_id = models.CharField(max_length=100)
    recommendation_category = models.CharField(max_length=50)
    stimulus_name = models.CharField(max_length=100)
    trigger_reason = models.CharField(max_length=100)
    message = models.TextField()
    action_started_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'recommendation'