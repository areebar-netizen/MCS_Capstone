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

class Prediction(models.Model):
    prediction_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    session_id = models.CharField(max_length=100)
    predicted_label = models.CharField(max_length=50)
    confidence = models.FloatField()
    n_windows = models.IntegerField()
    total_seconds = models.FloatField()
    relaxed_seconds = models.FloatField()
    neutral_seconds = models.FloatField()
    concentrating_seconds = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'prediction'

class SessionSummary(models.Model):
    session_id = models.CharField(max_length=100, unique=True)
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    task_id = models.CharField(max_length=100, null=True, blank=True)
    csv_file_path = models.CharField(max_length=500)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField(null=True, blank=True)
    total_duration_seconds = models.FloatField()
    average_focus_score = models.FloatField()
    peak_focus_score = models.FloatField()
    relaxed_seconds = models.FloatField()
    neutral_seconds = models.FloatField()
    concentrating_seconds = models.FloatField()
    data_points_count = models.IntegerField()
    
    # Advanced Analytics Fields
    longest_focus_streak = models.FloatField(default=0.0, help_text="Longest continuous focus period in seconds")
    focus_latency = models.FloatField(default=0.0, help_text="Time to reach initial focus state in seconds")
    state_switch_count = models.IntegerField(default=0, help_text="Number of times focus state changed")
    avg_confidence = models.FloatField(default=0.0, help_text="Mean confidence score across all windows")
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'session_summary'

class PreSessionCheckIn(models.Model):
    """Pre-session check-in data for detailed session context"""
    check_in_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    session_id = models.CharField(max_length=100)
    
    # Task and Study Context
    subject_task = models.CharField(max_length=50, choices=[
        ('Math', 'Math'),
        ('Reading', 'Reading'),
        ('Writing', 'Writing'),
        ('Coding', 'Coding'),
        ('Research', 'Research'),
        ('Studying', 'Studying'),
        ('Problem Solving', 'Problem Solving'),
        ('Creative Work', 'Creative Work'),
        ('Other', 'Other'),
    ])
    task_difficulty = models.IntegerField(help_text="Task difficulty from 1-10")
    estimated_length = models.CharField(max_length=20, choices=[
        ('15-30m', '15-30 minutes'),
        ('30-60m', '30-60 minutes'),
        ('1-2h', '1-2 hours'),
        ('2h+', '2+ hours'),
    ])
    assignment_deadline = models.DateTimeField(null=True, blank=True, help_text="Optional assignment deadline")
    session_goal = models.TextField(help_text="Specific goal for this session")
    
    # Personal State
    energy_level = models.IntegerField(help_text="Energy level from 1-10")
    mood_emoji = models.CharField(max_length=50, choices=[
        ('Happy', '😊 Happy'),
        ('Calm', '😌 Calm'),
        ('Focused', '🎯 Focused'),
        ('Anxious', '😰 Anxious'),
        ('Tired', '😴 Tired'),
        ('Stressed', '😤 Stressed'),
        ('Excited', '🤩 Excited'),
        ('Neutral', '😐 Neutral'),
    ])
    stress_level = models.IntegerField(help_text="Stress level from 1-10")
    
    # Physical and Environmental Context
    time_since_meal = models.CharField(max_length=20, choices=[
        ('<1h', 'Less than 1 hour'),
        ('1-2h', '1-2 hours'),
        ('2-4h', '2-4 hours'),
        ('4h+', '4+ hours'),
    ])
    caffeine_intake = models.CharField(max_length=20, choices=[
        ('None', 'None'),
        ('1 cup', '1 cup'),
        ('2 cups', '2 cups'),
        ('3-5 cups', '3-5 cups'),
    ])
    time_since_waking = models.CharField(max_length=20, choices=[
        ('<1h', 'Less than 1 hour'),
        ('1-3h', '1-3 hours'),
        ('3-6h', '3-6 hours'),
        ('6h+', '6+ hours'),
    ])
    physical_activity = models.CharField(max_length=20, choices=[
        ('None', 'None'),
        ('Light', 'Light'),
        ('Moderate', 'Moderate'),
        ('Intense', 'Intense'),
    ])
    
    # New Context Fields
    current_noise = models.CharField(max_length=100, help_text="Current noise level/description")
    lighting_conditions = models.CharField(max_length=100, help_text="Lighting conditions")
    study_method = models.CharField(max_length=100, help_text="Study method/approach")
    current_location = models.CharField(max_length=100, help_text="Current study location")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'pre_session_check_in'

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