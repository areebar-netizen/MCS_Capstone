# Generated migration for PreSessionCheckIn model

from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('secondBrain_App', '0011_remove_recommendation_session_id'),
    ]

    operations = [
        migrations.CreateModel(
            name='PreSessionCheckIn',
            fields=[
                ('check_in_id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('session_id', models.CharField(max_length=100)),
                ('subject_task', models.CharField(choices=[('Math', 'Math'), ('Reading', 'Reading'), ('Writing', 'Writing'), ('Coding', 'Coding'), ('Research', 'Research'), ('Studying', 'Studying'), ('Problem Solving', 'Problem Solving'), ('Creative Work', 'Creative Work'), ('Other', 'Other')], max_length=50)),
                ('task_difficulty', models.IntegerField(help_text='Task difficulty from 1-10')),
                ('estimated_length', models.CharField(choices=[('15-30m', '15-30 minutes'), ('30-60m', '30-60 minutes'), ('1-2h', '1-2 hours'), ('2h+', '2+ hours')], max_length=20)),
                ('assignment_deadline', models.DateTimeField(blank=True, help_text='Optional assignment deadline', null=True)),
                ('session_goal', models.TextField(help_text='Specific goal for this session')),
                ('energy_level', models.IntegerField(help_text='Energy level from 1-10')),
                ('mood_emoji', models.CharField(choices=[('Happy', '😊 Happy'), ('Calm', '😌 Calm'), ('Focused', '🎯 Focused'), ('Anxious', '😰 Anxious'), ('Tired', '😴 Tired'), ('Stressed', '😤 Stressed'), ('Excited', '🤩 Excited'), ('Neutral', '😐 Neutral')], max_length=50)),
                ('stress_level', models.IntegerField(help_text='Stress level from 1-10')),
                ('time_since_meal', models.CharField(choices=[('<1h', 'Less than 1 hour'), ('1-2h', '1-2 hours'), ('2-4h', '2-4 hours'), ('4h+', '4+ hours')], max_length=20)),
                ('caffeine_intake', models.CharField(choices=[('None', 'None'), ('1 cup', '1 cup'), ('2 cups', '2 cups'), ('3-5 cups', '3-5 cups')], max_length=20)),
                ('time_since_waking', models.CharField(choices=[('<1h', 'Less than 1 hour'), ('1-3h', '1-3 hours'), ('3-6h', '3-6 hours'), ('6h+', '6+ hours')], max_length=20)),
                ('physical_activity', models.CharField(choices=[('None', 'None'), ('Light', 'Light'), ('Moderate', 'Moderate'), ('Intense', 'Intense')], max_length=20)),
                ('current_noise', models.CharField(help_text='Current noise level/description', max_length=100)),
                ('lighting_conditions', models.CharField(help_text='Lighting conditions', max_length=100)),
                ('study_method', models.CharField(help_text='Study method/approach', max_length=100)),
                ('current_location', models.CharField(help_text='Current study location', max_length=100)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='secondBrain_App.userprofile')),
            ],
            options={
                'db_table': 'pre_session_check_in',
            },
        ),
    ]
