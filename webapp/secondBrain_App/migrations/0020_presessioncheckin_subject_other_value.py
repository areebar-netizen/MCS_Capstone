# Generated migration for subject_other_value field

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('secondBrain_App', '0019_recommendation_optimal_study_environment_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='presessioncheckin',
            name='subject_other_value',
            field=models.CharField(blank=True, help_text="Custom subject value when 'Other' is selected", max_length=100, null=True),
        ),
    ]
