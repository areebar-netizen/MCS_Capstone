# Generated migration for Recommendation model change
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('secondBrain_App', '0009_sessionsummary_session_date'),
    ]

    operations = [
        migrations.AddField(
            model_name='recommendation',
            name='session',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                to='secondBrain_App.sessionsummary'
            ),
        ),
        migrations.AlterField(
            model_name='recommendation',
            name='session_id',
            field=models.CharField(max_length=100, null=True, blank=True),
        ),
    ]
