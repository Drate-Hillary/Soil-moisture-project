# Generated by Django 4.2 on 2025-07-10 13:56

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('authentication', '0003_soilmoistureprediction'),
    ]

    operations = [
        migrations.CreateModel(
            name='TechnicianLocationAssignment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('location', models.CharField(max_length=100)),
                ('technician', models.ForeignKey(limit_choices_to={'role': 'technician'}, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'technician_location_assignments',
                'unique_together': {('technician', 'location')},
            },
        ),
    ]
