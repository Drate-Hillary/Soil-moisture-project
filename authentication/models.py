# models.py
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone

class CustomUser(AbstractUser):
    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('technician', 'Technician'),
        ('farmer', 'Farmer'),
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='farmer')

    def __str__(self):
        return self.email

class TechnicianLocationAssignment(models.Model):
    technician = models.ForeignKey(CustomUser, on_delete=models.CASCADE, limit_choices_to={'role': 'technician'})
    location = models.CharField(max_length=100)

    class Meta:
        db_table = 'technician_location_assignments'
        unique_together = ('technician', 'location')  # Ensure a technician can't be assigned to the same location twice

    def __str__(self):
        return f"{self.technician.get_full_name() or self.technician.username} - {self.location}"

class SoilMoistureRecord(models.Model):
    record_id = models.IntegerField(unique=True)
    sensor_id = models.CharField(max_length=50)
    location = models.CharField(max_length=100)
    soil_moisture_percent = models.FloatField()
    temperature_celsius = models.FloatField()
    humidity_percent = models.FloatField()
    timestamp = models.DateTimeField()
    status = models.CharField(max_length=50)
    battery_voltage = models.FloatField()
    irrigation_action = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.record_id} - {self.location} - {self.timestamp}"

    class Meta:
        db_table = 'soil_moisture_records'

class SoilMoisturePrediction(models.Model):
    location = models.CharField(max_length=100)
    predicted_moisture = models.FloatField()
    current_moisture = models.FloatField(default=0.0)
    temperature = models.FloatField(default=0.0)   
    humidity = models.FloatField(default=0.0)          
    precipitation = models.FloatField(default=0.0)
    prediction_for = models.DateTimeField(default=timezone.now)
    status = models.CharField(max_length=100, default="unknown")
    
    def __str__(self):
        return f"{self.location} - {self.predicted_moisture}%"



class TechnicianSoilMoisturePrediction(models.Model):
    """
    Model to store soil moisture predictions made by technicians.
    """
    location = models.CharField(max_length=100, help_text="Location of the prediction")
    timestamp = models.DateTimeField(default=timezone.now, help_text="Time of prediction")
    current_moisture = models.FloatField(help_text="Current soil moisture percentage")
    temperature = models.FloatField(help_text="Temperature in Celsius")
    humidity = models.FloatField(help_text="Humidity percentage")
    precipitation = models.FloatField(default=0.0, help_text="Precipitation in mm")
    predicted_category = models.CharField(max_length=20, help_text="Predicted moisture category")
    predicted_moisture_value = models.FloatField(help_text="Predicted moisture value (%)")
    confidence = models.FloatField(null=True, blank=True, help_text="Prediction confidence score")
    created_at = models.DateTimeField(auto_now_add=True, help_text="Record creation time")
    updated_at = models.DateTimeField(auto_now=True, help_text="Record last updated time")

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['location', 'timestamp']),
        ]

    def __str__(self):
        return f"{self.location} - {self.timestamp.strftime('%Y-%m-%d %H:%M')} - {self.predicted_category}"