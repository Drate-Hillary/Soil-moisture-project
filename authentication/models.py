from django.contrib.auth.models import AbstractUser
from django.db import models
from django.contrib.auth.models import User

class CustomUser(AbstractUser):
    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('technician', 'Technician'),
        ('farmer', 'Farmer'),
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='farmer')

    def __str__(self):
        return self.email
    


# uploading a csv file into the database
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
    timestamp = models.DateTimeField(auto_now_add=True)
    location = models.CharField(max_length=100)
    predicted_moisture = models.FloatField()
    input_moisture = models.FloatField()
    input_temperature = models.FloatField()
    input_humidity = models.FloatField()

    def __str__(self):
        return f"{self.location} - {self.predicted_moisture}%"