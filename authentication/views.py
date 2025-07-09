from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.core.mail import send_mail
from django.conf import settings
from .forms import CustomUserCreationForm
from .decorators import role_required
from django.contrib.auth.decorators import login_required
import logging
import csv
import io
from django.contrib import messages
from .models import SoilMoistureRecord
from datetime import datetime
from .ml_model import predict_soil_moisture
import os
import requests
from .ml_model import train_model

weather_api = os.getenv("OPENWEATHER_API_KEY")

logger = logging.getLogger(__name__)

def home(request):
    return render(request, 'home.html')

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user, backend='django.contrib.auth.backends.ModelBackend')
            # Send email notification
            try:
                send_mail(
                    subject='Welcome to Farm System!',
                    message=f'Hi {user.username},\n\nYour account has been created successfully!',
                    from_email=settings.EMAIL_HOST_USER,
                    recipient_list=[user.email],
                    fail_silently=False,
                )
                logger.info(f"Welcome email sent to {user.email}")
            except Exception as e:
                logger.error(f"Failed to send email to {user.email}: {e}")
            if user.role == 'admin':
                return redirect('admin_dashboard')
            elif user.role == 'farmer':
                return redirect('farmer_dashboard')
            elif user.role == 'technician':
                return redirect('technician_dashboard')
            else:
                return redirect('home')
        else:
            logger.error(f"Form validation failed: {form.errors}")
    else:
     
        form = CustomUserCreationForm()
    return render(request, 'accounts/register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        logger.debug(f"Attempting login with email: {email}")
        user = authenticate(request, email=email, password=password)
        if user is not None:
            logger.debug(f"User authenticated: {user.email}")
            login(request, user)
            if user.role == 'admin':
                return redirect('admin_dashboard')
            elif user.role == 'farmer':
                return redirect('farmer_dashboard')
            elif user.role == 'technician':
                return redirect('technician_dashboard')
            else:
                return redirect('home')
        else:
            logger.error(f"Authentication failed for email: {email}")
            return render(request, 'accounts/login.html', {'error': 'Invalid credentials'})
    return render(request, 'accounts/login.html')

def user_logout(request):
    logout(request)
    return redirect('login')

@role_required('admin')
def admin_dashboard(request):
    # Get filter parameters
    location = request.GET.get('location', '')
    start_date = request.GET.get('start_date', '')
    end_date = request.GET.get('end_date', '')

    # Query soil moisture records
    records = SoilMoistureRecord.objects.all().order_by('-timestamp')

    # Apply filters
    if location:
        records = records.filter(location=location)
    
    if start_date:
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            records = records.filter(timestamp__gte=start_date)
        except ValueError:
            messages.error(request, 'Invalid start date format.')
    
    if end_date:
        try:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            records = records.filter(timestamp__lte=end_date)
        except ValueError:
            messages.error(request, 'Invalid end date format.')

    # Get unique locations for dropdown
    locations = SoilMoistureRecord.objects.values_list('location', flat=True).distinct()

    context = {
        'user': request.user,
        'records': records,
        'locations': locations,
        'selected_location': location,
        'start_date': start_date,
        'end_date': end_date,
    }
    return render(request, 'dashboards/admin_dashboard.html', context)

@role_required('farmer')
def farmer_dashboard(request):
    return render(request, 'dashboards/farmer_dashboard.html', {'user': request.user})

@role_required('technician')
def technician_dashboard(request):
    return render(request, 'dashboards/technician_dashboard.html', {'user': request.user})

@login_required
def profile(request):
    if request.method == 'POST':
        user = request.user
        user.email = request.POST.get('email').lower()
        user.username = request.POST.get('username')
        user.save()
        return redirect('profile')
    return render(request, 'accounts/profile.html', {'user': request.user})



# Uploading the csv file to the database
def upload_csv(request):
    if request.method == 'POST':
        csv_file = request.FILES.get('csv-upload')
        if not csv_file:
            messages.error(request, 'No file uploaded.')
            return redirect('admin_dashboard')

        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'Please upload a valid CSV file.')
            return redirect('admin_dashboard')

        try:
            # Read CSV file
            csv_data = csv_file.read().decode('utf-8')
            io_string = io.StringIO(csv_data)
            reader = csv.DictReader(io_string)

            # Validate required columns
            required_columns = [
                'record_id', 'sensor_id', 'location', 'soil_moisture_percent',
                'temperature_celsius', 'humidity_percent', 'timestamp',
                'status', 'battery_voltage', 'irrigation_action'
            ]
            if not all(col in reader.fieldnames for col in required_columns):
                messages.error(request, 'CSV file is missing required columns.')
                return redirect('admin_dashboard')

            # Process each row
            for row in reader:
                try:
                    # Convert timestamp to datetime
                    timestamp = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')

                    # Create or update record
                    SoilMoistureRecord.objects.update_or_create(
                        record_id=int(row['record_id']),
                        defaults={
                            'sensor_id': row['sensor_id'],
                            'location': row['location'],
                            'soil_moisture_percent': float(row['soil_moisture_percent']),
                            'temperature_celsius': float(row['temperature_celsius']),
                            'humidity_percent': float(row['humidity_percent']),
                            'timestamp': timestamp,
                            'status': row['status'],
                            'battery_voltage': float(row['battery_voltage']),
                            'irrigation_action': row['irrigation_action']
                        }
                    )
                except (ValueError, KeyError) as e:
                    messages.warning(request, f"Error processing row {row['record_id']}: {str(e)}")
                    continue

            messages.success(request, 'CSV data uploaded successfully!')
            return redirect('admin_dashboard')

        except Exception as e:
            messages.error(request, f'Error processing CSV file: {str(e)}')
            return redirect('admin_dashboard')

    return render(request, 'admin_dashboard')


# Upload a ml model
@login_required
@role_required('admin')
def upload_model(request):
    if request.method == 'POST':
        model_file = request.FILES.get('ml-model-upload')
        if not model_file:
            messages.error(request, 'No file uploaded.')
            return redirect('admin_dashboard')
        
        if not model_file.name.endswith(('.pkl', '.h5')):
            messages.error(request, 'Please upload a valid model file (.pkl or .h5).')
            return redirect('admin_dashboard')
        
        try:
            # Save the model file
            model_path = os.path.join(settings.BASE_DIR, 'models', model_file.name)
            with open(model_path, 'wb') as f:
                for chunk in model_file.chunks():
                    f.write(chunk)
            
            # Optionally retrain the model
            if request.POST.get('retrain'):
                train_model()
            
            messages.success(request, 'Model uploaded successfully!')
            return redirect('admin_dashboard')
        
        except Exception as e:
            messages.error(request, f'Error uploading model: {str(e)}')
            return redirect('admin_dashboard')
    

    return render(request, 'dashboards/admin_dashboard.html')



# Weather forecat api
def get_weather_forecast(location):
    if not weather_api:
        logger.error("OpenWeatherMap API key is not set.")
        return {'precipitation': 0.0}  # Fallback value

    try:
        # OpenWeatherMap API endpoint for 5-day forecast (3-hour intervals)
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={weather_api}&units=metric"
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise an exception for bad status codes

        data = response.json()
        # Extract precipitation for the next 24 hours (first 8 intervals of 3 hours)
        precipitation = 0.0
        for forecast in data['list'][:8]:  # Next 24 hours (8 * 3-hour intervals)
            precipitation += forecast.get('rain', {}).get('3h', 0.0)  # Rainfall in mm for 3 hours

        return {'precipitation': precipitation}
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch weather forecast for {location}: {str(e)}")
        return {'precipitation': 0.0}  # Fallback value


# Making soil moisture predictions
# views.py

@login_required
@role_required('admin')
def predict_moisture(request):
    if request.method == 'POST':
        try:
            location = request.POST.get('location')
            current_moisture = float(request.POST.get('soil_moisture_percent'))
            temperature = float(request.POST.get('temperature'))
            humidity = float(request.POST.get('humidity'))
            
            weather_forecast = get_weather_forecast(location)
            
            # Make prediction
            prediction = predict_soil_moisture(
                location, current_moisture, temperature, humidity, weather_forecast
            )
            
            # Store prediction in context
            context = {
                'prediction': round(prediction, 2),
                'location': location,
                'current_moisture': current_moisture,
                'temperature': temperature,
                'humidity': humidity,
            }
            return render(request, 'dashboards/prediction_result.html', context)
        
        except ValueError as e:
            messages.error(request, f"Invalid input: {str(e)}")
            return redirect('admin_dashboard')
    
    return render(request, 'dashboards/admin_dashboard.html')