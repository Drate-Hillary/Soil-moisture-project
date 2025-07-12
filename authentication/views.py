from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.core.mail import send_mail
from django.conf import settings
from .forms import CustomUserCreationForm
from .decorators import role_required, roles_required
from django.contrib.auth.decorators import login_required
import logging
import csv
import io
from django.contrib import messages
from .models import SoilMoistureRecord, SoilMoisturePrediction
from datetime import datetime
from .ml_model import predict_soil_status, train_model, get_model_metrics, get_irrigation_schedule_recommendation
import os
import requests
from .ml_model import train_model
from dotenv import load_dotenv
from django.db.models.functions import TruncDay
import json
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from openpyxl import Workbook
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from openpyxl import Workbook
from django.db.models import Avg, Min, Max
from datetime import timedelta
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import CustomUser, TechnicianLocationAssignment
from urllib.parse import urlencode 
from django.urls import reverse
from django.http import HttpResponseRedirect

from .models import SoilMoistureRecord



load_dotenv()
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
                # Get the technician's assigned farms
                assigned_farms = TechnicianLocationAssignment.objects.filter(technician=user)
                assigned_locations = assigned_farms.values_list('location', flat=True).distinct()
                if assigned_locations:
                    query_params = urlencode({'location': assigned_locations[0]})
                    base_url = reverse('technician_dashboard')
                    url_with_params = f'{base_url}?{query_params}'
                    return HttpResponseRedirect(url_with_params)
                else:
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
    show_all = request.GET.get('show_all', 'false').lower() == 'true'

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
    
    # Set default location - use first location if available, otherwise a hardcoded default
    default_location = locations[0] if locations else "Nairobi"
    
    # Get weather data
    current_weather = None
    if default_location:
        current_weather = get_weather_forecast(default_location)

    # Calculate average soil moisture
    average_moisture = records.aggregate(Avg('soil_moisture_percent'))['soil_moisture_percent__avg']
    average_moisture = round(average_moisture, 2) if average_moisture is not None else None

    # Calculate average temperature
    average_temperature = records.aggregate(Avg('temperature_celsius'))['temperature_celsius__avg']
    average_temperature = round(average_temperature, 2) if average_temperature is not None else None

    # Calculate average humidity
    average_humidity = records.aggregate(Avg('humidity_percent'))['humidity_percent__avg']
    average_humidity = round(average_humidity, 2) if average_humidity is not None else None

    # Calculate daily averages for moisture trends chart
    daily_averages = (
        records.annotate(date=TruncDay('timestamp'))
        .values('date')
        .annotate(avg_moisture=Avg('soil_moisture_percent'))
        .order_by('date')
    )
    
    chart_data = {
        'labels': [record['date'].strftime('%Y-%m-%d') for record in daily_averages],
        'data': [round(record['avg_moisture'], 2) for record in daily_averages],
    }

    model_metrics = get_model_metrics()

    technicians = CustomUser.objects.filter(role='technician').select_related()
    farms = TechnicianLocationAssignment.objects.all()
    
    # Limit records to 10 unless show_all is true
    displayed_records = records[:10] if not show_all else records

    context = {
        'user': request.user,
        'records': displayed_records,
        'total_records': records.count(),
        'show_all': show_all,
        'locations': locations,
        'selected_location': location,
        'start_date': start_date,
        'end_date': end_date,
        'model_metrics': model_metrics,
        'average_moisture': average_moisture,
        'average_temperature': average_temperature,
        'average_humidity': average_humidity,
        'chart_data': json.dumps(chart_data),
        'current_weather': current_weather,
        'weather_location': default_location,
        'technicians': technicians,
        'farms': farms,
    }
    return render(request, 'dashboards/admin_dashboard.html', context)


@role_required('farmer')
def farmer_dashboard(request):
    if request.method == 'POST':
        try:
            # Extract form data
            soil_moisture = float(request.POST.get('soil-moisture'))
            temperature = float(request.POST.get('temperature'))
            humidity = float(request.POST.get('humidity'))

            # Get unique locations for the dropdown
            locations = SoilMoistureRecord.objects.values_list('location', flat=True).distinct()

            # Query soil moisture records
            records = SoilMoistureRecord.objects.all().order_by('-timestamp')

            # Apply filters (corrected: filter by a single location, not multiple)
            location = request.POST.get('location', '')  # Get location from POST if available
            if location:
                records = records.filter(location=location)

            # Calculate daily averages for moisture trends chart
            daily_averages = (
                records.annotate(date=TruncDay('timestamp'))
                .values('date')
                .annotate(avg_moisture=Avg('soil_moisture_percent'))
                .order_by('date')
            )
    
            chart_data = {
                'labels': [record['date'].strftime('%Y-%m-%d') for record in daily_averages],
                'data': [round(record['avg_moisture'], 2) for record in daily_averages],
            }

            # Make prediction using the ML model
            prediction_result = predict_soil_status(
                soil_moisture=soil_moisture,
                temperature=temperature,
                humidity=humidity
            )

            # Get irrigation schedule recommendation
            irrigation_schedule = get_irrigation_schedule_recommendation(
                soil_status=prediction_result['status']
            )

            # Store prediction in the database
            SoilMoisturePrediction.objects.create(
                location='default',  # Use a default location since location is removed
                predicted_moisture=soil_moisture,
                current_moisture=soil_moisture,
                temperature=temperature,
                humidity=humidity,
                precipitation=0,  # Set precipitation to 0 since weather API is removed
                prediction_for=datetime.now() + timedelta(hours=24),
                status=prediction_result['status']
            )

            # Prepare context for rendering results
            context = {
                'user': request.user,
                'prediction': {
                    'status': prediction_result['status'],
                    'irrigation_recommendation': prediction_result['irrigation_recommendation'],
                    'confidence': prediction_result['confidence'],
                    'method': prediction_result['method'],
                    'input_values': prediction_result['input_values'],
                    'schedule': irrigation_schedule
                },
                'locations': locations,
                'selected_location': location,
                'chart_data': json.dumps(chart_data),
            }

            return render(request, 'dashboards/farmer_dashboard.html', context)

        except ValueError as e:
            messages.error(request, f"Invalid input: {str(e)}")
            logger.error(f"Input validation error for farmer {request.user.email}: {str(e)}")
            return redirect('farmer_dashboard')
        except Exception as e:
            messages.error(request, f"Error processing prediction: {str(e)}")
            logger.error(f"Prediction error for farmer {request.user.email}: {str(e)}")
            return redirect('farmer_dashboard')

    # GET request: Render dashboard with locations
    locations = SoilMoistureRecord.objects.values_list('location', flat=True).distinct()
    # Set default location - use first location if available, otherwise a hardcoded default
    default_location = locations[0] if locations else "Nairobi"
    
    # Get weather data
    current_weather = None
    if default_location:
        current_weather = get_weather_forecast(default_location)
    records = SoilMoistureRecord.objects.all().order_by('-timestamp')
    if default_location:
        records = records.filter(location=default_location)
    daily_averages = (
        records.annotate(date=TruncDay('timestamp'))
        .values('date')
        .annotate(avg_moisture=Avg('soil_moisture_percent'))
        .order_by('date')
    )
    chart_data = {
        'labels': [record['date'].strftime('%Y-%m-%d') for record in daily_averages],
        'data': [round(record['avg_moisture'], 2) for record in daily_averages],
    }
    context = {
        'user': request.user,
        'locations': locations,
        'selected_location': default_location,
        'chart_data': json.dumps(chart_data),
        'current_weather': current_weather,
        'weather_location': default_location,
    }
    return render(request, 'dashboards/farmer_dashboard.html', context)


@login_required
@role_required('technician')
def technician_dashboard(request):
    # Get the logged-in technician
    technician = request.user

    # Get assigned locations
    assigned_locations = list(TechnicianLocationAssignment.objects.filter(
        technician=technician
    ).values_list('location', flat=True).distinct())

    if not assigned_locations:
        messages.warning(request, "You are not assigned to any locations. Please contact your administrator.")
        return render(request, 'dashboards/technician_dashboard.html', {
            'user': request.user,
            'records': [],
            'recent_records': [],
            'locations': [],
            'chart_data': json.dumps({'labels': [], 'data': []}),
            'average_moisture': None,
            'average_temperature': None,
            'average_humidity': None,
            'active_sensors': 0,
            'total_sensors': 0,
            'current_weather': None,
            'weather_location': None,
            'total_records': 0,
            'last_update': None,
            'last_upload': None,
        })

    # Get filter parameters
    location = request.GET.get('location', '')
    start_date = request.GET.get('start_date', '')
    end_date = request.GET.get('end_date', '')
    show_all = request.GET.get('show_all', 'false').lower() == 'true'

    # If no location is provided, default to the first assigned location
    if not location and assigned_locations:
        location = assigned_locations[0]

    # Query soil moisture records for assigned locations
    records = SoilMoistureRecord.objects.filter(location__in=assigned_locations).order_by('-timestamp')

    # Apply filters
    if location and location in assigned_locations:
        records = records.filter(location=location)
    
    if start_date:
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            records = records.filter(timestamp__gte=start_date_obj)
        except ValueError:
            messages.error(request, 'Invalid start date format.')
    
    if end_date:
        try:
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            records = records.filter(timestamp__lte=end_date_obj)
        except ValueError:
            messages.error(request, 'Invalid end date format.')

    # Get recent records for data ingestion section (last 10 records)
    recent_records = records[:10]

    # Get unique locations for dropdown (only assigned locations)
    locations = assigned_locations

    # Set default location for weather and other data
    default_location = location if location in assigned_locations else locations[0] if locations else None
    
    # Get weather data for default location
    current_weather = None
    if default_location:
        current_weather = get_weather_forecast(default_location)

    # Calculate average soil moisture
    average_moisture = records.aggregate(Avg('soil_moisture_percent'))['soil_moisture_percent__avg']
    average_moisture = round(average_moisture, 2) if average_moisture is not None else None

    # Calculate average temperature
    average_temperature = records.aggregate(Avg('temperature_celsius'))['temperature_celsius__avg']
    average_temperature = round(average_temperature, 2) if average_temperature is not None else None

    # Calculate average humidity
    average_humidity = records.aggregate(Avg('humidity_percent'))['humidity_percent__avg']
    average_humidity = round(average_humidity, 2) if average_humidity is not None else None

    # Calculate sensor statistics
    total_sensors = records.values('sensor_id').distinct().count()
    active_sensors = records.filter(status='active').values('sensor_id').distinct().count()

    # Get database metrics
    total_records = records.count()
    last_update = records.first().timestamp if records.exists() else None

    # Calculate daily averages for moisture trends chart
    daily_averages = (
        records.annotate(date=TruncDay('timestamp'))
        .values('date')
        .annotate(avg_moisture=Avg('soil_moisture_percent'))
        .order_by('date')
    )
    
    chart_data = {
        'labels': [record['date'].strftime('%Y-%m-%d') for record in daily_averages],
        'data': [round(record['avg_moisture'], 2) for record in daily_averages],
    }

    # Get prediction data from session if available
    prediction_data = request.session.get('prediction_data', {})
    if prediction_data:
        # Clear prediction data from session after using it
        del request.session['prediction_data']

    # Limit records to 10 unless show_all is true
    displayed_records = records[:10] if not show_all else records

    context = {
        'user': request.user,
        'records': displayed_records,
        'total_records': records.count(),
        'show_all': show_all,
        'recent_records': recent_records,
        'locations': locations,
        'selected_location': location,
        'start_date': start_date,
        'end_date': end_date,
        'average_moisture': average_moisture,
        'average_temperature': average_temperature,
        'average_humidity': average_humidity,
        'active_sensors': active_sensors,
        'total_sensors': total_sensors,
        'total_records': total_records,
        'last_update': last_update,
        'last_upload': 'N/A',
        'chart_data': json.dumps(chart_data),
        'current_weather': current_weather,
        'weather_location': default_location,
        'prediction': prediction_data.get('prediction'),
        'current_moisture': prediction_data.get('current_moisture'),
        'temperature': prediction_data.get('temperature'),
        'humidity': prediction_data.get('humidity'),
    }
    return render(request, 'dashboards/technician_dashboard.html', context)

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
            model_path = os.path.join(settings.BASE_DIR, 'ml_models', model_file.name)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                for chunk in model_file.chunks():
                    f.write(chunk)
            
            # Optionally retrain the model
            if request.POST.get('retrain'):
                _, metrics = train_model()  # Get metrics from retraining
                messages.success(request, f'Model retrained successfully! R² Score: {metrics["r2_score"]}% | RMSE: {metrics["rmse"]}')
            else:
                messages.success(request, 'Model uploaded successfully!')
            
            return redirect('admin_dashboard')
        
        except Exception as e:
            messages.error(request, f'Error uploading model: {str(e)}')
            return redirect('admin_dashboard')
    

    return render(request, 'dashboards/admin_dashboard.html')


@login_required
@role_required('admin')
def assign_technician(request):
    if request.method == 'POST':
        technician_id = request.POST.get('technician_id')
        location = request.POST.get('location')

        # Validate that the location exists in SoilMoistureRecord
        if not SoilMoistureRecord.objects.filter(location=location).exists():
            messages.error(request, f"No soil moisture records found for location: {location}")
            logger.error(f"Attempted to assign technician to invalid location: {location}")
            return redirect('admin_dashboard')

        try:
            technician = CustomUser.objects.get(id=technician_id, role='technician')

            # Check if the technician is already assigned to the location
            if TechnicianLocationAssignment.objects.filter(technician=technician, location=location).exists():
                messages.warning(
                    request,
                    f"Technician {technician.get_full_name() or technician.username} is already assigned to {location}."
                )
                logger.warning(f"Attempted to assign already assigned technician {technician.email} to location {location}")
            else:
                # Create new assignment
                TechnicianLocationAssignment.objects.create(
                    technician=technician,
                    location=location
                )
                messages.success(
                    request,
                    f"Technician {technician.get_full_name() or technician.username} assigned to {location} successfully!"
                )
                logger.info(f"Technician {technician.email} assigned to location {location} by {request.user.email}")

        except CustomUser.DoesNotExist:
            messages.error(request, "Invalid technician selected.")
            logger.error(f"Failed to assign technician: Invalid technician ID {technician_id}")
        except Exception as e:
            messages.error(request, f"Error assigning technician: {str(e)}")
            logger.error(f"Error assigning technician: {str(e)}")

        return redirect('admin_dashboard')

    return redirect('admin_dashboard')

@login_required
@role_required('admin')
def unassign_technician(request):
    if request.method == 'POST':
        technician_id = request.POST.get('technician_id')
        location = request.POST.get('location')

        try:
            technician = CustomUser.objects.get(id=technician_id, role='technician')

            # Check if the technician is assigned to the location
            assignment = TechnicianLocationAssignment.objects.filter(
                technician=technician,
                location=location
            ).first()

            if assignment:
                assignment.delete()

                # Check if any other technicians are assigned to this location
                remaining_assignments = TechnicianLocationAssignment.objects.filter(location=location).count()
                if remaining_assignments == 0:
                    # Delete SoilMoistureRecord entries for this location
                    deleted_count = SoilMoistureRecord.objects.filter(location=location).delete()[0]
                    messages.success(
                        request,
                        f"Technician {technician.get_full_name() or technician.username} unassigned from {location} "
                        f"and {deleted_count} soil moisture records deleted."
                    )
                    logger.info(
                        f"Technician {technician.email} unassigned from location {location} by {request.user.email}. "
                        f"{deleted_count} records deleted."
                    )
                else:
                    messages.success(
                        request,
                        f"Technician {technician.get_full_name() or technician.username} unassigned from {location}."
                    )
                    logger.info(f"Technician {technician.email} unassigned from location {location} by {request.user.email}")

            else:
                messages.warning(
                    request,
                    f"Technician {technician.get_full_name() or technician.username} is not assigned to {location}."
                )
                logger.warning(f"Attempted to unassign unassigned technician {technician.email} from location {location}")

        except CustomUser.DoesNotExist:
            messages.error(request, "Invalid technician selected.")
            logger.error(f"Failed to unassign technician: Invalid technician ID {technician_id}")
        except Exception as e:
            messages.error(request, f"Error unassigning technician: {str(e)}")
            logger.error(f"Error unassigning technician: {str(e)}")

        return redirect('admin_dashboard')

    return redirect('admin_dashboard')

@login_required
@role_required('admin')
def add_technician(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name', '').strip()
        last_name = request.POST.get('last_name', '').strip()
        email = request.POST.get('email', '').lower().strip()

        try:
            if CustomUser.objects.filter(email=email).exists():
                messages.error(request, "A user with this email already exists.")
                logger.error(f"Failed to add technician: Email {email} already exists")
            else:
                # Create username from email
                username = email.split('@')[0]
                
                # Ensure username is unique
                counter = 1
                original_username = username
                while CustomUser.objects.filter(username=username).exists():
                    username = f"{original_username}{counter}"
                    counter += 1

                technician = CustomUser.objects.create_user(
                    username=username,
                    email=email,
                    password='TechnicianDefault123!',  # Set a secure default password
                    role='technician',
                    first_name=first_name,
                    last_name=last_name
                )
                
                full_name = f"{first_name} {last_name}".strip()
                messages.success(request, f"Technician {full_name} ({email}) added successfully!")
                logger.info(f"Technician {full_name} ({email}) added by {request.user.email}")

                # Send welcome email
                try:
                    send_mail(
                        subject='Welcome to AgriSense Soil Monitoring System!',
                        message=f'Hi {first_name},\n\nYour technician account has been created.\n\nEmail: {email}\nTemporary Password: TechnicianDefault123!\n\nPlease change your password after first login.',
                        from_email=settings.EMAIL_HOST_USER,
                        recipient_list=[email],
                        fail_silently=False,
                    )
                    logger.info(f"Welcome email sent to {email}")
                except Exception as e:
                    logger.error(f"Failed to send email to {email}: {e}")

        except Exception as e:
            messages.error(request, f"Error adding technician: {str(e)}")
            logger.error(f"Error adding technician: {str(e)}")

        return redirect('admin_dashboard')

    return redirect('admin_dashboard')

import urllib.request

# Weather forecat api
def get_weather_forecast(location="Kampala"):
    if not weather_api:
        logger.error("OpenWeatherMap API key is not set.")
        return None  # Return None if no API key

    # Map field names to actual locations
    kampala_fields = ['Farm A', 'Farm B', 'Farm C', 'Farm D']
    if location in kampala_fields:
        location = 'Kampala'

    try:
        # OpenWeatherMap API endpoint for current weather
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_api}&units=metric"
        # Create request and get response
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode('utf-8'))
            else:
                logger.error(f"API request failed with status code: {response.getcode()}")
                return None
        
        # Extract relevant weather information
        weather_data = {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'description': data['weather'][0]['description'],
            'icon': data['weather'][0]['icon'],
            'wind_speed': data['wind']['speed'],
            'precipitation': data.get('rain', {}).get('1h', 0)  # Rain volume for last hour (mm)
        }
        
        return weather_data
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch current weather for {location}: {str(e)}")
        return None

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
            prediction_result = predict_soil_status(
                soil_moisture=current_moisture,
                temperature=temperature,
                humidity=humidity,
                location=location
            )
            
            # Store prediction
            SoilMoisturePrediction.objects.create(
                location=location,
                predicted_moisture=prediction_result['status'],  # Use status or derived value
                current_moisture=current_moisture,
                temperature=temperature,
                humidity=humidity,
                precipitation=weather_forecast.get('rain', {}).get('1h', 0) if weather_forecast else 0,
                prediction_for=datetime.now() + timedelta(hours=24)
            )
            
            # Store prediction in context
            context = {
                'prediction': prediction_result['status'],
                'location': location,
                'current_moisture': current_moisture,
                'temperature': temperature,
                'humidity': humidity,
                'confidence': prediction_result['confidence'],
                'method': prediction_result['method'],
                'irrigation_recommendation': prediction_result['irrigation_recommendation'],
                'input_values': prediction_result['input_values']
            }
            return render(request, 'dashboards/prediction_result.html', context)
        
        except ValueError as e:
            return redirect('admin_dashboard')
    
    return render(request, 'dashboards/admin_dashboard.html')

#Generating Reports
@login_required
@roles_required('admin', 'technician')
def generate_report(request):
    if request.method == 'POST':
        report_type = request.POST.get('report_type')
        format_type = request.POST.get('format_type')
        start_date = request.POST.get('start_date', '')
        end_date = request.POST.get('end_date', '')

        # Validate inputs
        if report_type not in ['daily', 'weekly', 'monthly']:
            messages.error(request, 'Invalid report type.')
            return redirect('technician_dashboard' if request.user.role == 'technician' else 'admin_dashboard')
        if format_type not in ['pdf', 'excel']:
            messages.error(request, 'Invalid format type.')
            return redirect('technician_dashboard' if request.user.role == 'technician' else 'admin_dashboard')

        # Determine date range based on report type
        end_date = datetime.now() if not end_date else datetime.strptime(end_date, '%Y-%m-%d')
        if report_type == 'daily':
            start_date = end_date - timedelta(days=1)
        elif report_type == 'weekly':
            start_date = end_date - timedelta(days=7)
        else:  # monthly
            start_date = end_date - timedelta(days=30)

        # Query data
        moisture_records = SoilMoistureRecord.objects.filter(
            timestamp__range=[start_date, end_date]
        ).order_by('timestamp')
        prediction_records = SoilMoisturePrediction.objects.filter(
            timestamp__range=[start_date, end_date]
        ).order_by('timestamp')

        # Filter by assigned locations for technicians
        if request.user.role == 'technician':
            assigned_farms = TechnicianLocationAssignment.objects.filter(technician=request.user)
            assigned_locations = assigned_farms.values_list('location', flat=True).distinct()
            moisture_records = moisture_records.filter(location__in=assigned_locations)
            prediction_records = prediction_records.filter(location__in=assigned_locations)

        # Aggregate statistics
        stats = moisture_records.aggregate(
            avg_moisture=Avg('soil_moisture_percent'),
            min_moisture=Min('soil_moisture_percent'),
            max_moisture=Max('soil_moisture_percent')
        )
        stats = {
            'avg_moisture': float(stats['avg_moisture']) if stats['avg_moisture'] is not None else 0.0,
            'min_moisture': float(stats['min_moisture']) if stats['min_moisture'] is not None else 0.0,
            'max_moisture': float(stats['max_moisture']) if stats['max_moisture'] is not None else 0.0
        }

        if format_type == 'pdf':
            return generate_pdf_report(request, report_type, moisture_records, prediction_records, stats, start_date, end_date)
        else:  # excel
            return generate_excel_report(request, report_type, moisture_records, prediction_records, stats, start_date, end_date)

    return redirect('technician_dashboard' if request.user.role == 'technician' else 'admin_dashboard')

def generate_pdf_report(request, report_type, moisture_records, prediction_records, stats, start_date, end_date):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica", 12)

    # Title
    p.drawString(100, 750, f"{report_type.capitalize()} Soil Moisture Report")
    p.drawString(100, 730, f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Statistics
    p.drawString(100, 700, "Summary Statistics:")
    p.drawString(100, 680, f"Average Moisture: {stats['avg_moisture']:.2f}%")
    p.drawString(100, 660, f"Min Moisture: {stats['min_moisture']:.2f}%")
    p.drawString(100, 640, f"Max Moisture: {stats['max_moisture']:.2f}%")

    # Moisture Records
    y = 600
    p.drawString(100, y, "Soil Moisture Records:")
    y -= 20
    for record in moisture_records[:10]:  # Limit to 10 for brevity
        p.drawString(100, y, f"{record.timestamp.strftime('%Y-%m-%d %H:%M')}: {record.location}, {record.soil_moisture_percent}%")
        y -= 20
        if y < 100:
            p.showPage()
            y = 750

    # Prediction Records
    p.drawString(100, y, "Prediction Records:")
    y -= 20
    for pred in prediction_records[:10]:  # Limit to 10 for brevity
        p.drawString(100, y, f"{pred.timestamp.strftime('%Y-%m-%d %H:%M')}: {pred.location}, Predicted: {pred.predicted_moisture}%")
        y -= 20
        if y < 100:
            p.showPage()
            y = 750

    p.showPage()
    p.save()
    buffer.seek(0)
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{report_type}_report_{datetime.now().strftime("%Y%m%d")}.pdf"'
    response.write(buffer.getvalue())
    buffer.close()
    return response

def generate_excel_report(request, report_type, moisture_records, prediction_records, stats, start_date, end_date):
    wb = Workbook()
    ws = wb.active
    ws.title = f"{report_type.capitalize()} Report"

    # Write headers
    ws.append([f"{report_type.capitalize()} Soil Moisture Report"])
    ws.append([f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"])
    ws.append([])
    ws.append(["Summary Statistics"])
    ws.append(["Average Moisture", f"{stats['avg_moisture']:.2f}%"])
    ws.append(["Min Moisture", f"{stats['min_moisture']:.2f}%"])
    ws.append(["Max Moisture", f"{stats['max_moisture']:.2f}%"])
    ws.append([])

    # Moisture Records
    ws.append(["Soil Moisture Records"])
    ws.append(["Timestamp", "Location", "Moisture (%)", "Status"])
    for record in moisture_records:
        ws.append([
            record.timestamp.strftime('%Y-%m-%d %H:%M'),
            record.location,
            record.soil_moisture_percent,
            record.status
        ])

    # Prediction Records
    ws.append([])
    ws.append(["Prediction Records"])
    ws.append(["Timestamp", "Location", "Predicted Moisture (%)", "Current Moisture (%)"])
    for pred in prediction_records:
        ws.append([
            pred.timestamp.strftime('%Y-%m-%d %H:%M'),
            pred.location,
            pred.predicted_moisture,
            pred.current_moisture
        ])

    buffer = BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = f'attachment; filename="{report_type}_report_{datetime.now().strftime("%Y%m%d")}.xlsx"'
    response.write(buffer.getvalue())
    buffer.close()
    return response

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
            prediction = predict_soil_status(
                location, current_moisture, temperature, humidity, weather_forecast
            )
            
            # Store prediction
            SoilMoisturePrediction.objects.create(
                location=location,
                predicted_moisture=round(prediction, 2),
                current_moisture=current_moisture,
                temperature=temperature,
                humidity=humidity,
                precipitation=weather_forecast['precipitation'],
                prediction_for=datetime.now() + timedelta(hours=24)
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


@login_required
@role_required('technician')
def technician_predict_moisture(request):
    technician = request.user
    assigned_farms = TechnicianLocationAssignment.objects.filter(technician=technician)
    assigned_locations = assigned_farms.values_list('location', flat=True).distinct()

    if request.method == 'POST':
        location = request.POST.get('location')
        if location not in assigned_locations:
            messages.error(request, "You can only make predictions for assigned farms.")
            return redirect('technician_dashboard')

        try:
            current_moisture = float(request.POST.get('soil_moisture_percent'))
            temperature = float(request.POST.get('temperature'))
            humidity = float(request.POST.get('humidity'))
            
            weather_forecast = get_weather_forecast(location)

            # Handle case when weather forecast is None
            if weather_forecast is None:
                precipitation = 0  # Default value if weather API fails
            else:
                precipitation = weather_forecast.get('rain', {}).get('1h', 0)
            
            prediction = predict_soil_status(
                location, current_moisture, temperature, humidity, weather_forecast
            )
            
            SoilMoisturePrediction.objects.create(
                location=location,
                predicted_moisture=round(prediction, 2),
                current_moisture=current_moisture,
                temperature=temperature,
                humidity=humidity,
                precipitation=precipitation,  # Fixed variable name here
                prediction_for=datetime.now() + timedelta(hours=24)
            )
            
            # Store prediction data in session
            request.session['prediction_data'] = {
                'prediction': round(prediction, 2),
                'location': location,
                'current_moisture': current_moisture,
                'temperature': temperature,
                'humidity': humidity,
            }
            return redirect('technician_dashboard')
        
        except ValueError as e:
            messages.error(request, f"Invalid input: {str(e)}")
            return redirect('technician_dashboard')
    
    context = {
        'locations': assigned_locations,
    }
    return render(request, 'dashboards/prediction_form.html', context)
