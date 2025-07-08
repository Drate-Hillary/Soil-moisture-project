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
    return render(request, 'dashboards/admin_dashboard.html', {'user': request.user})

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