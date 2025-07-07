from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.core.mail import send_mail
from django.conf import settings
from .forms import CustomUserCreationForm
from django.contrib.auth import authenticate, login, logout
from .templates.accounts.decorators import role_required


def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            # Send email notification
            send_mail(
                subject='Welcome to Farm System!',
                message=f'Hi {user.username},\n\nYour account has been created successfully!',
                from_email=settings.EMAIL_HOST_USER,
                recipient_list=[user.email],
                fail_silently=False,
            )
            return redirect('home')  # Redirect to home or dashboard
    else:
        form = CustomUserCreationForm()
    return render(request, 'accounts/register.html', {'form': form})


def user_login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, email=email, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            return render(request, 'accounts/login.html', {'error': 'Invalid credentials'})
    return render(request, 'accounts/login.html')


def user_logout(request):
    logout(request)
    return redirect('login')



@role_required('admin')
def admin_dashboard(request):
    return render(request, 'accounts/admin_dashboard.html', {'user': request.user})

@role_required('farmer')
def farmer_dashboard(request):
    return render(request, 'accounts/farmer_dashboard.html', {'user': request.user})

from django.contrib.auth.decorators import login_required

@login_required
def profile(request):
    if request.method == 'POST':
        user = request.user
        user.email = request.POST.get('email')
        user.username = request.POST.get('username')
        user.save()
        return redirect('profile')
    return render(request, 'accounts/profile.html', {'user': request.user})