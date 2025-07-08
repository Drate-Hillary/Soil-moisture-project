from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('admin_dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('farmer_dashboard/', views.farmer_dashboard, name='farmer_dashboard'),
    path('technician_dashboard/', views.technician_dashboard, name='technician_dashboard'),
    path('profile/', views.profile, name='profile'),
]