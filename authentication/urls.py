from django.urls import path
from . import views
from .views import upload_csv, predict_moisture, upload_model    

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),  # This will be removed/renamed later
    path('logout/', views.user_logout, name='logout'),
    path('admin_dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('farmer_dashboard/', views.farmer_dashboard, name='farmer_dashboard'),
    path('technician_dashboard/', views.technician_dashboard, name='technician_dashboard'),
    path('profile/', views.profile, name='profile'),
    path('upload_csv/', upload_csv, name='upload_csv'),
    path('predict_moisture/', predict_moisture, name='predict_moisture'),
    path('upload_model/', upload_model, name='upload_model'),
    path('generate_report/', views.generate_report, name='generate_report'),
    path('assign-technician/', views.assign_technician, name='assign_technician'),
    path('unassign-technician/', views.unassign_technician, name='unassign_technician'),
    path('add-technician/', views.add_technician, name='add_technician'), 
    path('technician/predict/', views.technician_predict_moisture, name='technician_predict_moisture'),

]