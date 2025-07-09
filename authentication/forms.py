from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser
from django.core.exceptions import ValidationError

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    role = forms.ChoiceField(choices=CustomUser.ROLE_CHOICES, required=True)

    class Meta:
        model = CustomUser
        fields = ['email', 'username', 'role', 'password1', 'password2']

    def clean_email(self):
        email = self.cleaned_data.get('email').lower()
        if CustomUser.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already registered.")
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email'].lower()
        user.role = self.cleaned_data['role']
        if commit:
            user.save()
        return user

class SoilMoistureFilterForm(forms.Form):
    location = forms.CharField(max_length=100, required=False)
    start_date = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}))
    end_date = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}))
    source = forms.ChoiceField(choices=[('', 'All Sources'), ('iot', 'IoT Device'), ('csv', 'CSV Upload'), ('manual', 'Manual Input')], required=False)
    moisture_min = forms.FloatField(required=False, min_value=0, max_value=100, widget=forms.NumberInput(attrs={'step': '0.1', 'class': 'form-control'}))
    moisture_max = forms.FloatField(required=False, min_value=0, max_value=100, widget=forms.NumberInput(attrs={'step': '0.1', 'class': 'form-control'}))

    def clean(self):
        cleaned_data = super().clean()
        start_date = cleaned_data.get('start_date')
        end_date = cleaned_data.get('end_date')

        if start_date and end_date and start_date > end_date:
            raise ValidationError("Start date must be before end date.")
        return cleaned_data