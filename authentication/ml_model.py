# soil_moisture_app/ml_model.py
import joblib
import pandas as pd
import os
from django.conf import settings
from sklearn.ensemble import RandomForestRegressor
from .models import SoilMoistureRecord

# Define MODEL_PATH using settings.BASE_DIR for portability
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'soil_moisture_model.pkl')

# Initialize model variable
model = None

# Load or train the model
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        # If model file doesn't exist, train a new model (or set to None)
        print(f"Model file not found at {MODEL_PATH}. Training a new model...")
        # Ensure models directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        # Train a new model (call the train_model function defined below)
        model = train_model()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None  # Fallback to None if loading/training fails

def predict_soil_moisture(location, current_moisture, temperature, humidity, weather_forecast):
    if model is None:
        raise ValueError("Machine learning model is not loaded or trained.")
    
    # Prepare input data
    input_data = pd.DataFrame({
        'soil_moisture_percent': [current_moisture],
        'temperature_celsius': [temperature],
        'humidity_percent': [humidity],
        'precipitation_forecast': [weather_forecast.get('precipitation', 0)],
        # Add encoded location or other features as needed
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    return prediction

def train_model():
    # Fetch historical data
    records = SoilMoistureRecord.objects.all().values(
        'soil_moisture_percent', 'temperature_celsius', 'humidity_percent', 'timestamp'
    )
    df = pd.DataFrame(records)
    
    # Check if there is enough data to train
    if len(df) < 2:  # Need at least 2 records for shift(-1)
        raise ValueError("Insufficient data to train the model.")
    
    # Preprocess data (example: simple feature engineering)
    X = df[['soil_moisture_percent', 'temperature_celsius', 'humidity_percent']]
    y = df['soil_moisture_percent'].shift(-1).dropna()  # Predict next moisture level
    X = X.iloc[:-1]  # Align X and y
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model