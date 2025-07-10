import joblib
import pandas as pd
import os
from django.conf import settings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .models import SoilMoistureRecord
import json
import logging
import numpy as np
from datetime import datetime

# Define paths for model and metrics
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'soil_moisture_model.pkl')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'scaler.pkl')
METRICS_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'model_metrics.json')

# Initialize model variable
model = None
scaler = None

logger = logging.getLogger(__name__)

def create_features(df):
    """Create engineered features for better prediction"""
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    
    # Lag features (previous values)
    df['moisture_lag_1'] = df['soil_moisture_percent'].shift(1)
    df['moisture_lag_2'] = df['soil_moisture_percent'].shift(2)
    df['temp_lag_1'] = df['temperature_celsius'].shift(1)
    df['humidity_lag_1'] = df['humidity_percent'].shift(1)
    
    # Rolling averages
    df['moisture_rolling_3'] = df['soil_moisture_percent'].rolling(window=3, min_periods=1).mean()
    df['temp_rolling_3'] = df['temperature_celsius'].rolling(window=3, min_periods=1).mean()
    df['humidity_rolling_3'] = df['humidity_percent'].rolling(window=3, min_periods=1).mean()
    
    # Rate of change
    df['moisture_change'] = df['soil_moisture_percent'].diff()
    df['temp_change'] = df['temperature_celsius'].diff()
    df['humidity_change'] = df['humidity_percent'].diff()
    
    return df

def train_model():
    global model, scaler
    try:
        # Fetch historical data
        records = SoilMoistureRecord.objects.all().values(
            'soil_moisture_percent', 'temperature_celsius', 'humidity_percent', 'timestamp'
        )
        df = pd.DataFrame(records)
        
        # Check if there is enough data to train
        if len(df) < 50:  # Increased minimum requirement
            logger.error(f"Insufficient data to train the model: {len(df)} records found. At least 50 records are recommended.")
            raise ValueError(f"Insufficient data to train the model. Found {len(df)} records, but at least 50 are recommended for meaningful predictions.")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create engineered features
        df = create_features(df)
        
        # Define features for training
        feature_columns = [
            'temperature_celsius', 'humidity_percent', 'hour', 'day_of_year', 'month',
            'moisture_lag_1', 'moisture_lag_2', 'temp_lag_1', 'humidity_lag_1',
            'moisture_rolling_3', 'temp_rolling_3', 'humidity_rolling_3',
            'moisture_change', 'temp_change', 'humidity_change'
        ]
        
        # Prepare features and target
        X = df[feature_columns].copy()
        y = df['soil_moisture_percent'].copy()
        
        # Remove rows with NaN values (due to lag features)
        valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) < 20:
            logger.error(f"After feature engineering, only {len(X)} valid samples remain. At least 20 are needed.")
            raise ValueError(f"After feature engineering, only {len(X)} valid samples remain. At least 20 are needed.")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train model with better hyperparameters
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Additional metrics
        mae = np.mean(np.abs(y_test - y_pred))  # Mean Absolute Error
        
        # Save model and scaler
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        logger.info(f"Model and scaler saved successfully")
        
        # Save metrics
        metrics = {
            'rmse': round(rmse, 3),
            'r2_score': round(r2 * 100, 2),
            'mae': round(mae, 3),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'total_samples': len(X)
        }
        
        try:
            with open(METRICS_PATH, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved successfully: {metrics}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")
            raise
        
        return model, metrics
    
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}", exc_info=True)
        raise

# Load or train the model
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"Model and scaler loaded successfully")
    else:
        logger.warning(f"Model files not found. Training a new model...")
        model, _ = train_model()
except Exception as e:
    logger.error(f"Error loading model: {str(e)}", exc_info=True)
    model = None
    scaler = None

def predict_soil_moisture(location, current_moisture, temperature, humidity, weather_forecast):
    if model is None or scaler is None:
        logger.error("Machine learning model or scaler is not loaded or trained.")
        raise ValueError("Machine learning model or scaler is not loaded or trained.")
    
    # For prediction, we need to simulate the engineered features
    # This is simplified - in a real scenario, you'd want to maintain recent history
    now = datetime.now()
    
    # Prepare input data with engineered features
    input_data = pd.DataFrame({
        'temperature_celsius': [temperature],
        'humidity_percent': [humidity],
        'hour': [now.hour],
        'day_of_year': [now.timetuple().tm_yday],
        'month': [now.month],
        'moisture_lag_1': [current_moisture],  # Use current as lag
        'moisture_lag_2': [current_moisture],  # Simplified
        'temp_lag_1': [temperature],
        'humidity_lag_1': [humidity],
        'moisture_rolling_3': [current_moisture],
        'temp_rolling_3': [temperature],
        'humidity_rolling_3': [humidity],
        'moisture_change': [0.0],  # Simplified
        'temp_change': [0.0],
        'humidity_change': [0.0]
    })
    
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    try:
        prediction = model.predict(input_scaled)[0]
        logger.info(f"Prediction made for {location}: {prediction}%")
        return prediction
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        raise

def get_model_metrics():
    try:
        logger.info(f"Loading metrics from {METRICS_PATH}")
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
                logger.info(f"Metrics loaded: {metrics}")
                return metrics
        logger.warning(f"Metrics file not found at {METRICS_PATH}")
        return {'rmse': None, 'r2_score': None, 'mae': None, 'training_samples': None}
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}", exc_info=True)
        return {'rmse': None, 'r2_score': None, 'mae': None, 'training_samples': None}