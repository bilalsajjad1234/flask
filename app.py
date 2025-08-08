from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import datetime
import os

app = Flask(__name__)
CORS(app)

# Global variables for model components
rf_model = None
scaler = None
feature_columns = None
r2_score_value = None

def load_model():
    """Load pre-trained model components"""
    global rf_model, scaler, feature_columns, r2_score_value
    
    try:
        # Load pre-trained model components
        rf_model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        r2_score_value = 0.85  # Store your actual R2 score here
        
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load model on startup
if not load_model():
    print("Warning: Model not loaded. Using dummy values.")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': rf_model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if rf_model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 500
            
        data = request.json
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Year': [int(data.get('manufactureYear', 2020))],
            'Kilometers_Driven': [float(data.get('mileage', 0))],
            'Fuel_Type': [data.get('fuelType', 'Petrol').capitalize()],
            'Transmission': [data.get('transmission', 'Automatic').capitalize()],
            'Engine_CC': [float(data.get('engineCC', 1500))],
            'Body_Type': [data.get('bodyType', 'Sedan').capitalize()]
        })

        # Add derived features
        car_name = data.get('carName', '')
        manufacturer = car_name.split(' ')[0] if car_name else 'Unknown'
        input_data["Manufacturer"] = manufacturer
        input_data["Car_Age"] = datetime.datetime.now().year - input_data["Year"]

        # Handle categorical variables
        categorical_cols = ["Manufacturer", "Fuel_Type", "Transmission", "Body_Type"]
        input_data = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)

        # Ensure all columns match training data
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns to match training data
        input_data = input_data.reindex(columns=feature_columns, fill_value=0)

        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = rf_model.predict(input_scaled)

        return jsonify({
            'predicted_price': float(prediction[0]),
            'status': 'success',
            'message': 'Prediction successful'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    


@app.route('/api/metrics', methods=['GET'])
def metrics():
    try:
        return jsonify({
            'r2_score': r2_score_value or 0.0,
            'status': 'success',
            'message': 'Model evaluation successful'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Car Price Prediction API',
        'status': 'running',
        'endpoints': ['/api/predict', '/api/metrics', '/health']
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)