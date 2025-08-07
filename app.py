from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import datetime

app = Flask(__name__)
CORS(app)

# Load and preprocess dataset
dataset = pd.read_csv("usedCarsFinal.csv")

def preprocess_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    column_mapping = {
        'name': 'Name',
        'model_year': 'Year',
        'mileage': 'Kilometers_Driven',
        'engine_type': 'Fuel_Type',
        'transmission': 'Transmission',
        'registered_in': 'Registered_In',
        'engine_capacity': 'Engine_CC',
        'body_type': 'Body_Type',
        'price': 'Price'
    }
    df = df.rename(columns=column_mapping)

    df["Manufacturer"] = df["Name"].str.split(" ", expand=True)[0]
    df["Car_Age"] = datetime.datetime.now().year - df["Year"]

    def clean_numeric(col):
        if col.dtype == object:
            return pd.to_numeric(col.astype(str).str.replace('[^\d.]', '', regex=True), errors='coerce')
        return col

    for col in ['Kilometers_Driven', 'Engine_CC', 'Price']:
        if col in df.columns:
            df[col] = clean_numeric(df[col])
            df[col] = df[col].fillna(df[col].median())

    return df

dataset = preprocess_data(dataset)

# Features and target
X = dataset.drop(["Name", "location", "color", "assembly", "Price", "Registered_In", "url", "other_features_list"], axis=1, errors='ignore')
y = dataset["Price"]

# Handle categorical variables
categorical_cols = ["Manufacturer", "Fuel_Type", "Transmission", "Body_Type"]
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

# Calculate R2 Score for metrics endpoint
y_pred_train = rf_model.predict(X_scaled)
r2 = r2_score(y, y_pred_train)

# Prediction endpoint
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame({
            'Name': [data.get('carName', '')],
            'Year': [int(data.get('manufactureYear', 2020))],
            'Kilometers_Driven': [float(data.get('mileage', 0))],
            'Fuel_Type': [data.get('fuelType', 'Petrol').capitalize()],
            'Transmission': [data.get('transmission', 'Automatic').capitalize()],
            'Engine_CC': [float(data.get('engineCC', 1500))],
            'Body_Type': [data.get('bodyType', 'Sedan').capitalize()]
        })

        input_data["Manufacturer"] = input_data["Name"].str.split(" ", expand=True)[0]
        input_data["Car_Age"] = datetime.datetime.now().year - input_data["Year"]
        input_data = input_data.drop(["Name"], axis=1)

        input_data = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)

        # Ensure all columns match training data
        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[X.columns]

        input_scaled = scaler.transform(input_data)
        prediction = rf_model.predict(input_scaled)

        return jsonify({
            'predicted_price': float(prediction[0]),
            'status': 'success',
            'message': 'Prediction successful'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# R2 Score endpoint
@app.route('/api/metrics', methods=['GET'])
def metrics():
    try:
        return jsonify({
            'r2_score': round(r2, 4),
            'status': 'success',
            'message': 'Model evaluation successful'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
