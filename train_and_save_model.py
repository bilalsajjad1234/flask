import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Sample training data — replace this with your own dataset if you have one
data = pd.DataFrame({
    'Year': [2015, 2018, 2020, 2012, 2019],
    'Kilometers_Driven': [50000, 30000, 20000, 70000, 25000],
    'Fuel_Type': ['Petrol', 'Diesel', 'Petrol', 'CNG', 'Diesel'],
    'Transmission': ['Manual', 'Automatic', 'Manual', 'Manual', 'Automatic'],
    'Engine_CC': [1500, 1200, 1800, 1000, 1600],
    'Body_Type': ['Sedan', 'SUV', 'Hatchback', 'Sedan', 'SUV'],
    'Manufacturer': ['Honda', 'Toyota', 'Suzuki', 'Hyundai', 'Toyota'],
    'Selling_Price': [5.5, 7.2, 6.0, 3.8, 8.0]
})

# Derived feature
data['Car_Age'] = 2025 - data['Year']

# Drop 'Year' after using it to create 'Car_Age'
data.drop('Year', axis=1, inplace=True)

# Handle categorical variables
categorical_cols = ['Manufacturer', 'Fuel_Type', 'Transmission', 'Body_Type']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Features and target
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Save feature columns
feature_columns = X.columns.tolist()

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
rf_model = RandomForestRegressor()
rf_model.fit(X_scaled, y)

# Save the model, scaler, and feature columns
joblib.dump(rf_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')

print("✅ Model, Scaler, and Feature Columns saved successfully.")
