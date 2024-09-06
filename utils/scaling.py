from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Function to scale the data
def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))  # Assuming data is a single column like 'Close'
    return scaled_data, scaler

# Function to inverse the scaling
def inverse_scale(scaled_data, scaler):
    return scaler.inverse_transform(scaled_data)

# Function to scale new data based on the existing scaler (used for predictions)
def scale_new_data(data, scaler):
    return scaler.transform(data.values.reshape(-1, 1))
