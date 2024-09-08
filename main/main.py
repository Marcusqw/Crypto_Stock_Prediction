import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import re

# ----------------------- Data Loader -----------------------
def get_historical_data(ticker, period):
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            raise ValueError(f"No data available for {ticker} with period '{period}'.")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise

# ------------------- Flexible Date Input with Validation -------------------
def get_period_input():
    # Input options for both years and days
    period_type = input("Enter 'years' for a year-based period or 'days' for day-based period: ").lower()
    
    # Handle year-based periods
    if period_type == 'years':
        years = int(input("Enter the number of years (1, 2, 5, or 10 years): "))
        valid_years = [1, 2, 5, 10]  # Limit options to valid periods
        if years in valid_years:
            return f'{years}y'
        else:
            raise ValueError("Please enter one of the valid periods: 1, 2, 5, or 10 years.")
    
    # Handle day-based periods
    elif period_type == 'days':
        days = int(input("Enter the number of days (50-300 days): "))
        if 50 <= days <= 300:
            return f'{days}d'
        else:
            raise ValueError("Please enter a value between 50 and 300 days.")
    
    else:
        raise ValueError("Invalid period type. Enter 'years' or 'days'.")

# ------------------- Feature Engineering -------------------
def add_technical_indicators(data):
    data['SMA'] = data['Close'].rolling(window=14).mean()
    data['EMA'] = data['Close'].ewm(span=14, adjust=False).mean()
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Adding Bollinger Bands (20-day SMA +/- 2 standard deviations)
    data['20_day_SMA'] = data['Close'].rolling(window=20).mean()
    data['Upper_BB'] = data['20_day_SMA'] + 2 * data['Close'].rolling(window=20).std()
    data['Lower_BB'] = data['20_day_SMA'] - 2 * data['Close'].rolling(window=20).std()
    
    # Adding Volatility (standard deviation of returns)
    data['Volatility'] = data['Close'].pct_change().rolling(window=14).std()
    
    # Adding MACD (Moving Average Convergence Divergence)
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    
    return data.dropna()

def add_lagged_features(data, lag=3):
    for i in range(1, lag+1):
        data[f'lag_{i}'] = data['Close'].shift(i)
    data = data.dropna()  # Remove rows with NaN values
    return data

# ----------------------- Scaling --------------------------
def scale_data(data):
    if data.shape[0] == 0:
        raise ValueError("Cannot scale an empty dataset.")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
    return scaled_data, scaler

def inverse_scale(scaled_data, scaler):
    return scaler.inverse_transform(scaled_data)

# -------------------- Train-Test Split --------------------
def train_test_split(data, test_size=0.2):
    train_size = int(len(data) * (1 - test_size))
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# -------------------- Model Preparation -------------------
def prepare_lstm_data(data, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])  # Use past 'lookback' values
        y.append(data[i, 0])  # Predict the current value
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, lookback

# --------------------- Model Training ---------------------
def train_lstm_model(X_train, y_train, epochs=100, batch_size=32, patience=10):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))  # Regularization to prevent overfitting
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Adding EarlyStopping callback
    early_stopping = EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[early_stopping])
    
    return model

# -------------------- Model Save/Load ---------------------
def save_model(model, file_path='lstm_model.h5'):
    model.save(file_path)
    print(f"Model saved to {file_path}.")

def load_model(file_path='lstm_model.h5'):
    if os.path.exists(file_path):
        model = tf.keras.models.load_model(file_path)
        print(f"Model loaded from {file_path}.")
        return model
    else:
        raise FileNotFoundError(f"No model file found at {file_path}.")

# ----------------------- Evaluation ------------------------
def evaluate_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    print(f'MSE: {mse}')
    print(f'R2 Score: {r2}')
    return mse, r2

def plot_predictions(data, actual, predicted, future=False):
    plt.figure(figsize=(10, 6))
    
    if future:
        # For future predictions, only plot the predicted prices and dates
        plt.plot(data.index, predicted, color='green', label='Predicted Future Prices')
    else:
        # For historical data, plot both actual and predicted prices
        plt.plot(data.index, actual, color='blue', label='Actual Prices')
        plt.plot(data.index, predicted, color='red', linestyle='--', label='Predicted Prices')

    plt.title("Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

# ------------------------ Prediction Accuracy ------------------------
def calculate_accuracy(actual, predicted):
    # Calculate RMSE (Root Mean Squared Error) and return accuracy percentage
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    accuracy = 100 - (rmse / np.mean(actual) * 100)
    print(f'Prediction Accuracy: {accuracy:.2f}%')
    return accuracy

# ------------------------ Future Prediction Time ------------------------
def parse_prediction_input():
    """Parses user input to predict future prices in days or hours, supporting both singular and plural forms."""
    user_input = input("Enter how many days or hours into the future you'd like to predict (e.g., '10 days', '5 hours'): ").lower()
    
    # Modify the regex to accept both 'day'/'days' and 'hour'/'hours'
    match = re.match(r'(\d+)\s*(day|days|hour|hours)', user_input)
    
    if not match:
        raise ValueError("Invalid input. Please enter a number followed by 'days' or 'hours' (e.g., '10 days' or '5 hours').")
    
    value, interval = match.groups()
    value = int(value)  # Convert the numeric part to an integer
    return value, interval


def future_predictions(data, model, scaler, lookback, predict_value, interval='days'):
    """Generates future predictions for a specified number of days or hours."""
    last_data = data[-lookback:]  # Get the last "lookback" amount of data for prediction
    future_predictions = []
    
    for _ in range(predict_value):
        future_pred = model.predict(last_data.reshape(1, lookback, 1))
        future_pred_rescaled = inverse_scale(future_pred, scaler)
        future_predictions.append(future_pred_rescaled[0][0])
        last_data = np.append(last_data[1:], future_pred, axis=0)

    # Get the last date from the original data (assuming it's a pandas DataFrame)
    last_date = pd.to_datetime(data.index[-1])  # Get the last valid date from the original data
    
    # Generate future dates based on the interval ('days' or 'hours')
    if interval == 'days':
        future_dates = pd.date_range(start=last_date, periods=predict_value+1)[1:]
    elif interval == 'hours':
        future_dates = pd.date_range(start=last_date, periods=predict_value+1, freq='H')[1:]  # Hourly prediction
    else:
        raise ValueError("Interval must be 'days' or 'hours'.")
    
    return future_predictions, future_dates


# ------------------------ Main Script -----------------------
try:
    # Step 1: Get period input for flexibility
    period = get_period_input()

    # Step 2: Load the historical data
    ticker = input("Enter the stock ticker (e.g., NVDA): ")
    data = get_historical_data(ticker, period)

    # Step 3: Add technical indicators and lagged features
    data = add_technical_indicators(data)
    data = add_lagged_features(data)

    # Step 4: Train-test split
    train_data, test_data = train_test_split(data['Close'])

    # Step 5: Scale the data
    scaled_train_data, scaler_train = scale_data(train_data)
    scaled_test_data, scaler_test = scale_data(test_data)

    # Step 6: Prepare the data for LSTM
    X_train, y_train, lookback = prepare_lstm_data(scaled_train_data)
    X_test, y_test, _ = prepare_lstm_data(scaled_test_data)

    # Step 7: Train the LSTM model with early stopping
    model = train_lstm_model(X_train, y_train, epochs=100, batch_size=32)

    # Step 8: Save the model
    save_model(model)

    # Step 9: Make predictions on the test set
    predicted_prices_scaled = model.predict(X_test)
    predicted_prices = inverse_scale(predicted_prices_scaled, scaler_test)

    # Step 10: Evaluate the model
    evaluate_model(test_data.values[-len(predicted_prices):], predicted_prices.flatten())

    # Step 11: Calculate prediction accuracy
    accuracy = calculate_accuracy(test_data.values[-len(predicted_prices):], predicted_prices.flatten())

    # Step 12: Get future prediction input
    predict_value, interval = parse_prediction_input()

    # Step 13: Predict future prices (for specified days or hours)
    future_prices, future_dates = future_predictions(scaled_train_data, model, scaler_train, lookback, predict_value, interval)

    # Step 14: Plot the future predictions
    plot_predictions(pd.DataFrame(index=future_dates), None, future_prices, future=True)

except ValueError as ve:
    print(f"ValueError: {ve}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
