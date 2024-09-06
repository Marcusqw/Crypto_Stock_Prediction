import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# ----------------------- Data Loader -----------------------
def get_historical_data(ticker, period):
    data = yf.download(ticker, period=period)
    return data

# ------------------- Feature Engineering -------------------
def add_technical_indicators(data):
    # Adding Simple Moving Average (SMA), Exponential Moving Average (EMA), and Relative Strength Index (RSI)
    data['SMA'] = data['Close'].rolling(window=14).mean()
    data['EMA'] = data['Close'].ewm(span=14, adjust=False).mean()
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data.dropna()

def add_lagged_features(data, lag=3):
    for i in range(1, lag+1):
        data[f'lag_{i}'] = data['Close'].shift(i)
    data = data.dropna()  # Remove rows with NaN values
    return data

# ----------------------- Scaling --------------------------
def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
    return scaled_data, scaler

def inverse_scale(scaled_data, scaler):
    return scaler.inverse_transform(scaled_data)

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
def train_lstm_model(X_train, y_train, epochs=100, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))  # Regularization to prevent overfitting
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    
    return model

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

# ------------------------ Main Script -----------------------
# Step 1: Load the historical data
ticker = input("Enter the stock ticker (e.g., NVDA): ")
training_period = input("Enter the historical training period (e.g., '1y', '2y', '5y'): ")
predict_days = int(input("Enter how many days into the future you'd like to predict: "))

data = get_historical_data(ticker, training_period)

# Step 2: Add technical indicators and lagged features
data = add_technical_indicators(data)
data = add_lagged_features(data)

# Step 3: Scale the data
scaled_data, scaler = scale_data(data['Close'])

# Step 4: Prepare the data for LSTM
X_train, y_train, lookback = prepare_lstm_data(scaled_data)

# Step 5: Train the LSTM model
model = train_lstm_model(X_train, y_train, epochs=100, batch_size=32)

# Step 6: Make predictions on the test set
predicted_prices_scaled = model.predict(X_train)
predicted_prices = inverse_scale(predicted_prices_scaled, scaler)

# Step 7: Evaluate the model
evaluate_model(data['Close'].values[-len(predicted_prices):], predicted_prices.flatten())

# Step 8: Future Predictions
def future_predictions(data, model, scaler, lookback, predict_days):
    last_data = scaled_data[-lookback:]
    future_predictions = []
    for _ in range(predict_days):
        future_pred = model.predict(last_data.reshape(1, lookback, 1))
        future_pred_rescaled = inverse_scale(future_pred, scaler)
        future_predictions.append(future_pred_rescaled[0][0])
        last_data = np.append(last_data[1:], future_pred, axis=0)
    return future_predictions

# Fix: Generate correct number of future dates (1 date for each predicted day)
future_prices = future_predictions(data, model, scaler, lookback, predict_days)
future_dates = pd.date_range(start=data.index[-1], periods=predict_days+1)[1:]  # Skip the starting date

# Step 10: Plot the Future Predictions
print(f"Predicted prices for the next {predict_days} days:")
for i, price in enumerate(future_prices, start=1):
    print(f"Day {i}: {price}")

# Pass the correctly shaped data
plot_predictions(pd.DataFrame(index=future_dates), None, future_prices, future=True)
