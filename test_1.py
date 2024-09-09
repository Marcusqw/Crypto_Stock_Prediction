import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Fetch stock data using yfinance
def get_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    return stock_data

# Fetch real-time stock price
def get_real_time_stock_price(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    today_data = stock.history(period='1d')
    return today_data['Close'].iloc[-1]

# Prepare dataset for LSTM model
def prepare_data(df, feature_column, window_size):
    data = df[feature_column].values
    data = data.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X_train, y_train = [], []
    for i in range(window_size, len(scaled_data)):
        X_train.append(scaled_data[i-window_size:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train, scaler

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], 1)))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=25))
    model.add(Dense(units=1))  # Output layer
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict future stock prices
def predict_stock_prices(model, df, window_size, scaler, future_days):
    last_window = df[-window_size:].values
    last_window_scaled = scaler.transform(last_window.reshape(-1, 1))  # Fix: Reshaping the last window
    
    predictions = []
    for _ in range(future_days):
        # Reshape the data for prediction
        input_data = np.reshape(last_window_scaled, (1, window_size, 1))
        
        # Make the prediction
        predicted_price = model.predict(input_data)
        predicted_price_rescaled = scaler.inverse_transform(predicted_price)
        predictions.append(predicted_price_rescaled[0, 0])
        
        # Update the last window to include the predicted value
        last_window_scaled = np.append(last_window_scaled[1:], predicted_price)
        last_window_scaled = np.reshape(last_window_scaled, (window_size, 1))
    
    return predictions

# Main script
stock_symbol = 'AAPL'  # Apple stock
start_date = '2012-01-01'
end_date = '2023-01-01'
window_size = 60  # 60 days window for LSTM
future_days = 30  # Predict next 30 days

# Fetch historical data
df = get_stock_data(stock_symbol, start_date, end_date)

# Convert 'Date' column to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Prepare training data
X_train, y_train, scaler = prepare_data(df, 'Close', window_size)

# Build the model
model = build_lstm_model(X_train.shape)

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10)

# Get real-time stock price
real_time_price = get_real_time_stock_price(stock_symbol)
print(f"Real-time stock price for {stock_symbol}: {real_time_price}")

# Append real-time price to the DataFrame
df.loc[pd.Timestamp.today()] = [real_time_price, real_time_price, real_time_price, real_time_price, 0, 0]  # Assuming OHLC data same for real-time price

# Predict future stock prices
predicted_prices = predict_stock_prices(model, df['Close'], window_size, scaler, future_days)

# Create future dates
last_date = df.index[-1]
future_dates = pd.date_range(last_date, periods=future_days + 1).tolist()[1:]

# Plot actual vs predicted prices
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Close'], label='Actual Prices')
plt.plot(future_dates, predicted_prices, label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.legend()
plt.grid(True)
plt.show()

