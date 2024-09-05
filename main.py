import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import datetime

def get_historical_data(ticker, period):
    # Fetch historical data from Yahoo Finance
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            raise ValueError(f"No data found for {ticker} with period {period}.")
        
        # Calculate technical indicators
        data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
        data['MACD'] = ta.trend.macd_diff(data['Close'])
        data['BollingerHigh'] = ta.volatility.bollinger_hband(data['Close'])
        data['BollingerLow'] = ta.volatility.bollinger_lband(data['Close'])
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def train_model(data):
    if data.empty:
        print("No data available to train the model.")
        return None, None

    # Features and Target
    X = data[['RSI', 'MACD', 'BollingerHigh', 'BollingerLow']]
    y = data['Close']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("Not enough data to perform train-test split.")
        return None, None
    
    # Scaling features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Model Training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    predictions = model.predict(X_test)
    
    # Model Evaluation
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
    
    return model, scaler

def predict_and_visualize(model, scaler, data, prediction_period, ticker):
    if model is None or scaler is None:
        print("Model or Scaler not available. Exiting prediction process.")
        return
    
    # Generate future dates
    last_date = data.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, prediction_period + 1)]
    
    # Prepare prediction data (e.g., using last known indicators)
    X_future = scaler.transform(data[['RSI', 'MACD', 'BollingerHigh', 'BollingerLow']].iloc[-1].values.reshape(1, -1))
    
    # Predicting Future Prices
    predictions = []
    for i in range(prediction_period):
        prediction = model.predict(X_future)
        predictions.append(prediction[0])
        # Update X_future with the predicted value (Assuming the indicators would slightly change)
        X_future = np.roll(X_future, -1)
        X_future[0][-1] = prediction

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label="Historical Prices")
    plt.plot(future_dates, predictions, label="Predicted Prices", linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f"Price Prediction for {ticker}")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ticker = input("Enter the stock/crypto ticker symbol: ")
    period = input("Enter the historical period (e.g., '1y', '6mo', '3mo', '1d'): ")
    prediction_period = input("Enter the prediction period (e.g., '1D', '1W', '1M', '3M', '1Y'): ")
    
    data = get_historical_data(ticker, period)
    model, scaler = train_model(data)
    
    prediction_days = {'1D': 1, '1W': 7, '1M': 30, '3M': 90, '1Y': 365}
    predict_and_visualize(model, scaler, data, prediction_days.get(prediction_period.upper(), 1), ticker)
