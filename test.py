import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import datetime
import plotly.graph_objects as go
from xgboost import XGBRegressor  # Using XGBoost for better performance

def get_historical_data(ticker, period):
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            raise ValueError(f"No data found for {ticker} with period {period}.")
        
        # Calculate technical indicators
        data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
        data['MACD'] = ta.trend.macd_diff(data['Close'])
        data['BollingerHigh'] = ta.volatility.bollinger_hband(data['Close'])
        data['BollingerLow'] = ta.volatility.bollinger_lband(data['Close'])
        
        # Add more features
        data['SMA'] = ta.trend.sma_indicator(data['Close'], window=20)
        data['EMA'] = ta.trend.ema_indicator(data['Close'], window=20)
        data['Momentum'] = ta.momentum.awesome_oscillator(data['High'], data['Low'])
        
        # Add lagged features (previous days' prices)
        data['Lag1'] = data['Close'].shift(1)
        data['Lag2'] = data['Close'].shift(2)
        data['Lag3'] = data['Close'].shift(3)
        
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
    X = data.drop(['Close'], axis=1)
    y = data['Close']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("Not enough data to perform train-test split.")
        return None, None
    
    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize XGBRegressor for better performance
    model = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=5, random_state=42)
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Model Evaluation
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    return model, scaler, X_test, y_test, y_pred  # Return additional data for plotting

def predict_and_visualize(model, scaler, data, prediction_period, ticker, X_test, y_test, y_pred):
    if model is None or scaler is None:
        print("Model or Scaler not available. Exiting prediction process.")
        return
    
    # Generate future dates
    last_date = data.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, prediction_period + 1)]
    
    # Prepare prediction data (using last known indicators)
    X_future = scaler.transform(data.drop(['Close'], axis=1).iloc[-1].values.reshape(1, -1))
    
    # Predicting Future Prices
    predictions = []
    for i in range(prediction_period):
        prediction = model.predict(X_future)
        predictions.append(prediction[0])
        # Update X_future with the predicted value (simple assumption)
        X_future = np.roll(X_future, -1)
        X_future[0][-1] = prediction
    
    # Print predicted values
    print("Predicted Prices for the next", prediction_period, "days:")
    for i, price in enumerate(predictions):
        print(f"Day {i + 1}: {price}")

    # Create interactive plot with Plotly
    fig = go.Figure()

    # Add historical prices
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name="Historical Prices"))

    # Add predicted prices
    fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines+markers', name="Predicted Prices"))

    # Show plot with dates when hovering
    fig.update_layout(
        title=f"Price Prediction for {ticker}",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x'
    )
    
    # Show graph
    fig.show()

if __name__ == "__main__":
    ticker = input("Enter the stock/crypto ticker symbol: ").upper()
    period = input("Enter the historical period (e.g., '1y', '6mo', '3mo', '1d'): ")
    prediction_period_input = input("Enter the prediction period (e.g., '1D', '1W', '1M', '3M', '1Y'): ").upper()
    
    data = get_historical_data(ticker, period)
    model, scaler, X_test, y_test, y_pred = train_model(data)
    
    # Mapping prediction periods to number of days
    prediction_days_map = {
        '1D': 1,
        '1W': 7,
        '1M': 30,
        '3M': 90,
        '1Y': 365
    }
    
    prediction_period = prediction_days_map.get(prediction_period_input, 1)  # Default to 1 day if invalid input
    predict_and_visualize(model, scaler, data, prediction_period, ticker, X_test, y_test, y_pred)
