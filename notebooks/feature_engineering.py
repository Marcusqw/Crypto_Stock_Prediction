import ta
import pandas as pd

def add_technical_indicators(data):
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    data['MACD'] = ta.trend.macd_diff(data['Close'])
    data['BollingerHigh'] = ta.volatility.bollinger_hband(data['Close'])
    data['BollingerLow'] = ta.volatility.bollinger_lband(data['Close'])
    data['SMA'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['EMA'] = ta.trend.ema_indicator(data['Close'], window=20)
    data.dropna(inplace=True)
    return data

def add_lagged_features(data, n_lags=5):
    for lag in range(1, n_lags + 1):
        data[f'Close_lag_{lag}'] = data['Close'].shift(lag)
    data.dropna(inplace=True)
    return data
