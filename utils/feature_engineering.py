import pandas as pd

def add_technical_indicators(df):
    # Adding a 7-day Moving Average
    df['MA7'] = df['Close'].rolling(window=7).mean()
    
    # Adding a 21-day Moving Average
    df['MA21'] = df['Close'].rolling(window=21).mean()
    
    # Adding Relative Strength Index (RSI)
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Adding Bollinger Bands
    df['Upper_BB'] = df['MA21'] + 2 * df['Close'].rolling(window=21).std()
    df['Lower_BB'] = df['MA21'] - 2 * df['Close'].rolling(window=21).std()
    
    return df
