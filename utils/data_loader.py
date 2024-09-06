import yfinance as yf
import pandas as pd

def get_historical_data(ticker, period):
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            raise ValueError(f"No data found for {ticker} with period {period}.")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
