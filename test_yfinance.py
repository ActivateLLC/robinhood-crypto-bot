# test_yfinance.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

try:
    symbol = 'BTC-USD'
    end_date = datetime.now()
    # Fetch slightly more to ensure we get enough points even with market closures/gaps
    start_date = end_date - timedelta(days=5) # Fetch 5 days of hourly data 
    print(f"Attempting to download {symbol} from {start_date.date()} to {end_date.date()} with interval '1h'")
    
    data = yf.download(
        symbol, 
        start=start_date, 
        end=end_date, 
        interval="1h", 
        progress=False
    )

    if data is None or data.empty:
        print("Failed: yfinance returned None or empty DataFrame.")
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date, interval="1h")
        if hist is None or hist.empty:
            print("Failed: yfinance ticker.history also returned None or empty DataFrame.")
        else:
             print(f"Success via ticker.history: Downloaded {len(hist)} rows.")

    else:
        print(f"Success via yf.download: Downloaded {len(data)} rows.")
        # print(data.head())
        # print(data.tail())
except Exception as e:
    import traceback
    print(f"Failed: Encountered exception: {e}")
    print(traceback.format_exc())
