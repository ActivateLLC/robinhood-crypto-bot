import ccxt
import pandas as pd
from datetime import datetime

# Initialize the exchange (e.g., Binance)
# TODO: Make exchange configurable
exchange = ccxt.binanceus({
    'enableRateLimit': True, # Enable rate limiting
    # Add API keys here if needed for private data, otherwise fetches public data
    # 'apiKey': 'YOUR_API_KEY',
    # 'secret': 'YOUR_SECRET',
})

def fetch_ohlcv(symbol: str, timeframe: str = '1h', since: str = None, limit: int = 500) -> pd.DataFrame:
    """Fetches historical OHLCV data for a given symbol and timeframe.

    Args:
        symbol: The trading pair symbol (e.g., 'BTC/USD').
        timeframe: The timeframe string (e.g., '1m', '5m', '1h', '1d').
        since: Start date string (e.g., '2023-01-01T00:00:00Z') or timestamp in milliseconds.
        limit: The maximum number of candles to fetch.

    Returns:
        A pandas DataFrame with OHLCV data, indexed by datetime.
    """
    print(f"Fetching {symbol} {timeframe} data...")

    since_timestamp = None
    if since:
        # Convert date string to timestamp milliseconds if necessary
        if isinstance(since, str):
            try:
                since_dt = datetime.strptime(since, '%Y-%m-%dT%H:%M:%SZ')
                since_timestamp = int(since_dt.timestamp() * 1000)
            except ValueError:
                print("Invalid 'since' date format. Use 'YYYY-MM-DDTHH:MM:SSZ'.")
                return pd.DataFrame()
        elif isinstance(since, int):
            since_timestamp = since # Assume it's already in milliseconds

    try:
        # Check if the market exists
        markets = exchange.load_markets()
        if symbol not in markets:
            print(f"Error: Symbol {symbol} not found on {exchange.id}")
            # Suggest available symbols or handle error appropriately
            # print("Available symbols:", list(markets.keys()))
            return pd.DataFrame()

        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=limit)

        if not ohlcv:
            print("No data returned from exchange.")
            return pd.DataFrame()

        # Convert to pandas DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Convert timestamp to datetime and set as index
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)

        print(f"Successfully fetched {len(df)} data points for {symbol}.")
        return df

    except ccxt.NetworkError as e:
        print(f"Network Error fetching data: {e}")
    except ccxt.ExchangeError as e:
        print(f"Exchange Error fetching data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return pd.DataFrame() # Return empty DataFrame on error

# Example usage:
if __name__ == '__main__':
    # Fetch last 500 hours of BTC/USD data
    btc_data = fetch_ohlcv('BTC/USD', timeframe='1h', limit=500)
    if not btc_data.empty:
        print("BTC/USD 1h Data:")
        print(btc_data.head())
        print(btc_data.tail())

    # Fetch data since a specific date
    eth_data = fetch_ohlcv('ETH/USD', timeframe='1d', since='2024-01-01T00:00:00Z')
    if not eth_data.empty:
        print("\nETH/USD 1d Data since 2024-01-01:")
        print(eth_data.head())
        print(eth_data.tail())
