#!/usr/bin/env python3
"""
Alternative Cryptocurrency Data Provider

This module provides price data for cryptocurrencies not available on Robinhood.
It uses yfinance as the primary source and CoinGecko's API as a fallback to fetch historical price data for various tokens.
Also provides sentiment analysis based on news headlines.
"""

import os
import time
import logging
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
import sys
import yfinance as yf
from pycoingecko import CoinGeckoAPI
from config import YFINANCE_PERIOD, YFINANCE_INTERVAL
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# Imports for Sentiment Analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Download VADER lexicon if not already present (run once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    logger.info("Downloading VADER lexicon for sentiment analysis...")
    nltk.download('vader_lexicon', quiet=True)
except LookupError:
    logger.info("VADER lexicon lookup failed, attempting download...")
    try:
        nltk.download('vader_lexicon', quiet=True)
    except Exception as download_exc:
        logger.error(f"Failed to download VADER lexicon: {download_exc}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("alt_crypto_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Map common symbols to their CoinGecko IDs for efficiency and accuracy
# Add more as needed
COMMON_COINGECKO_IDS = {
    'btc': 'bitcoin',
    'eth': 'ethereum',
    'sol': 'solana',
    'bnb': 'binancecoin', # Added BNB
    # Add other frequently traded symbols here
}

class AltCryptoDataProvider:
    """
    A provider for cryptocurrency data not available on Robinhood.
    Uses yfinance as the primary source and CoinGecko API as a fallback to fetch historical price data.
    Also provides sentiment analysis based on news headlines.
    """
    
    def __init__(self, retries=3, delay=5):
        """Initialize the alternative cryptocurrency data provider."""
        self.cg = CoinGeckoAPI() # Keep CoinGecko instance for fallback
        self.retries = retries
        self.delay = delay
        logger.info("AltCryptoDataProvider initialized. Primary source: yfinance, Fallback: CoinGecko")
        self.base_url = "https://api.coingecko.com/api/v3"

        # Sentiment Analysis Initialization
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.news_api_key = os.getenv('NEWS_API_KEY')
        if not self.news_api_key:
            logger.warning("NEWS_API_KEY not found in .env file. Sentiment analysis will be disabled.")
            self.newsapi = None
        else:
            try:
                self.newsapi = NewsApiClient(api_key=self.news_api_key)
                logger.info("NewsApiClient initialized for sentiment analysis.")
            except Exception as e:
                logger.error(f"Failed to initialize NewsApiClient: {e}")
                self.newsapi = None

        self.coin_ids = {
            # Meme coins
            "POPCAT": "popcat",
            "SHIB": "shiba-inu",
            "PEPE": "pepe",
            "BONK": "bonk",
            "WIF": "dogwifhat",
            "FLOKI": "floki",
            "PENGU": "pengu",
            "PNUT": "peanut",
            "TRUMP": "trump-2024",

            # Major Cryptocurrencies (ensure these are mapped)
            "BTC": "bitcoin",
            "ETH": "ethereum",

            # Other cryptocurrencies on Robinhood
            "AAVE": "aave",
            "AVAX": "avalanche-2",
            "ADA": "cardano",
            "LINK": "chainlink",
            "COMP": "compound-governance-token",
            "ETC": "ethereum-classic",
            "DOGE": "dogecoin",  # Added Dogecoin
            "LTC": "litecoin", # Added Litecoin
            "SOL": "solana",   # Added Solana
            "XLM": "stellar",
            "XTZ": "tezos",
            "UNI": "uniswap",
            "USDC": "usd-coin",
            "XRP": "ripple"
        }
        self.price_history = {}
        
    def _format_symbol_for_yfinance(self, symbol: str) -> str:
        """Ensure symbol is in the format expected by yfinance (e.g., BTC)."""
        # Strip any potential extra parts like '-USD' if already present but handle base symbols
        base_symbol = symbol.split('-')[0].upper()
        yf_symbol = f"{base_symbol}-USD"
        logger.debug(f"Formatting symbol '{symbol}' to '{yf_symbol}' for yfinance.")
        return yf_symbol

    def _get_coin_id(self, symbol: str) -> Optional[str]:
        """Helper to get CoinGecko coin ID from symbol (e.g., BTC -> bitcoin).
           Checks a predefined map first, then searches CoinGecko list as fallback.
        """
        base_symbol = symbol.split('-')[0].lower()

        # 1. Check predefined map first
        if base_symbol in COMMON_COINGECKO_IDS:
            coin_id = COMMON_COINGECKO_IDS[base_symbol]
            logger.debug(f"Using predefined CoinGecko ID '{coin_id}' for symbol '{symbol}'")
            return coin_id

        # 2. Fallback: Search the full list (less efficient, prone to errors)
        logger.warning(f"Symbol '{base_symbol}' not in predefined map. Falling back to full CoinGecko list search.")
        try:
            # TODO: Consider caching this list to avoid frequent API calls
            coin_list = self.cg.get_coins_list()
            for coin in coin_list:
                # Prioritize exact ID match if symbol is ambiguous
                if coin['id'] == base_symbol:
                     logger.debug(f"Found exact CoinGecko ID match '{coin['id']}' for symbol '{symbol}'")
                     return coin['id']
                # Fallback to symbol match (less reliable)
                if coin['symbol'] == base_symbol:
                    logger.debug(f"Found CoinGecko ID '{coin['id']}' via symbol match for '{symbol}'")
                    return coin['id'] # Return the first symbol match found

            logger.warning(f"CoinGecko ID not found for symbol: {symbol} via list search.")
            return None
        except Exception as e:
            logger.error(f"Error during CoinGecko coin list search: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5), retry=retry_if_exception_type(Exception))
    def _fetch_with_yfinance(self, symbol: str, days: int, interval: str) -> Optional[pd.DataFrame]: 
        """Attempts to fetch data using yfinance, respecting config for period and using the passed interval."""
        yf_symbol = self._format_symbol_for_yfinance(symbol)
        # Calculate start/end date based on the 'days' parameter passed to the function
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        # Format dates for yfinance if needed (usually accepts datetime objects)
        # start_date_str = start_date.strftime('%Y-%m-%d')
        # end_date_str = end_date.strftime('%Y-%m-%d')

        # Use interval from parameter, do not use period from config
        # fetch_period = YFINANCE_PERIOD # Don't use config period
        fetch_interval = interval # Use the passed interval

        logger.info(f"Fetching price history for {yf_symbol} from yfinance (Start: {start_date.date()}, End: {end_date.date()}, Interval: {fetch_interval})...")
        
        try:
            ticker = yf.Ticker(yf_symbol)
            # Use start/end dates derived from 'days' and the passed interval
            hist = ticker.history(start=start_date, end=end_date, interval=fetch_interval)

            if hist.empty:
                logger.warning(f"yfinance returned no data for {yf_symbol} (Start: {start_date.date()}, End: {end_date.date()}, Interval: {fetch_interval}).")
                # Continue to next retry if empty
                raise Exception("yfinance returned empty data")

            # Select and rename columns
            hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']]
            hist.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)

            # Ensure index is DatetimeIndex and timezone-naive
            if isinstance(hist.index, pd.DatetimeIndex):
                if hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)
            else:
                 logger.warning(f"yfinance index for {yf_symbol} is not DatetimeIndex. Attempting conversion.")
                 try:
                     hist.index = pd.to_datetime(hist.index).tz_localize(None)
                 except Exception as conv_err:
                     logger.error(f"Failed to convert yfinance index to DatetimeIndex for {yf_symbol}: {conv_err}")
                     raise Exception("Failed to convert yfinance index to DatetimeIndex")

            # Drop rows with NaN in critical columns
            hist.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

            if hist.empty:
                logger.warning(f"No valid OHLC data for {yf_symbol} from yfinance after cleaning.")
                raise Exception("No valid OHLC data for symbol after cleaning")

            logger.info(f"Successfully fetched {len(hist)} data points for {yf_symbol} from yfinance.")
            return hist # Return successfully fetched data

        except Exception as e:
            logger.error(f"Failed to fetch data for {yf_symbol} from yfinance: {e}")
            raise Exception("Failed to fetch data from yfinance")

    def _fetch_with_coingecko(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
         """Attempts to fetch data using CoinGecko (fallback)."""
         coin_id = self._get_coin_id(symbol)
         if not coin_id:
             logger.error(f"Could not find CoinGecko ID for symbol {symbol}. Cannot fetch data.")
             return None

         # Determine valid CoinGecko 'days' parameter (must be 1, 7, 14, 30, 90, 180, 365, 'max')
         valid_cg_days = [1, 7, 14, 30, 90, 180, 365]
         cg_days = days # Default if days is already valid or for 'max'
         if days not in valid_cg_days and days <= 365:
             # Find the smallest valid period >= requested days
             for valid_day in valid_cg_days:
                 if valid_day >= days:
                     cg_days = valid_day
                     break
             else: # If days > 365, consider using 'max'? For now, stick to 365 as max specific value
                 cg_days = 365 
                 logger.warning(f"Requested days ({days}) > 365, using CoinGecko max specific period 365 days.")
         elif days > 365:
             # Or potentially use 'max' if pycoingecko supports it, otherwise cap at 365
             cg_days = 365 # Sticking to 365 for simplicity now
             logger.warning(f"Requested days ({days}) > 365, using CoinGecko max specific period 365 days.")

         logger.info(f"Attempting fallback: Fetching price history for {symbol} ({coin_id}) from CoinGecko (requesting {cg_days} days for {days} required days)...")
         for attempt in range(self.retries):
             try:
                 # Fetch OHLC data using the adjusted cg_days
                 ohlc_data = self.cg.get_coin_ohlc_by_id(id=coin_id, vs_currency='usd', days=cg_days)
                 # Fetch volume data separately (CoinGecko API structure)
                 # Also use cg_days here for consistency, interval remains daily as market chart is daily beyond short periods
                 market_chart_data = self.cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=cg_days, interval='daily')

                 if not ohlc_data:
                     logger.warning(f"CoinGecko returned no OHLC data for {coin_id} (Attempt {attempt + 1}/{self.retries})")
                     time.sleep(self.delay)
                     continue

                 # Process OHLC data
                 df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                 df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                 df.set_index('timestamp', inplace=True)
                 df.index = df.index.normalize() # Normalize to midnight

                 # Process Volume data
                 if market_chart_data and 'total_volumes' in market_chart_data:
                     vol_df = pd.DataFrame(market_chart_data['total_volumes'], columns=['timestamp', 'volume'])
                     vol_df['timestamp'] = pd.to_datetime(vol_df['timestamp'], unit='ms')
                     vol_df.set_index('timestamp', inplace=True)
                     vol_df.index = vol_df.index.normalize() # Normalize to midnight
                     # Join volume data
                     df = df.join(vol_df['volume'], how='left')
                 else:
                     logger.warning(f"Could not fetch volume data for {coin_id} from CoinGecko.")
                     df['volume'] = 0 # Assign default volume if fetch fails

                 # Address FutureWarning: Avoid inplace on potentially copied series/slice
                 df['volume'] = df['volume'].fillna(0)

                 # Ensure correct data types
                 for col in ['open', 'high', 'low', 'close', 'volume']:
                     df[col] = pd.to_numeric(df[col], errors='coerce')

                 df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

                 if df.empty:
                     logger.warning(f"No valid OHLC data for {coin_id} from CoinGecko after cleaning (Attempt {attempt + 1}/{self.retries})")
                     time.sleep(self.delay)
                     continue

                 # Trim the dataframe to the originally requested number of days before returning
                 # Use .last() which works well with DatetimeIndex if frequency can be inferred, 
                 # otherwise, slice based on date after ensuring index is sorted.
                 df.sort_index(inplace=True)
                 # Use tail to get the last 'days' number of rows reliably
                 df = df.tail(days)

                 if len(df) < days:
                     logger.warning(f"CoinGecko data for {coin_id} has only {len(df)} points after requesting {cg_days} and taking tail({days}).")

                 logger.info(f"Successfully fetched {len(df)} data points for {coin_id} from CoinGecko.")
                 return df

             except Exception as e:
                 logger.error(f"Error fetching data from CoinGecko for {coin_id} (Attempt {attempt + 1}/{self.retries}): {e}")
                 # Check for specific rate limit error (adjust based on actual pycoingecko exception/response)
                 if '429' in str(e):
                     logger.warning(f"Rate limit likely exceeded for CoinGecko. Waiting {self.delay * (attempt + 1)}s...")
                     time.sleep(self.delay * (attempt + 1)) # Exponential backoff might be better
                 else:
                     time.sleep(self.delay)

         logger.error(f"Failed to fetch data from CoinGecko for {coin_id} after {self.retries} attempts.")
         # Optionally generate placeholder data if needed by the bot
         # logger.info(f"Generating placeholder price history for {symbol}...")
         # return self._generate_placeholder_data(days)
         return None

    def fetch_all_historical_data(self, symbol: str, interval: str = '1h', days: int = 30) -> pd.DataFrame:
        """
        Fetches historical data for a symbol, with optional days parameter.
        
        Args:
            symbol (str): Cryptocurrency symbol
            interval (str, optional): Interval for data points. Defaults to '1h'.
            days (int, optional): Number of days of historical data. Defaults to 30.
        
        Returns:
            pd.DataFrame: Historical price data
        """
        logger.info(f"Fetching price history for {symbol} from yfinance (Period: {days} days, Interval: {interval})...")
        
        try:
            # Use yfinance to fetch data
            yf_symbol = self._format_symbol_for_yfinance(symbol)
            df = self._fetch_with_yfinance(yf_symbol, days, interval)
            
            if df is None or df.empty:
                logger.warning(f"No data fetched from yfinance for {symbol}. Attempting CoinGecko fallback.")
                df = self._fetch_with_coingecko(symbol, days)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return self.generate_placeholder_data(symbol, days, interval)

    def fetch_price_history(
        self, symbol: str, days: int, interval: str = "1h" # Add interval parameter with default
    ) -> Optional[pd.DataFrame]:
        """Fetches historical OHLCV data using yfinance first, then CoinGecko as fallback."""

        df = None # Initialize df to None
        try:
            # Attempt 1: yfinance
            logger.info(f"Attempting to fetch data for {symbol} via yfinance...")
            df = self._fetch_with_yfinance(symbol, days, interval) # Pass interval here
            if df is not None and not df.empty:
                logger.info(f"Successfully fetched data for {symbol} via yfinance.")
                return df # Return if successful
            else:
                # Log if yfinance returned None or empty without raising an exception (shouldn't happen with current _fetch_with_yfinance)
                logger.warning(f"yfinance fetch for {symbol} returned None or empty DataFrame.")
            
        except Exception as yf_error:
            # Catch RetryError or other exceptions from _fetch_with_yfinance
            logger.error(f"yfinance fetch attempt failed for {symbol} after retries: {yf_error}")
            # df remains None, allowing fallback

        # If yfinance failed (df is None or empty), try CoinGecko
        logger.warning(f"yfinance fetch failed or returned empty data for {symbol}. Falling back to CoinGecko.")

        # Attempt 2: CoinGecko (Fallback) - Note: CoinGecko fallback might not respect the interval
        df_fallback = self._fetch_with_coingecko(symbol, days)
        if df_fallback is not None and not df_fallback.empty:
            return df_fallback

        logger.error(f"Failed to fetch data for {symbol} from both yfinance and CoinGecko.")
        return None

    def get_current_price(self, symbol: str) -> float:
        """
        Get the current price for a cryptocurrency.
        
        Args:
            symbol: The cryptocurrency symbol
            
        Returns:
            Current price in USD
        """
        # Check if we have price history for this symbol
        if symbol in self.price_history:
            # Return the most recent close price
            return self.price_history[symbol].iloc[-1]['close']
        
        # If not, fetch price history and return the most recent price
        df = self.fetch_price_history(symbol, days=1)
        if not df.empty:
            return df.iloc[-1]['close']
        
        # If all else fails, generate a placeholder price
        if symbol == "BTC":
            return 65000.0 + np.random.normal(0, 1000)
        elif symbol == "ETH":
            return 3500.0 + np.random.normal(0, 100)
        elif symbol == "DOGE":
            return 0.15 + np.random.normal(0, 0.01)
        elif symbol == "LTC":
            return 80.0 + np.random.normal(0, 2)
        elif symbol == "SOL":
            return 150.0 + np.random.normal(0, 5)
        else:
            # For other symbols, use the _generate_placeholder_data method
            df = self.generate_placeholder_data(symbol, days=1)
            return df.iloc[-1]['close']
    
    def get_portfolio_holdings(self) -> List[Dict[str, Any]]:
        """
        Get simulated portfolio holdings.
        
        Returns:
            List of dictionaries containing portfolio holdings
        """
        # Create simulated portfolio with popular cryptocurrencies
        holdings = [
            {"symbol": "BTC", "quantity": 0.05, "cost_basis": 60000.0},
            {"symbol": "ETH", "quantity": 1.2, "cost_basis": 3000.0},
            {"symbol": "DOGE", "quantity": 1000.0, "cost_basis": 0.12},
            {"symbol": "SOL", "quantity": 5.0, "cost_basis": 120.0},
            {"symbol": "SHIB", "quantity": 10000000.0, "cost_basis": 0.00001},
            {"symbol": "PEPE", "quantity": 5000000.0, "cost_basis": 0.0000008},
            {"symbol": "POPCAT", "quantity": 1000000.0, "cost_basis": 0.00004}
        ]
        
        # Calculate current value for each holding
        for holding in holdings:
            holding["current_price"] = self.get_current_price(holding["symbol"])
            holding["value"] = holding["quantity"] * holding["current_price"]
            holding["profit_loss"] = holding["value"] - (holding["quantity"] * holding["cost_basis"])
            holding["profit_loss_percent"] = (holding["profit_loss"] / (holding["quantity"] * holding["cost_basis"])) * 100
        
        return holdings

    def get_current_sentiment_score(self, symbol: str) -> float:
        """Fetches recent news for the symbol and calculates an average sentiment score.

        Args:
            symbol (str): The cryptocurrency symbol (e.g., 'BTC-USD', 'ETH').

        Returns:
            float: Average VADER compound sentiment score (-1.0 to +1.0), or 0.0 if unavailable.
        """
        if not self.newsapi:
            logger.debug("Sentiment analysis disabled (NewsAPI client not initialized).")
            return 0.0

        base_symbol = symbol.split('-')[0].upper()
        coin_name = COMMON_COINGECKO_IDS.get(base_symbol.lower(), base_symbol) # Get full name if possible
        query = f'{coin_name} OR {base_symbol} crypto OR cryptocurrency {base_symbol}'
        logger.debug(f"Fetching news for sentiment query: '{query}'")

        try:
            # Fetch news from the last 24 hours
            from_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S') # Removed 'Z'
            all_articles = self.newsapi.get_everything(
                q=query,
                language='en',
                sort_by='publishedAt', # Focus on recent news
                from_param=from_date,
                page_size=50 # Get a decent number of recent articles
            )

            if not all_articles or all_articles.get('status') != 'ok' or all_articles.get('totalResults', 0) == 0:
                logger.warning(f"No recent news found for query: '{query}'")
                return 0.0

            scores = []
            for article in all_articles.get('articles', []):
                title = article.get('title')
                description = article.get('description')
                text_to_analyze = f"{title}. {description}" if title and description else title or description

                if text_to_analyze:
                    vs = self.sentiment_analyzer.polarity_scores(text_to_analyze)
                    scores.append(vs['compound'])

            if not scores:
                logger.debug(f"No analyzable headlines found for query: '{query}'")
                return 0.0

            average_score = np.mean(scores)
            logger.info(f"Calculated average sentiment for {symbol}: {average_score:.4f} from {len(scores)} articles.")
            # Clip score to be strictly between -1 and 1
            return float(np.clip(average_score, -1.0, 1.0))

        except Exception as e:
            # Handle potential NewsAPI errors (rate limits, invalid key, etc.)
            logger.error(f"Error fetching or analyzing news sentiment for {symbol}: {e}")
            return 0.0 # Return neutral sentiment on error

    def generate_placeholder_data(self, symbol: str = 'BTC-USD', days: int = 30, interval: str = '1h') -> pd.DataFrame:
        """
        Generates placeholder data if fetching fails.
        
        Args:
            symbol (str): Cryptocurrency symbol
            days (int): Number of days of data to generate
            interval (str): Interval between data points
        
        Returns:
            pd.DataFrame: Synthetic price data
        """
        logger.warning(f"Generating placeholder data for {symbol}")
        
        # Determine base parameters
        base_price = {
            'BTC-USD': 65000.0,
            'ETH-USD': 3500.0,
            'DOGE-USD': 0.15,
            'SOL-USD': 150.0
        }.get(symbol, 100.0)
        
        # Calculate number of intervals
        intervals_per_day = {'1m': 1440, '5m': 288, '15m': 96, '30m': 48, '1h': 24, '4h': 6, '1d': 1}
        num_intervals = days * intervals_per_day.get(interval, 24)
        
        # Generate synthetic time series
        times = pd.date_range(end=pd.Timestamp.now(), periods=num_intervals, freq=interval)
        
        # Synthetic price generation with slight randomness and trend
        np.random.seed(42)  # For reproducibility
        trend = np.linspace(0, 0.2, num_intervals)  # Slight upward trend
        noise = np.random.normal(0, base_price * 0.02, num_intervals)
        
        close_prices = base_price * (1 + trend) + noise
        open_prices = close_prices * (1 + np.random.uniform(-0.005, 0.005, num_intervals))
        high_prices = np.maximum(close_prices, open_prices) * (1 + np.random.uniform(0, 0.01, num_intervals))
        low_prices = np.minimum(close_prices, open_prices) * (1 - np.random.uniform(0, 0.01, num_intervals))
        
        # Ensure positive volumes
        volumes = np.abs(np.random.uniform(1000, 100000, num_intervals))
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': times,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
        
        df.set_index('timestamp', inplace=True)
        
        return df


# Example usage
if __name__ == "__main__":
    provider = AltCryptoDataProvider()
    
    # Test with some meme coins
    if len(sys.argv) > 1:
        test_symbols = sys.argv[1:]
    else:
        test_symbols = ["POPCAT", "SHIB", "PEPE"]
    
    for symbol in test_symbols:
        try:
            # Fetch price history
            df = provider.fetch_price_history(symbol, days=30)
            
            # Get current price
            current_price = provider.get_current_price(symbol)
            
            # Print information
            print(f"{symbol} current price: ${current_price:.8f}")
            print(f"{symbol} price history shape: {df.shape}")
            print(f"{symbol} last 5 days:")
            print(df.tail())
            print()
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            print()

    # Test portfolio holdings
    holdings = provider.get_portfolio_holdings()
    for holding in holdings:
        print(f"{holding['symbol']}:")
        print(f"  Quantity: {holding['quantity']}")
        print(f"  Cost Basis: ${holding['cost_basis']:.2f}")
        print(f"  Current Price: ${holding['current_price']:.8f}")
        print(f"  Value: ${holding['value']:.2f}")
        print(f"  Profit/Loss: ${holding['profit_loss']:.2f} ({holding['profit_loss_percent']:.2f}%)")
        print()