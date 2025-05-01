import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Union, Dict
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class AltCryptoDataProvider:
    """
    Alternative Cryptocurrency Data Provider
    Provides historical price data and market information
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the data provider
        
        Args:
            cache_dir (Optional[str]): Directory to cache downloaded data
        """
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '..', 'data_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def fetch_crypto_data(self, symbol: str = 'BTC-USD', days: int = 365, interval: str = '1h'):
        """
        Fetch cryptocurrency data with robust fallback mechanisms
        
        Args:
            symbol (str): Cryptocurrency trading pair
            days (int): Number of historical days to fetch
            interval (str): Data granularity
        
        Returns:
            pd.DataFrame: Processed market data
        """
        try:
            # Limit days to 730 to comply with yfinance restrictions
            days = min(days, 730)
            
            # Fetch data using yfinance
            df = yf.download(
                symbol, 
                period=f'{days}d', 
                interval=interval,
                progress=False,
                auto_adjust=False
            )
            
            if df.empty:
                raise ValueError("Empty dataset from yfinance")
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            return df
        
        except Exception as yf_error:
            logger.error(f"yfinance fetch failed for {symbol}: {yf_error}")
            raise ValueError(f"Unable to fetch data for {symbol}")
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to price data
        
        Args:
            df (pd.DataFrame): Input price data
        
        Returns:
            pd.DataFrame: Price data with added technical indicators
        """
        # Compute moving averages
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()
        
        # Compute returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(1 + df['returns'])
        
        # Compute RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Compute MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volatility indicator
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Bollinger Bands
        df['BB_Middle'] = df['MA_20']
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['Volatility']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['Volatility']
        
        return df
    
    def calculate_profit_percentage(self, historical_data: pd.DataFrame = None, symbol: str = 'BTC-USD') -> Dict[str, float]:
        """
        Calculate profit percentage and related metrics
        
        Args:
            historical_data (pd.DataFrame): Historical price data
            symbol (str): Trading pair symbol
        
        Returns:
            Dict[str, float]: Profit percentage and related metrics
        """
        try:
            # Fetch data if not provided
            if historical_data is None:
                historical_data = self.fetch_crypto_data(symbol, days=365, interval='1d')
            
            # Ensure data is not empty
            if historical_data.empty:
                return {
                    'total_profit_percentage': 0,
                    'annualized_return': 0,
                    'start_price': 0,
                    'end_price': 0,
                    'max_drawdown_percentage': 0
                }
            
            # Calculate basic metrics
            start_price = historical_data['Close'].iloc[0]
            end_price = historical_data['Close'].iloc[-1]
            
            # Total profit percentage
            total_profit_percentage = ((end_price - start_price) / start_price) * 100
            
            # Annualized return (assuming daily data)
            trading_days = len(historical_data)
            years = trading_days / 365
            annualized_return = (((end_price / start_price) ** (1/years)) - 1) * 100
            
            # Maximum drawdown calculation
            cumulative_max = historical_data['Close'].cummax()
            drawdown = (historical_data['Close'] - cumulative_max) / cumulative_max * 100
            max_drawdown_percentage = drawdown.min()
            
            return {
                'total_profit_percentage': total_profit_percentage,
                'annualized_return': annualized_return,
                'start_price': start_price,
                'end_price': end_price,
                'max_drawdown_percentage': max_drawdown_percentage
            }
        
        except Exception as e:
            print(f"Error calculating profit percentage for {symbol}: {e}")
            return {
                'total_profit_percentage': 0,
                'annualized_return': 0,
                'start_price': 0,
                'end_price': 0,
                'max_drawdown_percentage': 0
            }

    def get_market_sentiment(self, symbol: str) -> Dict[str, Union[float, str]]:
        """
        Estimate market sentiment based on technical indicators
        
        Args:
            symbol (str): Trading pair symbol
        
        Returns:
            Dict[str, Union[float, str]]: Market sentiment analysis
        """
        try:
            # Fetch recent data
            df = self.fetch_crypto_data(symbol, days=30, interval='1d')
            
            # Sentiment calculation based on technical indicators
            latest_close = df['Close'].iloc[-1]
            ma_50 = df['MA_50'].iloc[-1]
            ma_200 = df['MA_200'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            
            # Sentiment scoring
            sentiment_score = 0
            
            # Price vs Moving Averages
            if latest_close > ma_50 and latest_close > ma_200:
                sentiment_score += 1
            elif latest_close < ma_50 and latest_close < ma_200:
                sentiment_score -= 1
            
            # RSI Sentiment
            if rsi > 70:
                sentiment_score -= 0.5  # Overbought
            elif rsi < 30:
                sentiment_score += 0.5  # Oversold
            
            # MACD Sentiment
            if macd > 0:
                sentiment_score += 0.5
            else:
                sentiment_score -= 0.5
            
            # Categorize sentiment
            if sentiment_score > 1:
                sentiment = 'Bullish'
            elif sentiment_score < -1:
                sentiment = 'Bearish'
            else:
                sentiment = 'Neutral'
            
            return {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'sentiment': sentiment,
                'latest_price': latest_close,
                'rsi': rsi,
                'macd': macd
            }
        
        except Exception as e:
            print(f"Error computing market sentiment for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e)
            }

def main():
    """
    Demonstration of AltCryptoDataProvider
    """
    data_provider = AltCryptoDataProvider()
    
    # Fetch historical data
    historical_data = data_provider.fetch_crypto_data('BTC-USD', days=365, interval='1d')
    
    # Get market sentiment
    sentiment = data_provider.get_market_sentiment('BTC-USD')
    print("Market Sentiment:")
    print(sentiment)
    
    # Calculate profit percentage
    profit_metrics = data_provider.calculate_profit_percentage(historical_data)
    print("\nProfit Metrics:")
    for key, value in profit_metrics.items():
        print(f"{key.replace('_', ' ').title()}: {value:.2f}%")

if __name__ == "__main__":
    main()
