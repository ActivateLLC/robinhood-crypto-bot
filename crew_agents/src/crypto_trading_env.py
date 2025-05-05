import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
import ccxt
from typing import Dict, Any, Tuple, Optional, Literal
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from decimal import Decimal
import importlib
import json

# Add project root to sys.path if necessary
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from alt_crypto_data import AltCryptoDataProvider # Import updated provider from root

# Import Broker Interfaces/Implementations

try:
    brokers_base = importlib.import_module("brokers.base_broker")
    Broker = brokers_base.BrokerInterface # Use BrokerInterface
    MarketData = brokers_base.MarketData # Also import MarketData if needed
except ImportError as e:
    print(f"ERROR: Failed to import brokers.base_broker: {e}")
    raise
except AttributeError as e:
    print(f"ERROR: Attribute not found in brokers.base_broker: {e}") # Add specific error handling
    raise

try:
    brokers_robinhood = importlib.import_module("brokers.robinhood")
    RobinhoodBroker = brokers_robinhood.RobinhoodBroker
except ImportError as e:
    print(f"ERROR: Failed to import brokers.robinhood: {e}")
    raise

try:
    brokers_ibkr = importlib.import_module("brokers.ibkr")
    IBKRBroker = brokers_ibkr.IBKRBroker
except ImportError as e:
    # Log warning instead of error if IBKR is optional/not fully implemented
    print(f"WARNING: Failed to import brokers.ibkr: {e}. Continuing without IBKR support.")
    IBKRBroker = None # Define as None if import fails

# Initialize logger
logging.basicConfig(level=logging.INFO)

# Define the experience log file path relative to this script
EXPERIENCE_LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
EXPERIENCE_LOG_FILE = os.path.join(EXPERIENCE_LOG_DIR, 'live_experience.jsonl')

# Ensure the log directory exists
os.makedirs(EXPERIENCE_LOG_DIR, exist_ok=True)

class CryptoTradingEnvironment(gym.Env):
    """
    Cryptocurrency Trading Environment for Reinforcement Learning
    Implements Gymnasium Env interface for training trading agents
    """
    def __init__(
        self, 
        data_provider: AltCryptoDataProvider, # Expecting the updated provider
        symbol: str = 'BTC-USD',
        initial_capital: float = 10000.0,  # Default set here
        lookback_window: int = 24,  # 24 hours of history
        trading_fee: float = 0.0005,  # Reduced fee for more frequent trading
        df: Optional[pd.DataFrame] = None,
        live_trading: bool = False,
        broker_type: Literal['simulation', 'robinhood', 'ibkr'] = 'simulation'
    ):
        super().__init__()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Core Parameters
        self.symbol = symbol
        self.data_provider = data_provider
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window
        self.trading_fee = trading_fee 
        self.live_trading = live_trading
        self.broker_type = broker_type if live_trading else 'simulation'
        self.broker: Optional[Broker] = None 

        self.logger.info(f"Initializing environment: Symbol={self.symbol}, Live Trading={self.live_trading}, Broker Type={self.broker_type}")

        # --- Live Trading Setup ---
        if self.live_trading:
            load_dotenv() 
            if self.broker_type == 'robinhood':
                api_key = os.getenv("RH_API_KEY")
                private_key = os.getenv("RH_PRIVATE_KEY")
                if not api_key or not private_key:
                    self.logger.error("Robinhood API Key or Private Key not found in .env file for live trading.")
                    raise ValueError("Missing Robinhood credentials for live trading.")
                try:
                    self.broker = RobinhoodBroker(api_key=api_key, private_key=private_key)
                    self.logger.info("Robinhood broker initialized successfully for live trading.")
                    # Fetch initial live state (capital, holdings) in reset()
                    # Override trading fee with broker's fee if applicable
                    # self.trading_fee = self.broker.get_trading_fee(self.symbol) 
                except Exception as e:
                    self.logger.error(f"Failed to initialize Robinhood broker: {e}", exc_info=True)
                    raise
            # elif self.broker_type == 'ibkr':
                # TODO: Add IBKR broker initialization
                # pass 
            else:
                self.logger.error(f"Unsupported broker type '{self.broker_type}' for live trading.")
                raise ValueError(f"Unsupported live broker: {self.broker_type}")
        # --------------------------

        # Action space: 0=Hold, 1=Buy 25%, 2=Buy 50%, 3=Buy 100%, 4=Sell 25%, 5=Sell 50%, 6=Sell 100%
        self.action_space = gym.spaces.Discrete(7)
        
        # Observation space: Includes market indicators + portfolio state (capital, holdings)
        # Shape needs to accommodate the 11 original indicators + 2 portfolio features + 1 sentiment
        # TODO: Potentially adjust shape/content based on live data availability
        observation_shape = (14,) # Correct shape accounting for sentiment
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(observation_shape,), 
            dtype=np.float32
        )
        
        # Data handling
        self.historical_data = None
        if not self.live_trading: 
            if df is not None:
                self.historical_data = df
            else:
                self.historical_data = self._fetch_historical_data()
            if self.historical_data is None or self.historical_data.empty:
                 self.logger.error("Failed to load historical data for simulation.")
                 raise ValueError("Historical data could not be loaded.")
            self.logger.info(f"Loaded historical data for simulation: {len(self.historical_data)} points.")
        else:
            # In live mode, we might fetch a small recent window in reset/step if needed for indicators
            self.logger.info("Live trading mode: Historical data will be fetched dynamically as needed.")
        
        # Trading state variables (will be updated from broker in live mode during reset)
        self.current_step = 0
        self.current_capital = initial_capital
        self.crypto_holdings = 0
        self.portfolio_value = initial_capital
        self.max_portfolio_value = initial_capital
        self.max_drawdown = 0.2  

        # --- Additions for Differential Sharpe Ratio ---
        self.sharpe_window = 100  
        self.risk_free_rate = 0.0 
        self.portfolio_history = [initial_capital] 
        self.previous_sharpe_ratio = 0.0
        # ---------------------------------------------
        
        # Prepare an initial observation structure (will be populated in reset)
        self._initial_observation = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        # Ensure initial state is set correctly in reset()
        # self.reset() 

        self.experience_log_file = EXPERIENCE_LOG_FILE

    def _fetch_historical_data(self) -> pd.DataFrame:
        """
        Advanced historical data fetching with comprehensive fallback

        Returns:
            pd.DataFrame: Validated historical price data
        """
        # Priority-ordered data sources with specific handling
        data_sources = [
            ('yfinance', self._fetch_yfinance_data),
            ('CCXT Exchanges', self._fetch_ccxt_data),
            ('Simulated', self._generate_simulated_data)
        ]

        from crew_agents.src.error_monitoring_agent import AdvancedErrorMonitoringAgent
        error_monitor = AdvancedErrorMonitoringAgent()

        for source_name, source_method in data_sources:
            try:
                data = source_method()
                if not data.empty and len(data) > 0:
                    return data
            except Exception as e:
                error_monitor.logger.warning(f"Data fetch failed for {source_name}: {str(e)}")
        
        # If all sources fail, generate simulated data
        return self._generate_simulated_data()
    
    def _fetch_yfinance_data(self) -> pd.DataFrame:
        """
        Fetch data using yfinance with specific configuration
        
        Returns:
            pd.DataFrame: Historical price data
        """
        self.logger.info(f"Fetching {self.symbol} data from yfinance...")
        ticker = yf.Ticker(self.symbol)
        try:
            # Fetch data for the last 729 days with 1-hour interval to stay within yfinance limits
            data = ticker.history(period='729d', interval='1h', auto_adjust=True)
            
            if data.empty:
                self.logger.warning("yfinance returned an empty dataset.")
            
            # Rename columns to standard format
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Add additional technical indicators
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(1 + data['returns'])
            
            return data
        
        except Exception as e:
            self.logger.error(f"yfinance data fetch error: {e}")
            raise
    
    def _fetch_ccxt_data(self) -> pd.DataFrame:
        """
        Fetch data from CCXT exchanges with robust error handling
        
        Returns:
            pd.DataFrame: Historical price data
        """
        exchanges = [
            ccxt.coinbase,   # Primary exchange
            ccxt.gateio,     # Alternative exchange
            ccxt.kraken      # Backup exchange
        ]
        
        for ExchangeClass in exchanges:
            try:
                exchange = ExchangeClass()
                
                # Use 'BTC/USDT' for most exchanges
                ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=2000)
                
                if ohlcv:
                    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                    data.set_index('timestamp', inplace=True)
                    
                    return data
            
            except Exception as exchange_error:
                self.logger.warning(f"Data fetch error from {ExchangeClass.__name__}: {exchange_error}")
        
        raise ValueError("Unable to fetch data from CCXT exchanges")
    
    def _fetch_coingecko_data(self) -> pd.DataFrame:
        """
        Fetch historical data using CoinGecko API
        
        Returns:
            pd.DataFrame: Historical price data
        """
        import requests
        
        # Extract base cryptocurrency from symbol
        base_crypto = self.symbol.split('-')[0]
        
        url = f"https://api.coingecko.com/api/v3/coins/{base_crypto.lower()}/market_chart?vs_currency=usd&days=730"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Extract price data
            prices = pd.DataFrame(data['prices'], columns=['Timestamp', 'Close'])
            prices['Timestamp'] = pd.to_datetime(prices['Timestamp'], unit='ms')
            prices.set_index('Timestamp', inplace=True)
            
            # Add additional columns
            prices['Open'] = prices['Close'].shift(1)
            prices['High'] = prices['Close'].rolling(window=5).max()
            prices['Low'] = prices['Close'].rolling(window=5).min()
            prices['Volume'] = 0  # CoinGecko doesn't provide volume in this endpoint
            
            return prices
        
        except Exception as e:
            self.logger.error(f"CoinGecko data fetch error: {e}")
            raise
    
    def _fetch_alphavantage_data(self) -> pd.DataFrame:
        """
        Fetch historical data using Alpha Vantage API
        
        Returns:
            pd.DataFrame: Historical price data
        """
        import requests
        
        # Retrieve API key from environment
        api_key = os.getenv('ALPHAVANTAGE_API_KEY')
        if not api_key:
            raise ValueError("Alpha Vantage API key not found")
        
        base_crypto = self.symbol.split('-')[0]
        
        url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={base_crypto}&market=USD&apikey={api_key}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Extract time series data
            time_series = data.get('Time Series (Digital Currency Daily)', {})
            
            if not time_series:
                raise ValueError("No time series data available")
            
            prices = []
            for date, values in time_series.items():
                prices.append({
                    'Timestamp': pd.to_datetime(date),
                    'Open': float(values['1a. open (USD)']),
                    'High': float(values['2a. high (USD)']),
                    'Low': float(values['3a. low (USD)']),
                    'Close': float(values['4a. close (USD)']),
                    'Volume': float(values['5. volume'])
                })
            
            prices_df = pd.DataFrame(prices)
            prices_df.set_index('Timestamp', inplace=True)
            prices_df.sort_index(inplace=True)
            
            return prices_df
        
        except Exception as e:
            self.logger.error(f"Alpha Vantage data fetch error: {e}")
            raise
    
    def _fetch_ccxt_data_advanced(self) -> pd.DataFrame:
        """
        Advanced CCXT data fetching with multiple exchange fallback
        
        Returns:
            pd.DataFrame: Historical price data
        """
        exchanges = [
            'coinbase', 'kraken', 
            'kucoin', 'bitfinex', 'gemini'
        ]
        
        for exchange_id in exchanges:
            try:
                exchange = getattr(ccxt, exchange_id)()
                exchange.load_markets()
                
                # Fetch OHLCV data
                ohlcv = exchange.fetch_ohlcv(
                    self.symbol, 
                    timeframe='1h',  # 1-hour candles
                    limit=730  # 2 years of data
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
                df.set_index('Timestamp', inplace=True)
                
                # Validate data
                if self._validate_data_source(df):
                    return df
            
            except Exception as e:
                self.logger.warning(f"CCXT data fetch error from {exchange_id}: {e}")
        
        raise ValueError("Unable to fetch data from any CCXT exchange")
    
    def _validate_data_source(self, data: pd.DataFrame) -> bool:
        """
        Comprehensive validation of historical data
        
        Args:
            data (pd.DataFrame): Input historical price data
        
        Returns:
            bool: Whether data meets minimum requirements
        """
        # Minimum data requirements
        min_data_points = 1000  # Approximately 1 month of hourly data
        min_date_range = 180  # Days
        
        # Check data is not empty
        if data is None or data.empty:
            self.logger.warning("Data source returned empty dataset")
            return False
        
        # Check number of data points
        if len(data) < min_data_points:
            self.logger.warning(f"Insufficient data points: {len(data)} < {min_data_points}")
            return False
        
        # Check date range
        date_range = (data.index.max() - data.index.min()).days
        if date_range < min_date_range:
            self.logger.warning(f"Insufficient date range: {date_range} days < {min_date_range}")
            return False
        
        # Check for NaN values
        nan_percentage = data.isna().mean().mean() * 100
        if nan_percentage > 5:  # More than 5% NaN values
            self.logger.warning(f"High NaN percentage: {nan_percentage:.2f}%")
            return False
        
        return True
    
    def _advanced_data_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced data preprocessing with multiple technical indicators
        
        Args:
            data (pd.DataFrame): Raw historical price data
        
        Returns:
            pd.DataFrame: Preprocessed data with technical indicators
        """
        # Ensure required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Missing required price columns")
        
        # Technical indicators
        def _calculate_rsi(prices, periods=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        def _calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
            exp1 = prices.ewm(span=fast_period, adjust=False).mean()
            exp2 = prices.ewm(span=slow_period, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=signal_period, adjust=False).mean()
            return macd, signal
        
        # Add technical indicators
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(1 + data['returns'])
        data['rsi'] = _calculate_rsi(data['close'])
        data['macd'], data['macd_signal'] = _calculate_macd(data['close'])
        
        # Volatility indicators
        data['volatility'] = data['close'].rolling(window=20).std()
        
        # Normalize volume
        data['normalized_volume'] = (data['volume'] - data['volume'].mean()) / data['volume'].std()
        
        # Drop initial NaN rows from indicator calculations
        data.dropna(inplace=True)
        
        return data
    
    def _generate_simulated_data(self, days=365) -> pd.DataFrame:
        """
        Generate advanced simulated market data with realistic characteristics
        
        Args:
            days (int): Number of days of simulated data
        
        Returns:
            pd.DataFrame: Simulated market data
        """
        import numpy as np
        import pandas as pd
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate timestamps
        timestamps = pd.date_range(end=pd.Timestamp.now(), periods=days*24, freq='H')
        
        # Simulate price movement with more realistic characteristics
        initial_price = 30000  # Starting price
        daily_volatility = 0.03  # 3% daily volatility
        trend_strength = 0.001  # Slight upward trend
        
        # Generate price series with mean-reverting random walk
        prices = [initial_price]
        for _ in range(len(timestamps) - 1):
            # Mean-reverting component
            mean_reversion = (initial_price - prices[-1]) * 0.01
            
            # Random walk with trend and volatility
            random_change = np.random.normal(trend_strength, daily_volatility) * prices[-1]
            
            # Combine components
            price_change = mean_reversion + random_change
            prices.append(prices[-1] + price_change)
        
        # Create DataFrame with realistic price ranges
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + np.random.uniform(0.001, 0.01)) for p in prices],
            'low': [p * (1 - np.random.uniform(0.001, 0.01)) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(mean=np.log(500), sigma=1, size=len(timestamps))
        })
        
        data.set_index('timestamp', inplace=True)
        
        return data
    
    def _calculate_advanced_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced technical indicators for more nuanced trading decisions
        
        Args:
            data (pd.DataFrame): Historical price data
        
        Returns:
            pd.DataFrame: Enhanced dataset with advanced indicators
        """
        # Existing indicators
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(1 + data['returns'])
        
        # Moving Averages
        data['MA20'] = data['close'].rolling(window=20).mean()
        data['MA50'] = data['close'].rolling(window=50).mean()
        data['MA200'] = data['close'].rolling(window=200).mean()
        
        # Relative Strength Index (RSI)
        delta = data['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        relative_strength = avg_gain / avg_loss
        data['RSI'] = 100.0 - (100.0 / (1.0 + relative_strength))
        
        # Bollinger Bands
        data['BB_Middle'] = data['close'].rolling(window=20).mean()
        data['BB_Upper'] = data['BB_Middle'] + 2 * data['close'].rolling(window=20).std()
        data['BB_Lower'] = data['BB_Middle'] - 2 * data['close'].rolling(window=20).std()
        
        # MACD - Moving Average Convergence Divergence
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Stochastic Oscillator
        low_14 = data['low'].rolling(window=14).min()
        high_14 = data['high'].rolling(window=14).max()
        data['%K'] = 100 * (data['close'] - low_14) / (high_14 - low_14)
        data['%D'] = data['%K'].rolling(window=3).mean()
        
        return data.dropna()
    
    def _advanced_action_selection(self, observation: np.ndarray) -> int:
        """
        Advanced action selection using multiple technical indicators
        
        Args:
            observation (np.ndarray): Current market state observation
        
        Returns:
            int: Selected trading action
        """
        # Unpack observation features
        close_price, rsi, macd, signal_line, stoch_k, stoch_d = observation[-6:]
        
        # Complex trading logic
        buy_signals = 0
        sell_signals = 0
        
        # RSI-based signal
        if rsi < 30:  # Oversold condition
            buy_signals += 1
        elif rsi > 70:  # Overbought condition
            sell_signals += 1
        
        # MACD signal
        if macd > signal_line:  # Bullish crossover
            buy_signals += 1
        elif macd < signal_line:  # Bearish crossover
            sell_signals += 1
        
        # Stochastic Oscillator
        if stoch_k < 20 and stoch_d < 20:  # Oversold
            buy_signals += 1
        elif stoch_k > 80 and stoch_d > 80:  # Overbought
            sell_signals += 1
        
        # Decision logic
        if buy_signals > sell_signals:
            return 1  # Buy
        elif sell_signals > buy_signals:
            return 2  # Sell
        else:
            return 0  # Hold
    
    def _generate_observation(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate advanced observation vector including portfolio state, mid momentum, volatility, and Heikin-Ashi trend
        
        Args:
            data (pd.DataFrame): Historical price data
        
        Returns:
            np.ndarray: Observation vector for RL agent
        """
        # Select most recent data point
        latest_data = data.iloc[-1]
        
        # Calculate mid momentum: (close - open)/(high - low) for the last bar
        try:
            mid_momentum = (latest_data['close'] - latest_data['open']) / (latest_data['high'] - latest_data['low']) if (latest_data['high'] - latest_data['low']) != 0 else 0.0
        except Exception:
            mid_momentum = 0.0
        
        # Volatility: standard deviation of close over last 20 bars
        try:
            volatility = data['close'].iloc[-20:].std()
        except Exception:
            volatility = 0.0
        
        # Heikin-Ashi trend (simple encoding: 1 = bullish, -1 = bearish, 0 = neutral)
        try:
            ha_close = (latest_data['open'] + latest_data['high'] + latest_data['low'] + latest_data['close']) / 4
            prev_data = data.iloc[-2] if len(data) > 1 else latest_data
            ha_open = (prev_data['open'] + prev_data['close']) / 2
            ha_trend = 1 if ha_close > ha_open else (-1 if ha_close < ha_open else 0)
        except Exception:
            ha_trend = 0
        
        # Existing market indicators (add more as needed)
        market_indicators = np.array([
            latest_data.get('close', 0.0),
            latest_data.get('volume', 0.0),
            latest_data.get('rsi', 0.0),
            latest_data.get('macd', 0.0),
            latest_data.get('macd_signal', 0.0),
            latest_data.get('returns', 0.0),
            latest_data.get('log_returns', 0.0),
            latest_data.get('bb_middle', 0.0),
            latest_data.get('bb_upper', 0.0),
            latest_data.get('bb_lower', 0.0),
            mid_momentum,
            volatility,
            ha_trend,
            # --- Fetch and add Sentiment Score --- Start
            self.data_provider.get_current_sentiment_score(self.symbol) # Fetch sentiment score
        ], dtype=np.float32)

        # Add portfolio state (consider normalization/scaling)
        normalized_capital = self.current_capital / self.initial_capital
        portfolio_state = np.array([
            normalized_capital, 
            self.crypto_holdings
        ], dtype=np.float32)

        # Concatenate market indicators and portfolio state
        observation = np.concatenate((market_indicators, portfolio_state))
        
        return observation
    
    def _calculate_reward(self, action: int, current_price: float) -> float:
        """
        Calculate reward based on the selected mode:
        - 'sharpe': Differential Sharpe Ratio (risk-adjusted)
        - 'exponential_return': Exponential/quadratic return scaling (aggressive ROI)
        - 'hybrid': Combination of both
        """
        reward_mode = getattr(self, 'reward_mode', 'sharpe')  # Default to 'sharpe' if not set

        # 1. Calculate current portfolio value
        current_portfolio_value = float(self.current_capital + (self.crypto_holdings * current_price))
        self.portfolio_history.append(current_portfolio_value)

        # 2. Check if enough history exists for Sharpe calculation
        if len(self.portfolio_history) < self.sharpe_window:
            # Not enough data yet, return 0 reward
            return 0.0

        # 3. Calculate step-wise returns for the window
        relevant_history = np.array(self.portfolio_history[-self.sharpe_window:], dtype=np.float32)
        step_returns = (relevant_history[1:] - relevant_history[:-1]) / relevant_history[:-1]
        step_returns = np.nan_to_num(step_returns, nan=0.0, posinf=0.0, neginf=0.0)

        std_dev = np.std(step_returns)
        mean_return = np.mean(step_returns)
        if std_dev == 0:
            current_sharpe_ratio = 0.0
        else:
            current_sharpe_ratio = (mean_return - self.risk_free_rate) / std_dev

        # --- Exponential/Quadratic Return Reward ---
        final_return = (current_portfolio_value / self.initial_capital) - 1.0
        if final_return >= 0:
            exp_reward = np.log1p(final_return) ** 2  # Quadratic scaling for positive returns
        else:
            exp_reward = -abs(np.log1p(abs(final_return))) ** 3  # Cubic penalty for negative returns
        exp_reward = float(np.clip(exp_reward, -2.0, 2.0))

        # --- Differential Sharpe Reward ---
        sharpe_reward = current_sharpe_ratio - self.previous_sharpe_ratio
        self.previous_sharpe_ratio = current_sharpe_ratio
        sharpe_reward = float(np.clip(sharpe_reward, -1.0, 1.0))

        # --- Hybrid Reward ---
        if reward_mode == 'exponential_return':
            reward = exp_reward
        elif reward_mode == 'hybrid':
            reward = sharpe_reward + exp_reward
        else:  # Default/risk-adjusted
            reward = sharpe_reward

        # Logging for debugging
        if self.current_step % 100 == 0:
            self.logger.info(f"Step {self.current_step}: Portfolio=${current_portfolio_value:.2f}, Sharpe={current_sharpe_ratio:.4f}, SharpeReward={sharpe_reward:.6f}, ExpReward={exp_reward:.6f}, Mode={reward_mode}, Reward={reward:.6f}")

        return reward
    
    def set_reward_mode(self, mode: str):
        """
        Set reward calculation mode.
        mode: 'sharpe', 'exponential_return', or 'hybrid'
        """
        assert mode in ['sharpe', 'exponential_return', 'hybrid'], "Invalid reward mode."
        self.reward_mode = mode

    def _get_action_mask(self) -> np.ndarray:
        """
        Calculates the action mask based on current capital and holdings.
        Action space: 0=Hold, 1=Buy 25%, 2=Buy 50%, 3=Buy 100%, 4=Sell 25%, 5=Sell 50%, 6=Sell 100%
        Returns:
            np.ndarray: Boolean mask where True indicates a valid action.
        """
        mask = np.ones(self.action_space.n, dtype=bool) # Start with all actions valid

        min_capital_threshold = 1e-6 # Minimum capital to consider buying possible
        min_holding_threshold = 1e-8 # Minimum holdings to consider selling possible

        # Mask buy actions if capital is too low
        if self.current_capital < min_capital_threshold:
            mask[1:4] = False # Mask Buy 25%, 50%, 100%

        # Mask sell actions if holdings are too low
        if self.crypto_holdings < min_holding_threshold:
            mask[4:7] = False # Mask Sell 25%, 50%, 100%

        return mask

    def _validate_state(self):
        """
        Validate the current trading environment state
        Raises:
            ValueError: If state is invalid
        """
        # Check numeric validity
        if not isinstance(self.current_capital, (int, float)):
            raise ValueError(f"Invalid current_capital: {self.current_capital}")
        
        if not isinstance(self.crypto_holdings, (int, float)):
            raise ValueError(f"Invalid crypto_holdings: {self.crypto_holdings}")
        
        # Prevent negative holdings
        if self.current_capital < 0:
            self.logger.warning(f"Negative capital detected: ${self.current_capital}")
            self.current_capital = 0
        
        if self.crypto_holdings < 0:
            self.logger.warning(f"Negative crypto holdings detected: {self.crypto_holdings}")
            self.crypto_holdings = 0
        
        # Check data integrity
        if self.historical_data is None or len(self.historical_data) == 0:
            raise ValueError("No historical data available")
        
        # Validate current step
        if self.current_step < 0 or self.current_step >= len(self.historical_data):
            raise ValueError(f"Invalid current step: {self.current_step}")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one trading step based on the selected action
        
        Args:
            action (int): Action selected by the agent
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.live_trading:
            return self._live_step(action)
        else:
            return self._simulation_step(action)

    def _simulation_step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Handles the logic for a simulation step using historical data."""
        if self.historical_data is None or self.historical_data.empty:
            raise ValueError("Simulation step requires historical data.")
        
        # Ensure current_step is within bounds
        if self.current_step >= len(self.historical_data):
            self.logger.warning(f"Simulation step {self.current_step} exceeds historical data length {len(self.historical_data)}. Ending episode.")
            # Return last valid observation? Or reset observation? Using last known state.
            # TODO: Improve error handling for exceeding data bounds.
            last_observation = self._generate_observation(self.historical_data.iloc[max(0, len(self.historical_data) - 1 - self.lookback_window) : len(self.historical_data) -1])
            return last_observation, 0, True, False, {"warning": "step_beyond_data"}
            
        try:
            # Get current price from historical data
            current_price = self.historical_data['close'].iloc[self.current_step]
            self.logger.debug(f"Simulation Step: {self.current_step}, Price: {current_price}, Action: {action}")

            # Existing simulation trade logic...
            # (Hold action)
            if action == 0:
                pass # No change in capital or holdings
            
            # Sell actions
            elif action in [4, 5, 6]: # Sell 25%, 50%, 100%
                sell_percentage = [0.25, 0.5, 1.0][action-4]
                sell_amount = sell_percentage * self.crypto_holdings
                
                self.logger.debug(f"Attempting Sell: Percentage={sell_percentage*100}%, Amount={sell_amount:.8f}")
                self.logger.debug(f"Crypto Holdings Before Sell: {self.crypto_holdings:.8f}")
                
                # Extremely precise threshold for sell
                sell_threshold = 1e-10  
                if self.crypto_holdings <= sell_threshold:
                    self.logger.warning(
                        f"Cannot sell: Insufficient crypto holdings. "
                        f"Holdings={self.crypto_holdings:.8f}, Required={sell_amount:.8f}"
                    )
                    # Assuming no reward/penalty for failed sell attempt
                    observation = self._generate_observation(self.historical_data.iloc[max(0, self.current_step-self.lookback_window):self.current_step+1])
                    return observation, 0, False, False, {"sell_failure_reason": "insufficient_holdings"}
                
                # Ensure sell_amount doesn't exceed actual holdings due to float precision
                sell_amount = min(sell_amount, self.crypto_holdings)
                
                proceeds = sell_amount * current_price
                fee = proceeds * self.trading_fee
                revenue = proceeds - fee
                
                self.logger.debug(f"Sell Proceeds: ${proceeds:.4f}")
                self.logger.debug(f"Sell Fee: ${fee:.4f}")
                self.logger.debug(f"Sell Revenue: ${revenue:.4f}")

                self.current_capital += revenue
                self.crypto_holdings -= sell_amount
                
                self.logger.debug(f"Capital After Sell: ${self.current_capital:.4f}")
                self.logger.debug(f"Crypto Holdings After Sell: {self.crypto_holdings:.8f}")
            
            # Buy actions 
            elif action in [1, 2, 3]: 
                buy_percentage = [0.25, 0.5, 1.0][action-1]
                investment_amount = buy_percentage * self.current_capital
                
                self.logger.debug(f"Attempting Buy: Percentage={buy_percentage*100}%, Investment=${investment_amount:.4f}")
                self.logger.debug(f"Capital Before Buy: ${self.current_capital:.4f}")

                if investment_amount > 1e-8: # Avoid tiny buys
                    cost = investment_amount
                    fee = cost * self.trading_fee # Fee is on the cost
                    actual_investment = cost - fee
                    position_size = actual_investment / current_price
                    
                    self.logger.debug(f"Buy Cost (incl. potential fee): ${cost:.4f}")
                    self.logger.debug(f"Fee: ${fee:.4f}")
                    self.logger.debug(f"Actual Investment (after fee): ${actual_investment:.4f}")
                    self.logger.debug(f"Position Size: {position_size:.8f}")
                    
                    self.current_capital -= cost
                    self.crypto_holdings += position_size
                    
                    self.logger.debug(f"Capital After Buy: ${self.current_capital:.4f}")
                    self.logger.debug(f"Crypto Holdings After Buy: {self.crypto_holdings:.8f}")
                else:
                    self.logger.warning(
                        f"Cannot buy: Insufficient capital or investment too small. "
                        f"Investment=${investment_amount:.4f}, Available=${self.current_capital:.4f}"
                    )
                    # Assuming no reward/penalty for failed buy attempt
                    observation = self._generate_observation(self.historical_data.iloc[max(0, self.current_step - self.lookback_window):self.current_step + 1])
                    return observation, 0, False, False, {"buy_failure_reason": "insufficient_capital_or_amount"}

            # Generate observation and calculate reward
            observation = self._generate_observation(
                self.historical_data.iloc[max(0, self.current_step-self.lookback_window):self.current_step+1]
            )
            reward = self._calculate_reward(action, current_price)
            
            # Increment current step
            self.current_step += 1
            
            # Check for episode termination
            terminated = self.current_step >= len(self.historical_data) - 1
            
            info = {}
            info['action_mask'] = self._get_action_mask()

            return observation, reward, terminated, False, info
        
        except IndexError as e:
            self.logger.error(f"IndexError at simulation step {self.current_step}: {e}", exc_info=True)
            # Likely stepped beyond data bounds after checks - should not happen often
            last_observation = self._generate_observation(self.historical_data.iloc[-self.lookback_window:])
            return last_observation, 0, True, False, {"error": "IndexError during step"}
        except Exception as e:
            self.logger.error(f"Critical error in simulation step: {e}", exc_info=True)
            # Return initial observation to prevent data slicing issues
            return self._initial_observation, 0, True, False, {"error": str(e)}
            
    def _live_step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Handles the logic for a live trading step using the broker."""
        if not self.broker:
            msg = "Broker not initialized for live trading step."
            self.logger.error(msg)
            raise ConnectionError(msg)

        # --- Capture state BEFORE action ---
        try:
            # Fetch data needed for the current observation
            # Assuming get_historical_data can fetch the needed lookback window
            current_data_df = self.broker.get_historical_data(self.symbol, periods=self.lookback_window + 1, interval='1h') # +1 to have current point
            if current_data_df is None or current_data_df.empty or len(current_data_df) < self.lookback_window:
                 self.logger.error("Failed to get sufficient historical data for current observation.")
                 # Return last known observation? Needs robust handling.
                 # For logging purposes, we might need to skip this step if we can't form a state.
                 # Returning previous observation might make the log inconsistent.
                 # Let's return the initial observation for now, indicating an issue.
                 # TODO: Improve error handling here.
                 return self._initial_observation, 0, False, False, {"error": "current_data_fetch_failed"}
            
            # Generate observation based on data BEFORE action
            current_observation = self._generate_observation(current_data_df.iloc[:-1]) # Use data up to the point before this step
        except Exception as e:
            self.logger.exception("Error generating current observation in live step.")
            # If observation fails, we can't reliably log state. Return initial observation.
            return self._initial_observation, 0, False, False, {"error": "current_observation_generation_failed"}
        # -----------------------------------

        info = {}
        reward = 0.0 # Reward in live trading is complex; log 0 for now, calculate later during training if needed.
        terminated = False # Live trading usually doesn't terminate based on steps
        truncated = False

        try:
            # 1. Get Current Market Price (might be redundant if already in current_data_df)
            # current_price_decimal = self.broker.get_current_price(self.symbol)
            # Use the latest close price from the fetched data instead
            if not current_data_df.empty:
                current_price = current_data_df['close'].iloc[-1]
            else:
                 # Fallback if data fetch failed earlier but somehow we got here
                 self.logger.error("Cannot determine current price from data. Using last close from fetched data.")
                 current_price = float(current_data_df['close'].iloc[-1]) if not current_data_df.empty else 0.0

            self.logger.debug(f"Live Step: {self.current_step}, Price: {current_price}, Action: {action}")

            # 2. Fetch Current State (e.g., balance, holdings - update internal state)
            # This should ideally happen *after* an order fills, or periodically
            # Let's assume self.current_capital and self.crypto_holdings are reasonably up-to-date
            # Consider adding a periodic refresh call here.

            # 3. Execute Trade Logic
            order_result = None
            action_taken = False # Flag if an order was attempted
            if action == 0: # Hold
                pass
            elif action in [1, 2, 3]: # Buy
                buy_percentage = [0.25, 0.5, 1.0][action-1]
                investment_amount = buy_percentage * self.current_capital
                if investment_amount > 1: # Minimum investment threshold
                    self.logger.info(f"Placing BUY order: {self.symbol}, Amount=${investment_amount:.2f}")
                    action_taken = True
                    order_result = self.broker.place_order(
                        symbol=self.symbol,
                        side='buy',
                        order_type='market',
                        amount=investment_amount # Use 'amount' for USD value
                    )
                else:
                    info['buy_failure_reason'] = 'investment_too_small'
                    self.logger.warning(f"Skipping buy: Investment amount ${investment_amount:.2f} too small.")
            elif action in [4, 5, 6]: # Sell
                sell_percentage = [0.25, 0.5, 1.0][action-4]
                sell_quantity = sell_percentage * self.crypto_holdings
                if sell_quantity * current_price > 1: # Minimum sell value threshold
                    self.logger.info(f"Placing SELL order: {self.symbol}, Quantity={sell_quantity:.8f}")
                    action_taken = True
                    order_result = self.broker.place_order(
                        symbol=self.symbol,
                        side='sell',
                        order_type='market',
                        quantity=sell_quantity
                    )
                else:
                    info['sell_failure_reason'] = 'quantity_too_small'
                    self.logger.warning(f"Skipping sell: Sell quantity {sell_quantity:.8f} too small.")

            # 4. Update Internal State Based on Order Result (if any)
            # This part might need refinement based on how the broker API confirms fills
            # For simplicity, assume immediate fill for logging, but real system needs polling/callbacks
            if order_result: # Simplified update, assumes market orders fill quickly
                # In a real system, we'd wait for confirmation before updating state and logging experience
                self.logger.info(f"Order Placed (assuming fill for logging): {order_result}")
                info['order_attempted'] = order_result
                # TODO: Implement robust order status checking and update state accurately based on fills
                # The following state update is PRELIMINARY and might be inaccurate until fill is confirmed
                temp_filled_quantity = order_result.get('filled_quantity', 0.0) # Placeholder
                temp_filled_price = order_result.get('filled_price', current_price) # Placeholder
                temp_fee = order_result.get('fee', 0.0) # Placeholder
                temp_side = order_result.get('side') # Placeholder

                if temp_side == 'buy':
                    cost = (temp_filled_quantity * temp_filled_price) + temp_fee
                    # self.current_capital -= cost # Don't update until confirmed
                    # self.crypto_holdings += temp_filled_quantity
                elif temp_side == 'sell':
                    revenue = (temp_filled_quantity * temp_filled_price) - temp_fee
                    # self.current_capital += revenue # Don't update until confirmed
                    # self.crypto_holdings -= temp_filled_quantity
            elif action_taken:
                 self.logger.warning(f"Order placement failed or returned no result.")
                 info['order_status'] = 'placement_failed_or_no_result'

            # --- Generate next state AFTER action ---
            try:
                # Fetch data again to represent the state AFTER the action potentially occurred
                next_data_df = self.broker.get_historical_data(self.symbol, periods=self.lookback_window + 1, interval='1h')
                if next_data_df is None or next_data_df.empty or len(next_data_df) < self.lookback_window:
                    self.logger.error("Failed to get sufficient historical data for next observation.")
                    # If we can't get next state, use current observation? Log will be less useful.
                    next_observation = current_observation # Fallback
                    info['error'] = info.get('error', '') + ' next_data_fetch_failed'
                else:
                    next_observation = self._generate_observation(next_data_df.iloc[:-1])
            except Exception as e:
                self.logger.exception("Error generating next observation in live step.")
                next_observation = current_observation # Fallback
                info['error'] = info.get('error', '') + ' next_observation_generation_failed'
            # ---------------------------------------

            # --- Log Experience --- TODO: Refine state updates based on actual fills before logging
            experience = {
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'symbol': self.symbol,
                'state': current_observation.tolist(), # Convert numpy array to list for JSON
                'action': action,
                'reward': reward, # Logged as 0, calculate actual reward during training if needed
                'next_state': next_observation.tolist(),
                'terminated': terminated,
                'truncated': truncated,
                'info': info
            }
            try:
                with open(self.experience_log_file, 'a') as f:
                    f.write(json.dumps(experience) + '\n')
            except Exception as e:
                self.logger.exception(f"Failed to write experience to {self.experience_log_file}")
            # ----------------------

            self.current_step += 1
            # TODO: Need robust way to check if simulation should end in live mode (e.g., external signal)

            return next_observation, reward, terminated, truncated, info

        except ConnectionError as e:
             self.logger.error(f"Broker connection error during live step: {e}")
             # Attempt to return last valid state? Or raise?
             # Returning current observation allows loop to continue, but indicates failure.
             return current_observation, 0, False, False, {"error": "broker_connection_error"}
        except Exception as e:
            self.logger.exception("Unhandled exception during live trading step.")
            # Critical error, might need to halt or return error state
            # Returning current observation allows loop to continue, but indicates failure.
            return current_observation, 0, False, False, {"error": "unhandled_live_step_exception"}
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the trading environment
        
        Args:
            seed (int, optional): Random seed for reproducibility
            options (Dict, optional): Additional reset options
        
        Returns:
            Tuple of initial observation and info dictionary
        """
        # Set random seed if provided
        super().reset(seed=seed)
        self.logger.info(f"Resetting environment. Live Trading: {self.live_trading}")

        if self.live_trading:
            if not self.broker:
                msg = "Broker not initialized for live trading reset."
                self.logger.error(msg)
                raise ConnectionError(msg)
            try:
                # --- Fetch Live State --- 
                self.logger.info("Fetching live account state from broker...")
                self.current_capital = float(self.broker.get_available_capital() or 0.0) # Use new method, convert Decimal
                self.crypto_holdings = float(self.broker.get_holding_quantity(self.symbol) or 0.0) # Use new method, convert Decimal
                self.initial_capital = self.current_capital # Reset initial capital to current live capital
                self.logger.info(f"Live state fetched: Capital={self.current_capital:.2f}, Holdings={self.crypto_holdings:.6f}")

                # --- Fetch Initial Live Data for Observation --- 
                self.logger.info("Fetching recent market data for initial observation...")
                # Use the broker's get_historical_data method
                live_data_df = self.broker.get_historical_data(self.symbol, periods=self.lookback_window, interval='1h') # Example signature
                if live_data_df is None or live_data_df.empty or len(live_data_df) < self.lookback_window:
                     msg = f"Failed to fetch sufficient live data ({len(live_data_df) if live_data_df is not None else 0}/{self.lookback_window}) for initial observation."
                     self.logger.error(msg)
                     # Fallback? Or raise error? For now, raise.
                     raise ValueError(msg)
                
                # Use fetched live data to generate the initial observation
                # Preprocess the fetched live data before generating observation
                processed_live_data = self._advanced_data_preprocessing(live_data_df.copy())
                self._initial_observation = self._generate_observation(processed_live_data)
                self.logger.info("Initial observation generated from live data.")

                # --- Reset Portfolio Tracking for Live --- 
                # Calculate initial portfolio value based on live data
                # Use get_current_price to get the initial valuation price
                current_price_decimal = self.broker.get_current_price(self.symbol)
                if current_price_decimal is None:
                    self.logger.error("Could not fetch current price during live reset. Using last close from fetched data.")
                    current_price = float(live_data_df['close'].iloc[-1]) if not live_data_df.empty else 0.0
                else:
                    current_price = float(current_price_decimal)

                self.portfolio_value = self.current_capital + (self.crypto_holdings * current_price)
                self.max_portfolio_value = self.portfolio_value
                self.portfolio_history = [self.portfolio_value] * self.lookback_window # Initialize history, adjusted size to lookback
                self.previous_sharpe_ratio = 0.0
                self.current_step = 0 # Reset step counter for live session

            except Exception as e:
                self.logger.error(f"Error during live trading reset: {e}", exc_info=True)
                # Handle error appropriately, maybe raise or return a default state?
                raise ConnectionError(f"Failed to reset live environment: {e}")
            
            # --- Calculate Initial Action Mask for Live ---    
            initial_mask = self._get_action_mask()
            
        else:
            # --- Simulation Reset Logic --- 
            if self.historical_data is None or self.historical_data.empty:
                 raise ValueError("Cannot reset simulation environment without historical data.")
            
            self.current_step = self.lookback_window # Start after lookback period
            self.current_capital = self.initial_capital
            self.crypto_holdings = 0
            self.portfolio_value = self.initial_capital
            self.max_portfolio_value = self.initial_capital

            # --- Reset Differential Sharpe Ratio tracking --- 
            self.portfolio_history = [self.initial_capital] * self.lookback_window # Start history with initial value
            self.previous_sharpe_ratio = 0.0
            # --------------------------------------------- 
            
            # Ensure initial observation is prepared correctly after reset
            self._initial_observation = self._generate_observation(
                self.historical_data.iloc[:self.lookback_window]
            )
            self.logger.info("Simulation environment reset completed.")
            
            # --- Calculate Initial Action Mask for Simulation --- 
            initial_mask = self._get_action_mask()

        # Return initial observation and info dictionary containing the mask
        return self._initial_observation, {"action_mask": initial_mask}
    
    def render(self):
        """
        Render environment state (optional)
        """
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        print(f"Crypto Holdings: {self.crypto_holdings}")
        print(f"Capital: ${self.current_capital:.2f}")

def main():
    """
    Example usage of CryptoTradingEnvironment
    """
    env = CryptoTradingEnvironment(symbol='BTC-USD')
    
    # Run a sample episode
    obs, info = env.reset()
    terminated, truncated = False, False
    
    while not (terminated or truncated):
        action = env._advanced_action_selection(obs)  
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break

if __name__ == "__main__":
    main()
