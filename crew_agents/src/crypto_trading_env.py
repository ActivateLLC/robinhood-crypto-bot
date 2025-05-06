import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym # Main RL environment library
import ccxt
from typing import Dict, Any, Tuple, Optional, Literal
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from decimal import Decimal
import importlib
import json
import pandas_ta as ta

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Corrected path to project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .utils import setup_logger # Corrected: utils.py is in the same directory (src)
# from ..src.feature_engineering import AdvancedFeatureEngineer # Module/Class not found
from alt_crypto_data import AltCryptoDataProvider # Corrected: alt_crypto_data.py is at project root
# from ..src.data_providers.base_data_provider import BaseDataProvider # Module/Class not found
from brokers.base_broker import BaseBroker # Corrected: Changed Broker to BaseBroker

# For type hinting, if BaseDataProvider is not found, use Any or a more generic type
# from typing import Optional, Dict, Any, List, Tuple # Ensure Any is imported if used elsewhere
from typing import Optional, Dict, Any, List, Tuple # Ensure Any is imported

# Define constants if they are used and not defined elsewhere, or ensure they are imported

# Import Broker Interfaces/Implementations

try:
    brokers_base = importlib.import_module("brokers.base_broker")
    BaseBroker = brokers_base.BaseBroker # Use BaseBroker, not BrokerInterface
    # MarketData = brokers_base.MarketData # Also import MarketData if needed
except ImportError as e:
    print(f"ERROR: Failed to import brokers.base_broker: {e}")
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
        df: pd.DataFrame, 
        config: Dict[str, Any], 
        live_trading: bool = False, 
        broker_type: str = 'simulation', 
        _test_trigger_debug_dump: bool = False,
        data_provider: Optional[Any] = None # Changed BaseDataProvider to Any as it's not found
    ):
        super().__init__()
        # Temporary, direct print to see incoming config *before* logger is set up
        print(f"DEBUG ENV INIT [RAW]: Received config type: {type(config)}")
        if isinstance(config, dict):
            print(f"DEBUG ENV INIT [RAW]: Received config keys: {list(config.keys())}")
            print(f"DEBUG ENV INIT [RAW]: Received config['window_size']: {config.get('window_size')}")
        else:
            print(f"DEBUG ENV INIT [RAW]: Received config is NOT a dict: {config}")

        self.broker_type = broker_type # Needs to be set before logger for logger name
        self.logger = setup_logger(
            name=f"CryptoTradingEnv.{self.broker_type}.{config.get('symbol', 'DefaultSymbol')}",
            log_level=config.get('log_level', 'INFO')
        )
        self.live_trading = live_trading
        # self.broker_type = broker_type # Moved up
        self._test_trigger_debug_dump = _test_trigger_debug_dump

        self.logger.info(f"DEBUG ENV INIT: Assigning config. Type before assign: {type(config)}")
        self.config = config
        self.logger.info(f"DEBUG ENV INIT: self.config type after assign: {type(self.config)}")
        if isinstance(self.config, dict):
            self.logger.info(f"DEBUG ENV INIT: self.config keys: {list(self.config.keys())}")
            self.logger.info(f"DEBUG ENV INIT: self.config['window_size'] via get: {self.config.get('window_size')}")
        else:
            self.logger.warning(f"DEBUG ENV INIT: self.config IS NOT A DICT after assignment! Value: {self.config}")

        if not isinstance(df, pd.DataFrame) or df.empty:
            self.logger.error("DataFrame is not valid or empty.")
        
        self.logger.info(f"DEBUG ENV INIT: About to set self.symbol and self.window_size.")
        self.logger.info(f"DEBUG ENV INIT: self.config type at this point: {type(self.config)}")
        if isinstance(self.config, dict):
            self.logger.info(f"DEBUG ENV INIT: self.config['window_size'] from dict before setting self.window_size: {self.config.get('window_size')}")
        else:
            self.logger.warning(f"DEBUG ENV INIT: self.config IS NOT A DICT before setting window_size! Value: {self.config}")

        # Configuration parameters
        self.symbol = self.config.get('symbol', 'BTC-USD')
        self.window_size = self.config.get('window_size', 24) # Default to 24 if not found
        self.logger.info(f"DEBUG ENV INIT: self.window_size set to: {self.window_size}")

        self.initial_balance = float(self.config.get('initial_balance', 10000))
        
        # Core Parameters
        if data_provider:
            self.data_provider = data_provider
            self.logger.info("Using provided data_provider.")
        else:
            self.data_provider = AltCryptoDataProvider() # Use default provider if none passed
            self.logger.info("Initialized default AltCryptoDataProvider.")
            
        self.initial_capital = self.initial_balance
        self.lookback_window = self.window_size
        self.trading_fee = 0.0005  # Reduced fee for more frequent trading 
        self.live_trading = live_trading
        self.broker_type = broker_type if live_trading else 'simulation'
        self.broker: Optional[BaseBroker] = None 
        self.log_experience = False # Store the parameter value

        self.logger.info(f"Initializing environment: Symbol={self.symbol}, Live Trading={self.live_trading}, Broker Type={self.broker_type}")

        # --- Live Trading Setup ---
        if self.live_trading:
            load_dotenv() 
            if self.broker_type == 'robinhood':
                try:
                    # Instantiate RobinhoodBroker without passing keys; it should load them itself
                    self.broker = RobinhoodBroker()
                    self.logger.info("Robinhood broker initialized successfully for live trading.")
                    # Fetch initial live state (capital, holdings) in reset()
                    # Override trading fee with broker's fee if applicable
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

        # Action space: Must match the model being loaded (Discrete(3) from error)
        # Original assumption: 0=Hold, 1=Buy 25%, 2=Buy 50%, 3=Buy 100%, 4=Sell 25%, 5=Sell 50%, 6=Sell 100%
        self.action_space = gym.spaces.Discrete(3) # Match the model's expected action space
        
        # Observation space: Includes market indicators + portfolio state (capital, holdings)
        # Needs to match the shape the model was trained with.
        observation_shape = (31,) # Corrected: Match the shape from the error message
        self.observation_space = gym.spaces.Box(
            low=-1.0, # Reverted to match model expectation
            high=1.0, # Reverted to match model expectation
            shape=observation_shape, # Pass the tuple directly
            dtype=np.float32
        )
        
        # Data handling
        self.historical_data = None
        if not self.live_trading: 
            if df is not None:
                self.logger.info(f"Received DataFrame df. Shape: {df.shape}, Is empty: {df.empty}")
                self.historical_data = df
                log_shape = self.historical_data.shape if self.historical_data is not None else 'None'
                log_empty = self.historical_data.empty if self.historical_data is not None else 'N/A'
                self.logger.info(f"Assigned df to self.historical_data. Shape: {log_shape}, Is empty: {log_empty}")
            else:
                self.logger.warning("No df passed, attempting internal fetch.") # Keep this line
                self.historical_data = self._fetch_historical_data()
            log_shape_check = self.historical_data.shape if self.historical_data is not None else 'None'
            log_empty_check = self.historical_data.empty if self.historical_data is not None else 'N/A'
            self.logger.info(f"Checking self.historical_data before raising error. Shape: {log_shape_check}, Is empty: {log_empty_check}")
            if self.historical_data is None or self.historical_data.empty:
                self.logger.error("Failed to load historical data for simulation (self.historical_data is None or empty).") # Keep this line
                raise ValueError("Historical data could not be loaded.")
            self.logger.info(f"Loaded historical data for simulation: {len(self.historical_data)} points.")
        else:
            # In live mode, we might fetch a small recent window in reset/step if needed for indicators
            self.logger.info("Live trading mode: Historical data will be fetched dynamically as needed.")
        
        # Trading state variables (will be updated from broker in live mode during reset)
        self.current_step = 0
        self.current_capital = self.initial_balance
        self.crypto_holdings = 0
        self.portfolio_value = self.initial_balance
        self.max_portfolio_value = self.initial_balance
        self.max_drawdown = 0.2  

        # --- Additions for Differential Sharpe Ratio ---
        self.sharpe_window = 100  
        self.risk_free_rate = 0.0 
        self.portfolio_history = [self.initial_balance] 
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
        """Enhanced data preprocessing with configurable TA indicators and refined logging."""
        if data.empty:
            self.logger.warning("Input DataFrame for preprocessing is empty. Returning as is.")
            return data

        # Initialize series/df variables to None
        atr_series = None
        keltner_channels = None
        roc_series = None

        # --- Basic Feature Engineering ---
        # Ensure necessary columns are present (High, Low, Close, Volume)
        required_cols_capitalized = ['High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols_capitalized if col not in data.columns]
        if missing_cols:
            self.logger.error(f"Critical columns missing from input data: {missing_cols}. Available: {data.columns.tolist()}")
            # Attempt to rename if lowercase versions exist
            renamed_cols = False
            for col_cap in required_cols_capitalized:
                col_low = col_cap.lower()
                if col_low in data.columns and col_cap not in data.columns:
                    data.rename(columns={col_low: col_cap}, inplace=True)
                    self.logger.info(f"Renamed column '{col_low}' to '{col_cap}'.")
                    renamed_cols = True
            
            # Re-check for missing columns after attempting rename
            missing_cols_after_rename = [col for col in required_cols_capitalized if col not in data.columns]
            if missing_cols_after_rename:
                self.logger.error(f"Critical columns still missing after rename attempt: {missing_cols_after_rename}. Processing cannot continue.")
                raise ValueError(f"Missing critical columns after rename: {missing_cols_after_rename}")
            elif renamed_cols:
                self.logger.info("Successfully renamed lowercase columns to uppercase.")
        else:
            self.logger.debug("All critical columns (High, Low, Close, Volume) are present.")

        if data.empty:
            self.logger.error("Input data for preprocessing is empty.")
            raise ValueError("Input data is empty.")
        
        if len(data) < 20: # Minimum length for some TAs like Keltner Channel default
            self.logger.warning(f"Data length ({len(data)}) is less than 20, some indicators might be NaN or unreliable.")
            # Depending on strategy, might raise ValueError or proceed with caution.

        # --- Feature Engineering: Technical Indicators ---
        self.logger.info("Calculating technical indicators...")
        data = data.copy() # Ensure we're working on a copy

        # Ensure columns are float, coerce errors to NaN for TAs that can handle it
        for col in required_cols_capitalized:
            if data[col].dtype != 'float64' and data[col].dtype != 'int64':
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    self.logger.debug(f"Converted column {col} to numeric.")
                except Exception as e:
                    self.logger.error(f"Could not convert column {col} to numeric: {e}. It will be left as is or may cause TA calculation errors.")
        
        # Basic Indicators
        try:
            data['SMA_20'] = ta.sma(data['Close'], length=20)
            data['EMA_20'] = ta.ema(data['Close'], length=20)
            data['RSI'] = ta.rsi(data['Close'], length=14)
            # MACD (Moving Average Convergence Divergence)
            macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
            if macd is not None and not macd.empty:
                data['MACD'] = macd['MACD_12_26_9']
                data['MACD_signal'] = macd['MACDs_12_26_9']
                data['MACD_hist'] = macd['MACDh_12_26_9']
            else:
                self.logger.warning("MACD calculation returned None or empty. Skipping MACD features.")
                data['MACD'] = data['MACD_signal'] = data['MACD_hist'] = np.nan
            
            # Bollinger Bands
            bbands = ta.bbands(data['Close'], length=20, std=2)
            if bbands is not None and not bbands.empty:
                data['BB_upper'] = bbands['BBU_20_2.0']
                data['BB_middle'] = bbands['BBM_20_2.0']
                data['BB_lower'] = bbands['BBL_20_2.0']
            else:
                self.logger.warning("Bollinger Bands calculation returned None or empty. Skipping BB features.")
                data['BB_upper'] = data['BB_middle'] = data['BB_lower'] = np.nan

            # ATR (Average True Range) - Using updated parameter
            atr_series = ta.atr(data['High'], data['Low'], data['Close'], length=self.atr_length_config) # Use config
            print(f"DEBUG: Raw atr_series from ta.atr:\n{atr_series.to_string() if atr_series is not None else 'None'}")
            if atr_series is not None and not atr_series.empty:
                data['ATR'] = atr_series # Directly assign the calculated Series
                data['ATR'] = data['ATR'].fillna(0.0)  # Specific fill for ATR NaNs
            else:
                self.logger.warning("ATR calculation returned None or empty Series.")
                data['ATR'] = 0.0 # Fallback, though fillna below would also cover NaNs
            if 'ATR' in data.columns:
                print(f"DEBUG ENV: data['ATR'] post-assign & specific fill (len {len(data['ATR'])}):\n{data['ATR'].to_string() if not data['ATR'].empty else 'Empty Series'}")
            else:
                print("DEBUG ENV: data['ATR'] column NOT created post-assign.")

            # Keltner Channels - Using updated parameter
            # Ensure 'mamode' is a string, default to 'sma' if not specified or not a string
            kc_mamode = str(self.keltner_mamode_config) if isinstance(self.keltner_mamode_config, str) else 'sma'
            keltner_channels = ta.kc(
                data['High'], data['Low'], data['Close'], 
                length=self.keltner_length_config, 
                scalar=self.keltner_scalar_config, 
                mamode=kc_mamode, 
                atr_length=self.keltner_atr_length_config
            )
            print(f"DEBUG: Raw keltner_channels from ta.kc:\n{keltner_channels.to_string() if keltner_channels is not None else 'None'}")
            if keltner_channels is not None and not keltner_channels.empty:
                self.logger.info(f"Keltner Channels calculated. Columns: {keltner_channels.columns.tolist()}")
                # Dynamically construct expected column names based on pandas-ta conventions
                # For mamode='sma', pandas-ta uses 's' suffix, e.g., KCBs_20_2.0
                # For other mamodes like 'ema', it includes mamode, e.g., KCB_20_ema_2.0
                # The atr_length is used in calculation but not typically in the column name string for 'sma'.
                base_col_name = f"KCB{'s' if kc_mamode == 'sma' else ''}_{self.keltner_length_config}"
                upper_col_name = f"KCU{'s' if kc_mamode == 'sma' else ''}_{self.keltner_length_config}"
                lower_col_name = f"KCL{'s' if kc_mamode == 'sma' else ''}_{self.keltner_length_config}"

                if kc_mamode != 'sma':
                    basis_col_name_expected = f"{base_col_name}_{kc_mamode}_{self.keltner_scalar_config}"
                    upper_col_name_expected = f"{upper_col_name}_{kc_mamode}_{self.keltner_scalar_config}"
                    lower_col_name_expected = f"{lower_col_name}_{kc_mamode}_{self.keltner_scalar_config}"
                else:
                    basis_col_name_expected = f"{base_col_name}_{self.keltner_scalar_config}" # e.g., KCBs_20_2.0
                    upper_col_name_expected = f"{upper_col_name}_{self.keltner_scalar_config}" # e.g., KCUs_20_2.0
                    lower_col_name_expected = f"{lower_col_name}_{self.keltner_scalar_config}" # e.g., KCLs_20_2.0

                # Assign to DataFrame if columns exist in calculated keltner_channels
                if basis_col_name_expected in keltner_channels.columns:
                    data['Keltner_Basis'] = keltner_channels[basis_col_name_expected]
                else:
                    self.logger.warning(f"Expected Keltner Basis column '{basis_col_name_expected}' not found. Available: {keltner_channels.columns.tolist()}")
                    data['Keltner_Basis'] = 0.0
                
                if upper_col_name_expected in keltner_channels.columns:
                    data['Keltner_Upper'] = keltner_channels[upper_col_name_expected]
                else:
                    self.logger.warning(f"Expected Keltner Upper column '{upper_col_name_expected}' not found. Available: {keltner_channels.columns.tolist()}")
                    data['Keltner_Upper'] = 0.0
                
                if lower_col_name_expected in keltner_channels.columns:
                    data['Keltner_Lower'] = keltner_channels[lower_col_name_expected]
                else:
                    self.logger.warning(f"Expected Keltner Lower column '{lower_col_name_expected}' not found. Available: {keltner_channels.columns.tolist()}")
                    data['Keltner_Lower'] = 0.0
                data['Keltner_Basis'] = data['Keltner_Basis'].fillna(0.0)
                data['Keltner_Upper'] = data['Keltner_Upper'].fillna(0.0)
                data['Keltner_Lower'] = data['Keltner_Lower'].fillna(0.0)
            else:
                self.logger.warning("Keltner Channels calculation returned None or empty. Skipping Keltner features.")
                data['Keltner_Upper'] = data['Keltner_Lower'] = data['Keltner_Basis'] = np.nan
                data['Keltner_Basis'] = data['Keltner_Basis'].fillna(0.0)
                data['Keltner_Upper'] = data['Keltner_Upper'].fillna(0.0)
                data['Keltner_Lower'] = data['Keltner_Lower'].fillna(0.0)
            if 'Keltner_Basis' in data.columns:
                print(f"DEBUG ENV: data['Keltner_Basis'] post-assign & specific fill (len {len(data['Keltner_Basis'])}):\n{data['Keltner_Basis'].to_string() if not data['Keltner_Basis'].empty else 'Empty Series'}")
            else:
                print("DEBUG ENV: data['Keltner_Basis'] column NOT created post-assign.")

            # ROC (Rate of Change) - Using updated parameter
            roc_series = ta.roc(data['Close'], length=self.roc_length_config) # Use config
            print(f"DEBUG: Raw roc_series from ta.roc:\n{roc_series.to_string() if roc_series is not None else 'None'}")
            if roc_series is not None and not roc_series.empty:
                data['ROC'] = roc_series # Directly assign the calculated Series
                data['ROC'] = data['ROC'].fillna(0.0)  # Specific fill for ROC NaNs
            else:
                self.logger.warning("ROC calculation returned None or empty Series.")
                data['ROC'] = 0.0 # Fallback, though fillna below would also cover NaNs
                data['ROC'] = data['ROC'].fillna(0.0)
            if 'ROC' in data.columns:
                print(f"DEBUG ENV: data['ROC'] post-assign & specific fill (len {len(data['ROC'])}):\n{data['ROC'].to_string() if not data['ROC'].empty else 'Empty Series'}")
            else:
                print("DEBUG ENV: data['ROC'] column NOT created post-assign.")

            # Stochastic Oscillator %K and %D
            stoch = ta.stoch(data['High'], data['Low'], data['Close'], k=14, d=3, smooth_k=3)
            if stoch is not None and not stoch.empty:
                data['Stoch_K'] = stoch['STOCHk_14_3_3']
                data['Stoch_D'] = stoch['STOCHd_14_3_3']
            else:
                self.logger.warning("Stochastic Oscillator calculation returned None or empty. Skipping Stoch features.")
                data['Stoch_K'] = data['Stoch_D'] = np.nan

            # TTM Squeeze (ta.squeeze or ta.squeeze_pro)
            # `ta.squeeze` is often used. It looks for Bollinger Bands within Keltner Channels.
            # Parameters: bb_length, bb_std, kc_length, kc_scalar (or atr_length for kc)
            squeeze = ta.squeeze(data['High'], data['Low'], data['Close'], 
                                 bb_length=20, bb_std=2.0, 
                                 kc_length=20, kc_scalar=1.5) # kc_scalar or use atr_length with kc
            if squeeze is not None and not squeeze.empty and 'SQZ_ON' in squeeze.columns:
                data['TTM_Squeeze_On'] = squeeze['SQZ_ON'] # Boolean indicating if squeeze is on
                # 'SQZ_OFF' or 'SQZ_NO' might also be available, or just use `~SQZ_ON`
                # 'SQZ_ALERT' if momentum breaks out of squeeze
                if 'SQZ_ALERT' in squeeze.columns: data['TTM_Squeeze_Alert'] = squeeze['SQZ_ALERT']
                else: data['TTM_Squeeze_Alert'] = 0 # Default to 0 if not present
            else:
                self.logger.warning("TTM Squeeze calculation returned None, empty, or missing 'SQZ_ON'. Skipping TTM Squeeze.")
                data['TTM_Squeeze_On'] = 0 # Default to 0 (squeeze off)
                data['TTM_Squeeze_Alert'] = 0

            # Volume-based indicators
            data['OBV'] = ta.obv(data['Close'], data['Volume'])
            data['VWAP'] = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume'])
        
        except Exception as e:
            self.logger.error(f"Error calculating one or more basic technical indicators: {e}", exc_info=True)
            # Fill potentially missing columns with NaN if error occurred mid-calculation
            ta_cols = ['SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 
                       'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'Keltner_Upper', 'Keltner_Lower', 
                       'Keltner_Basis', 'ROC', 'Stoch_K', 'Stoch_D', 'TTM_Squeeze_On', 'TTM_Squeeze_Alert', 
                       'OBV', 'VWAP']
            for col in ta_cols:
                if col not in data.columns:
                    data[col] = np.nan
        
        self.logger.info("Basic technical indicators calculated.")

        # --- Advanced Features: Momentum, Volatility, Trend (Heikin-Ashi) ---
        self.logger.info("Calculating advanced features (momentum, volatility, trend)...")
        try:
            # Momentum Indicators
            data['Momentum_10'] = data['Close'].diff(10) # 10-period momentum

            # Volatility Indicators (e.g., Standard Deviation of Close)
            data['Volatility_20'] = data['Close'].rolling(window=20).std()

            # Heikin-Ashi Candlesticks (requires a separate calculation as it modifies O,H,L,C)
            # Note: pandas_ta does not have a direct Heikin-Ashi transform that returns a full DataFrame in one go easily attachable.
            # We might need a helper or implement it if critical for observation.
            # For now, we'll skip direct Heikin-Ashi in the main features and assume other indicators capture trend.
            # If TTM Squeeze uses HA internally, that's handled by `ta.squeeze`.

            # Price relative to EMAs (trend direction)
            data['Price_vs_EMA50'] = data['Close'] / ta.ema(data['Close'], length=50) - 1
            data['Price_vs_EMA200'] = data['Close'] / ta.ema(data['Close'], length=200) - 1

        except Exception as e:
            self.logger.error(f"Error calculating advanced features: {e}", exc_info=True)
            adv_feat_cols = ['Momentum_10', 'Volatility_20', 'Price_vs_EMA50', 'Price_vs_EMA200']
            for col in adv_feat_cols:
                if col not in data.columns:
                    data[col] = np.nan
        self.logger.info("Advanced features calculated.")

        # --- Handling Missing Values --- 
        self.logger.info(f"Handling missing values. NaN count before ffill/bfill: {data.isnull().values.any()}")
        # Option 1: Forward-fill then backward-fill (common for time series)
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        # Option 2: Fill with a specific value (e.g., 0 or mean) if ffill/bfill leaves NaNs (e.g., at the very start)
        # data[numeric_cols] = data[numeric_cols].fillna(0.0) # Temporarily commented out for debugging
        numeric_cols = data.select_dtypes(include=np.number).columns
        data[numeric_cols] = data[numeric_cols].fillna(0.0) # Restored this line
        self.logger.info(f"Filled NaN values with 0.0. Data shape after NaN fill: {data.shape}")

        # Debug prints after final fillna
        if 'ATR' in data.columns:
            print(f"DEBUG ENV: data['ATR'] POST-FINAL-FILLNA:\n{data['ATR'].to_string()}")
        if 'Keltner_Basis' in data.columns:
            print(f"DEBUG ENV: data['Keltner_Basis'] POST-FINAL-FILLNA:\n{data['Keltner_Basis'].to_string()}")
        if 'ROC' in data.columns:
            print(f"DEBUG ENV: data['ROC'] POST-FINAL-FILLNA:\n{data['ROC'].to_string()}")

        # --- Normalization (Example: Z-score or Min-Max) ---
        # For RL, it's often good to normalize features to a similar range (e.g., [-1, 1] or [0, 1])
        # Here we apply a simple scaling for demonstration. Robust normalization might be needed.
        self.logger.info("Normalizing selected features...")
        # Select only numeric columns for normalization, exclude original OHLCV if model doesn't need them raw
        # or if they are implicitly used via returns/indicators.
        # For this example, let's assume we normalize all TA indicators and engineered features.
        cols_to_normalize = [
            'SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
            'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'Keltner_Upper', 'Keltner_Lower',
            'Keltner_Basis', 'ROC', 'Stoch_K', 'Stoch_D', 'TTM_Squeeze_On', 'TTM_Squeeze_Alert',
            'OBV', 'VWAP', 'Momentum_10', 'Volatility_20', 'Price_vs_EMA50', 'Price_vs_EMA200'
        ]
        # Add 'Volume' to normalization if it's part of the observation space directly
        # Original 'Open', 'High', 'Low', 'Close' might be scaled or used to derive returns, etc.

        for col in cols_to_normalize:
            if col in data.columns:
                # Simple min-max scaling to [-1, 1] (example)
                # More robust: Z-score ( (x - mean) / std ) or ( (x - min) / (max - min) ) * 2 - 1
                # Handle cases where max == min to avoid division by zero
                min_val = data[col].min()
                max_val = data[col].max()
                if max_val > min_val:
                    data[col] = 2 * (data[col] - min_val) / (max_val - min_val) - 1
                else:
                    data[col] = 0 # Or some other appropriate value if all values are the same
            else:
                self.logger.warning(f"Column {col} not found for normalization, was it calculated correctly?")
        self.logger.info("Feature normalization completed.")

        # --- Final Check for NaNs/Infs --- 
        if data.isnull().values.any() or np.isinf(data.select_dtypes(include=[np.number])).values.any():
            self.logger.error("NaN or Inf values still present after preprocessing and normalization!")
            # Log details of NaNs/Infs
            nan_counts = data.isnull().sum()
            inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
            self.logger.error(f"NaN counts per column:\n{nan_counts[nan_counts > 0]}")
            self.logger.error(f"Inf counts per column:\n{inf_counts[inf_counts > 0]}")
            # Consider raising an error or more robust imputation
            # For now, fill again to prevent crashes, but this indicates an issue
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data.fillna(0.0, inplace=True)
            self.logger.warning("Replaced Inf with NaN and re-ran fillna(0.0) as a fallback.")

        self.logger.info(f"Advanced data preprocessing completed. Final data shape: {data.shape}")
        
        return data

    def _get_initial_market_data(self, length: int = 200) -> pd.DataFrame:
        """Fetches initial market data for the environment."""
        # Fetch data from the data provider
        data = self.data_provider.fetch_price_history(self.symbol, days=length, interval='1h')
        if data is None or data.empty:
            self.logger.error("Failed to fetch initial market data.")
            raise ValueError("Initial market data could not be loaded.")
        return data

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
        
        # Use pre-calculated volatility from _advanced_data_preprocessing
        # Volatility: standard deviation of close over last 20 bars
        # try:
        #     volatility = data['close'].iloc[-20:].std()
        # except Exception:
        #     volatility = 0.0
        
        # Heikin-Ashi trend (simple encoding: 1 = bullish, -1 = bearish, 0 = neutral)
        try:
            ha_close = (latest_data['open'] + latest_data['high'] + latest_data['low'] + latest_data['close']) / 4
            prev_data = data.iloc[-2] if len(data) > 1 else latest_data
            ha_open = (prev_data['open'] + prev_data['close']) / 2
            ha_trend = 1 if ha_close > ha_open else (-1 if ha_close < ha_open else 0)
        except Exception:
            ha_trend = 0
        
        # Existing market indicators (add more as needed)
        market_indicators_list = [
            latest_data.get('close', 0.0),
            latest_data.get('volume', 0.0),
            latest_data.get('rsi', 0.0),
            latest_data.get('macd_line', 0.0),
            latest_data.get('macd_signal_line', 0.0),
            latest_data.get('macd_hist', 0.0),
            latest_data.get('returns', 0.0),
            latest_data.get('log_returns', 0.0),
            latest_data.get('ttm_squeeze_on', 0.0),
            latest_data.get('ttm_momentum', 0.0),
            latest_data.get('ttm_atr', 0.0),
            mid_momentum, 
            latest_data.get('volatility', 0.0), 
            ha_trend,
            self.data_provider.get_current_sentiment_score(self.symbol),
            latest_data.get('stoch_k', 0.0),
            latest_data.get('stoch_d', 0.0),
            latest_data.get('adx', 0.0),
            latest_data.get('willr', 0.0),
            # CCI removed
            latest_data.get('ema_50', 0.0),
            # EMA200 removed
            latest_data.get('proc', 0.0),
            latest_data.get('obv', 0.0),
            latest_data.get('close_lag_1', 0.0),
            latest_data.get('close_lag_2', 0.0),
            latest_data.get('close_lag_3', 0.0),
            latest_data.get('close_lag_5', 0.0),
            # close_lag_10 removed
            # New TTM Features
            latest_data.get('ttm_wave_trend', 0.0),
            latest_data.get('ttm_wave_cross', 0.0),
            latest_data.get('ttm_scalper_alert', 0.0)
        ]
        # self.logger.debug(f"Market indicators list length: {len(market_indicators_list)}")
        # self.logger.debug(f"Market indicators before clip: {market_indicators_list}")

        market_indicators = np.array(market_indicators_list, dtype=np.float32)
        # Clip market indicators to be within [-1, 1] if necessary, though proper scaling is better
        # This is a temporary measure if the model strictly expects [-1, 1] and normalization isn't fully implemented
        market_indicators = np.clip(market_indicators, -1.0, 1.0)

        # Portfolio state: current capital and crypto holdings (normalized)
        normalized_capital = self.current_capital / self.initial_balance if self.initial_balance > 0 else 0.0
        # Max holding could be estimated based on initial capital and some leverage or typical position size
        # For simplicity, let's assume max_crypto_to_hold is related to initial capital / some reference price, or a fixed large number
        # This normalization needs to be consistent with training
        max_possible_holding_value_estimate = self.initial_balance # Simplistic: can hold up to initial capital value in crypto
        current_price = latest_data.get('close', self.initial_price_guess)
        max_crypto_units = max_possible_holding_value_estimate / current_price if current_price > 0 else 1.0 # Avoid div by zero
        
        normalized_holdings = self.crypto_holdings / max_crypto_units if max_crypto_units > 0 else 0.0
        
        portfolio_state = np.array([
            np.clip(normalized_capital, -1.0, 1.0), 
            np.clip(normalized_holdings, -1.0, 1.0)
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
        final_return = (current_portfolio_value / self.initial_balance) - 1.0
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
                
                self.logger.debug(f"Attempting Buy: Percentage={buy_percentage*100}%, Investment=${investment_amount:.2f}")
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
                 # Return last valid observation? Needs robust handling.
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
            if self.log_experience:
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
                self.current_capital = float(self.broker.get_available_capital() or Decimal('0.0')) # Ensure Decimal for or
                
                # Correctly fetch all holdings and then get the specific one
                all_holdings = self.broker.get_holdings() # Returns Optional[Dict[str, Decimal]]
                base_symbol = self.symbol.split('-')[0] # e.g., 'BTC' from 'BTC-USD'
                if all_holdings and base_symbol in all_holdings:
                    self.crypto_holdings = float(all_holdings[base_symbol])
                else:
                    self.crypto_holdings = 0.0
                
                self.initial_balance = self.current_capital # Reset initial capital to current live capital
                self.logger.info(f"Live state fetched: Capital={self.current_capital:.2f}, Holdings={self.crypto_holdings:.6f}")

                # --- Fetch Initial Live Data for Observation --- 
                self.logger.info("Fetching recent market data for initial observation...")
                # Use the DATA PROVIDER's fetch_price_history method, NOT the broker's
                # Ensure the arguments match the fetch_price_history signature (symbol, days, interval)
                live_data_df = self.data_provider.fetch_price_history(self.symbol, days=self.lookback_window, interval='1h')
                
                if live_data_df is None or live_data_df.empty or len(live_data_df) < self.lookback_window:
                     msg = f"Failed to fetch sufficient live data via data_provider ({len(live_data_df) if live_data_df is not None else 0}/{self.lookback_window}) for initial observation."
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
                # Use get_latest_price to get the initial valuation price
                current_price_decimal = self.broker.get_latest_price(self.symbol)
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
            self.current_capital = self.initial_balance
            self.crypto_holdings = 0
            self.portfolio_value = self.initial_balance
            self.max_portfolio_value = self.initial_balance

            # --- Reset Differential Sharpe Ratio tracking --- 
            self.portfolio_history = [self.initial_balance] * self.lookback_window # Start history with initial value
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
