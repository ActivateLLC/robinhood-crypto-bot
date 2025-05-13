import logging
import time
import schedule
import os
import pandas as pd
import numpy as np
import pandas_ta as ta # Import pandas_ta
import matplotlib.pyplot as plt
import pickle
from decimal import Decimal
import uuid # For client_order_id
import json
from dotenv import load_dotenv
from stable_baselines3 import PPO
from brokers.base_broker import BaseBroker
from brokers.robinhood_broker import RobinhoodBroker
from typing import Dict, Any, Optional, List, Tuple, Union

# Load environment variables
load_dotenv()

# Local imports
from config import load_config, setup_logging, Config
from alt_crypto_data import AltCryptoDataProvider # Import only the class

# Define constants used for RL feature processing directly here
OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume'] # Standard OHLCV columns
# Features used for RL model input (MUST match training environment)
FEATURE_COLS = [
    'open', 'high', 'low', 'close' # Assuming model was trained only on OHLC
]

# Define the 27 technical indicators our model expects (must match CryptoTradingEnv)
TECHNICAL_INDICATOR_COLUMNS = [
    'sma_10', 'sma_30', 'ema_10', 'ema_30', 'ema_50',
    'price_above_ema50', 'volatility_10', 'volatility_30',
    'volume_sma_20', 'volume_ratio', 'ha_close', 'ha_open',
    'ha_high', 'ha_low', 'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_middle', 'bb_upper', 'bb_lower', 'kc_middle', 'kc_upper',
    'kc_lower', 'ttm_squeeze', 'ttm_momentum', 'ttm_signal'
]

# Default mapping from RL integer actions to trade signals
DEFAULT_RL_ACTIONS_MAP = {0: 'sell', 1: 'hold', 2: 'buy'}
# Define the expected observation space shape for the RL model
# This might include flattened historical data + current balance + current holding
# Example: (lookback_window * num_features) + 2
# Adjust this based on your actual feature engineering
EXPECTED_OBS_SHAPE = (31,) # Based on model error: 60 lookback * 4 features + 2 state vars

class RobinhoodCryptoBot:
    """
    Automated cryptocurrency trading bot for Robinhood using RL or traditional strategies.
    """
    def __init__(self):
        # --- CRITICAL: Setup Logging FIRST --- #
        # Load config first to pass to logging setup
        self.config = load_config()
        
        setup_logging(self.config) # Call setup_logging IMMEDIATELY after loading config
        # ------------------------------------ #

        # Now safe to make logging calls
        logging.info("--- Initializing Crypto Bot (Revised Logging) --- ")

        logging.info("Loading configuration...") # This is somewhat redundant now, but ok
        self.symbols = self.config.SYMBOLS_TO_TRADE
        self.enable_trading = self.config.ENABLE_TRADING
        self.trade_amount_usd = self.config.TRADE_AMOUNT_USD
        self.lookback_days = self.config.LOOKBACK_DAYS
        self.check_interval = self.config.INTERVAL_MINUTES
        self.trading_strategy = self.config.TRADING_STRATEGY
        self.rl_model_path = self.config.RL_MODEL_PATH
        self.interval = self.check_interval # Assign self.interval = self.check_interval
        self.plot_enabled = self.config.PLOT_ENABLED # Added from view
        self.plot_output_dir = self.config.PLOT_OUTPUT_DIR # Added from view

        # Initialize placeholders
        self.broker: Optional[BaseBroker] = None
        self.rl_agent = None
        self.data_provider = None
        self.symbol_map = {}
        self.historical_data = {}
        self.holdings = {'USD': {'quantity': Decimal('0')}}
        self.rl_lookback_window = self.config.RL_LOOKBACK_WINDOW # Added from view
        self.rl_features_mean = None # Added from view
        self.rl_features_std = None # Added from view
        self.norm_stats = {} # Initialize dict to store normalization stats per symbol
        self.current_holdings: Dict[str, Decimal] = {symbol.split('-')[0]: Decimal('0.0') for symbol in self.config.SYMBOLS_TO_TRADE} # Initialize holdings keyed by base asset (e.g., 'BTC') using SYMBOLS_TO_TRADE
        self.data_cache = {}
        self.actions_map = DEFAULT_RL_ACTIONS_MAP
        self.experience_logger = logging.getLogger('experience_logger') # Get the experience logger

        logging.info(f"Configuration loaded: Symbols={self.symbols}, Strategy={self.trading_strategy}, Trading Enabled={self.enable_trading}")

        # --- Complex Initializations ---

        # 1. Conditionally load RL agent
        if self.config.ENABLE_RL_MODEL and self.trading_strategy == 'rl': # Check both config flags
            logging.info(f"RL strategy selected. Attempting to load model from: {self.rl_model_path}")
            try:
                from stable_baselines3 import PPO # Or your specific agent
                if os.path.exists(self.rl_model_path):
                    self.rl_agent = PPO.load(self.rl_model_path) # Load the agent
                    logging.info("RL Agent loaded successfully.")
                    # Define the mapping from RL action index to signal string
                    self.rl_actions_map = {
                        0: 'buy_all',
                        1: 'buy_half',
                        2: 'hold',
                        3: 'sell_half',
                        4: 'sell_all'
                    }
                    # Also load normalization params if they exist
                    # norm_params_path = self.rl_model_path + "_norm.pkl"
                    # if os.path.exists(norm_params_path):
                    #    with open(norm_params_path, 'rb') as f:
                    #       self.rl_features_mean, self.rl_features_std = pickle.load(f)
                    #       logging.info("Normalization parameters loaded.")
                else:
                    logging.error(f"RL model file not found at {self.rl_model_path}. RL strategy disabled.")
                    self.trading_strategy = 'hold' # Fallback
                    self.rl_agent = None # Add check
                    logging.error("RL model failed to load. RL strategy will not function.")
            except ImportError:
                logging.error("stable_baselines3 not found. Please install it ('pip install stable-baselines3') to use the RL strategy. Falling back to hold.")
                self.trading_strategy = 'hold' # Fallback
                self.rl_agent = None # Add check
                logging.error("RL model failed to load. RL strategy will not function.")
            except Exception as e:
                logging.exception(f"Error loading RL agent: {e}. Falling back to hold.")
                self.trading_strategy = 'hold' # Fallback
                self.rl_agent = None # Add check
                logging.error("RL model failed to load. RL strategy will not function.")

        # 2. Initialize Broker Client (using the interface)
        self.broker: Optional[BaseBroker] = None # Initialize broker attribute
        if self.enable_trading:
            # --- Broker Selection Logic (Future Enhancement) ---
            # Currently hardcoded to Robinhood, later select based on config
            broker_type = self.config.BROKER_TYPE.lower() # Use self.config
            if broker_type == 'robinhood':
                broker_config = {
                    'api_key': self.config.ROBINHOOD_API_KEY,
                    'base64_private_key': self.config.ROBINHOOD_BASE64_PRIVATE_KEY
                }
                self.broker = self._initialize_broker(RobinhoodBroker, broker_config)
            # elif broker_type == 'ibkr':
            else:
                logger.error(f"Unsupported broker type specified in config: {self.config.BROKER_TYPE}")
                self.enable_trading = False

            # If broker initialization failed, disable trading
            if self.broker is None:
                self.enable_trading = False
            else:
                # Attempt to connect the broker
                if not self.broker.connect():
                    logging.error(f"Failed to connect to {broker_type} broker. Disabling trading.")
                    self.enable_trading = False

        # 3. Initialize Data Provider
        try:
            logging.info(f"Initializing data provider (Preference: {self.config.DATA_PROVIDER_PREFERENCE})...")
            self.data_provider = AltCryptoDataProvider() # Still need this if RL uses it?
            logging.info("Data provider initialized.")

            # 4. Fetch initial historical data and calculate normalization stats if using RL
            if self.trading_strategy == 'rl':
                logging.info("Calculating initial normalization statistics...")
                self._calculate_and_store_norm_stats() # <-- Correct method to fetch data and calc stats
                if not self.norm_stats:
                    logging.error("Failed to calculate initial normalization stats. RL may not function correctly.")
                else:
                    symbols_with_stats = list(self.norm_stats.keys())
                    logging.info(f"Normalization stats calculated for: {symbols_with_stats}")
                    missing_stats = [s for s in self.symbols if s not in self.norm_stats]
                    if missing_stats:
                        logging.warning(f"Could not calculate normalization stats for: {missing_stats}")
        except Exception as e:
            logging.exception(f"Failed to initialize data provider: {e}")
            self.data_provider = None

        # 5. Fetch initial holdings (Requires API Client)
        if self.broker:
            logging.info("Fetching initial holdings...")
            self.get_holdings() # <-- Correct method name (matches API client method)
        else:
            logging.warning("Skipping initial holdings fetch: API client not available or trading disabled.")
            self.holdings = {} # Initialize as empty if not fetched
        logging.info(f"RobinhoodCryptoBot __init__ finished. Trading Enabled: {self.enable_trading}")

    def _initialize_broker(self, broker_class: type[BaseBroker], broker_config: Dict[str, Any]) -> Optional[BaseBroker]:
        broker_type = broker_class.__name__
        try:
            # Get a logger specific to the broker being initialized
            broker_logger = logging.getLogger(f"brokers.{broker_type}")
            logging.info(f"Initializing broker (Type: {broker_type})...")
            if broker_type == 'RobinhoodBroker':
                broker_logger = logging.getLogger("brokers.robinhood_broker")
                broker = broker_class(broker_config, broker_logger) # Pass the correct logger
            else:
                broker = broker_class(broker_config, broker_logger)
            logging.info(f"{broker_type} initialized successfully.")
            return broker
        except Exception as e:
            logging.exception(f"Failed to initialize {broker_type}:")
            return None

    def _calculate_and_store_norm_stats(self):
        """
        Calculates and stores normalization statistics (mean, std) for the 27 technical indicators.
        """
        logging.info("Calculating initial normalization statistics for 27 RL technical indicators...")
        self.norm_stats = {} # Reset stats

        # Min rows for indicators (e.g., longest lookback for any single indicator like EMA50 or Volatility30 + some buffer)
        # This is for the data *before* passing to _add_all_technical_indicators
        min_rows_for_ohlcv_fetch = 90 # Fetch enough raw data for all indicators to warm up (e.g. 60 for indicators + 30 for some stddev)

        logging.info(f"Starting normalization stats calculation for symbols: {self.symbols}")
        for symbol in self.symbols:
            logging.debug(f"Processing normalization stats for symbol: {symbol}")
            try:
                # Fetch historical OHLCV data for the symbol
                df_ohlcv_history = self.data_provider.fetch_price_history(symbol, lookback_days=min_rows_for_ohlcv_fetch)

                if df_ohlcv_history is None or df_ohlcv_history.empty:
                    logging.warning(f"No historical OHLCV data found for {symbol}. Skipping normalization stat calculation.")
                    continue
                
                # It's good practice to ensure OHLCV_COLS are present before passing to helper, though helper also checks
                if not all(col.lower() in [c.lower() for c in df_ohlcv_history.columns] for col in OHLCV_COLS):
                    logging.warning(f"Fetched historical data for {symbol} is missing one or more OHLCV columns. Skipping normalization.")
                    continue
                
                # Ensure df_ohlcv_history has enough rows for meaningful indicator calculation by the helper
                # The helper itself needs a buffer; this check is for the input to the helper.
                # A general rule could be min_rows_for_ohlcv_fetch / 2 or a fixed number like 50-60.
                if len(df_ohlcv_history) < 60: # Arbitrary minimum, adjust based on longest indicator period in helper
                    logging.warning(f"Insufficient historical OHLCV data points for {symbol} ({len(df_ohlcv_history)} points, need ~60). Skipping normalization.")
                    continue

                # Add all technical indicators using the helper. Pass a copy.
                df_with_indicators = self._add_all_technical_indicators(df_ohlcv_history.copy())

                # Select only the technical indicator columns for stat calculation
                df_technical_features = df_with_indicators[TECHNICAL_INDICATOR_COLUMNS].copy()
                
                # Drop rows with any NaN values that might remain in technical features
                # (e.g., from initial warm-up of the longest indicator, despite helper's bfill/fillna(0))
                df_technical_features.dropna(inplace=True)

                # Check if enough data remains AFTER dropping NaNs for robust statistics
                # We need at least a few data points. rl_lookback_window is a good reference, or a fixed minimum (e.g., 30).
                min_data_points_for_stats = max(30, self.rl_lookback_window) 
                if len(df_technical_features) < min_data_points_for_stats:
                    logging.warning(f"Insufficient data for {symbol} after indicator calculation & NaN drop ({len(df_technical_features)} points, need {min_data_points_for_stats}). Cannot calc norm stats.")
                    continue

                means = df_technical_features.mean()
                stds = df_technical_features.std()
                stds[stds == 0] = 1e-8 # Avoid division by zero

                self.norm_stats[symbol] = {'means': means, 'stds': stds}
                logging.info(f"Successfully calculated normalization stats for {symbol} using {len(df_technical_features)} data points.")

            except Exception as e:
                logging.exception(f"Error calculating/storing normalization stats for {symbol}:")
                if symbol in self.norm_stats: # Clean up if partial calculation failed
                    del self.norm_stats[symbol]

        logging.info("Finished calculating initial normalization statistics.")
        if not self.norm_stats:
            logging.error("Failed to calculate normalization statistics for ANY symbol. RL strategy will likely fail.")
        else:
             symbols_with_stats = list(self.norm_stats.keys())
             logging.info(f"Normalization stats calculated for: {symbols_with_stats}")
             missing_stats = [s for s in self.symbols if s not in self.norm_stats]
             if missing_stats:
                 logging.warning(f"Could not calculate normalization stats for: {missing_stats}")

    def _add_all_technical_indicators(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Adds all 27 required technical indicators to the input DataFrame.
        Assumes df_ohlcv has 'open', 'high', 'low', 'close', 'volume' columns (case-insensitive, will be lowercased).
        Returns DataFrame with added indicators and placeholders for any that failed.
        """
        df_processed = df_ohlcv.copy()
        df_processed.columns = [col.lower() for col in df_processed.columns]

        required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df_processed.columns for col in required_ohlcv):
            missing_ohlcv = [col for col in required_ohlcv if col not in df_processed.columns]
            logging.warning(f"Input to _add_all_technical_indicators missing OHLCV columns: {missing_ohlcv}. Adding placeholders.")
            for col in missing_ohlcv:
                df_processed[col] = 0.0 # Add placeholder for missing OHLCV
        
        # Ensure critical 'close' column exists for most indicators
        if 'close' not in df_processed.columns or df_processed['close'].isnull().all():
            logging.error("Critical 'close' column is missing or all NaN in _add_all_technical_indicators. Most indicators will fail.")
            # Add placeholders for all technical indicators and return
            for col_name in TECHNICAL_INDICATOR_COLUMNS:
                if col_name not in df_processed.columns: df_processed[col_name] = 0.0
            return df_processed

        # Calculate indicators using pandas_ta
        # Explicitly pass column names if they are not the pandas_ta defaults (e.g. if not all lowercase)
        df_processed.ta.sma(length=10, close='close', append=True)
        df_processed.ta.sma(length=30, close='close', append=True)
        df_processed.ta.ema(length=10, close='close', append=True)
        df_processed.ta.ema(length=30, close='close', append=True)
        df_processed.ta.ema(length=50, close='close', append=True)
        df_processed.ta.rsi(length=14, close='close', append=True)
        df_processed.ta.macd(fast=12, slow=26, signal=9, close='close', append=True)
        df_processed.ta.bbands(length=20, std=2, close='close', append=True)

        rename_map = {
            'SMA_10': 'sma_10', 'SMA_30': 'sma_30',
            'EMA_10': 'ema_10', 'EMA_30': 'ema_30', 'EMA_50': 'ema_50',
            'RSI_14': 'rsi',
            'MACD_12_26_9': 'macd', 'MACDh_12_26_9': 'macd_hist', 'MACDs_12_26_9': 'macd_signal',
            'BBM_20_2.0': 'bb_middle', 'BBU_20_2.0': 'bb_upper', 'BBL_20_2.0': 'bb_lower'
        }
        df_processed.rename(columns=rename_map, inplace=True)

        # Volatility
        df_processed['volatility_10'] = df_processed['close'].rolling(window=10).std()
        df_processed['volatility_30'] = df_processed['close'].rolling(window=30).std()

        # Volume Analysis
        if 'volume' in df_processed.columns:
            df_processed['volume_sma_20'] = df_processed['volume'].rolling(window=20).mean()
            if 'volume_sma_20' in df_processed.columns and df_processed['volume_sma_20'].notna().any():
                 df_processed['volume_ratio'] = df_processed['volume'] / df_processed['volume_sma_20']
                 df_processed['volume_ratio'].replace([np.inf, -np.inf], 1.0, inplace=True)
                 df_processed['volume_ratio'].fillna(1.0, inplace=True)
            else:
                df_processed['volume_ratio'] = 1.0 # Placeholder
        else:
            df_processed['volume_sma_20'] = 0.0 # Placeholder
            df_processed['volume_ratio'] = 1.0 # Placeholder

        # Heikin-Ashi
        if all(c in df_processed.columns for c in ['open', 'high', 'low', 'close']):
            df_processed.ta.ha(open='open', high='high', low='low', close='close', append=True)
            ha_rename_map = {'HA_open': 'ha_open', 'HA_high': 'ha_high', 'HA_low': 'ha_low', 'HA_close': 'ha_close'}
            df_processed.rename(columns=ha_rename_map, inplace=True)
        
        # Keltner Channels
        if all(c in df_processed.columns for c in ['high', 'low', 'close']):
            df_processed.ta.kc(length=20, scalar=1.5, mamode='ema', atr_length=20, high='high', low='low', close='close', append=True)
            kc_rename_map_actual = {}
            for col in df_processed.columns: # Robust renaming based on actual generated names
                if col.startswith('KCU_'): kc_rename_map_actual[col] = 'kc_upper'
                elif col.startswith('KCL_'): kc_rename_map_actual[col] = 'kc_lower'
                elif col.startswith('KCM_'): kc_rename_map_actual[col] = 'kc_middle'
            df_processed.rename(columns=kc_rename_map_actual, inplace=True)

        # TTM Squeeze-like
        if all(col in df_processed.columns for col in ['bb_lower', 'kc_lower', 'bb_upper', 'kc_upper']):
            df_processed['ttm_squeeze'] = ((df_processed['bb_lower'] > df_processed['kc_lower']) & \
                                           (df_processed['bb_upper'] < df_processed['kc_upper'])).astype(int)
        
        df_processed['ttm_momentum'] = df_processed['close'] - df_processed['close'].rolling(window=12).mean()
        
        if 'ttm_squeeze' in df_processed.columns and 'ttm_momentum' in df_processed.columns:
            df_processed['ttm_signal'] = 0 # Default to 0
            df_processed.loc[(df_processed['ttm_squeeze'] == 1) & (df_processed['ttm_momentum'] > 0), 'ttm_signal'] = 1
            df_processed.loc[(df_processed['ttm_squeeze'] == 1) & (df_processed['ttm_momentum'] < 0), 'ttm_signal'] = -1
        
        # price_above_ema50
        if 'ema_50' in df_processed.columns:
             df_processed['price_above_ema50'] = (df_processed['close'] > df_processed['ema_50']).astype(int)
        
        # Ensure all TECHNICAL_INDICATOR_COLUMNS exist, add placeholder 0.0 if not & fill NaNs
        for col_name in TECHNICAL_INDICATOR_COLUMNS:
            if col_name not in df_processed.columns:
                logging.debug(f"Helper: Indicator '{col_name}' missing. Adding placeholder 0.0.")
                df_processed[col_name] = 0.0
            else:
                # Fill NaNs that might have occurred from rolling operations at the beginning
                df_processed[col_name].fillna(method='bfill', inplace=True) # Backfill first
                df_processed[col_name].fillna(0.0, inplace=True) # Then fill remaining NaNs with 0
                
        return df_processed

    def _get_normalized_features(self, symbol: str, df_current_history: pd.DataFrame) -> Optional[np.ndarray]:
        """Adds indicators, normalizes features, appends portfolio state for the 31-feature model.
        
        Args:
            symbol: The trading symbol (e.g., 'BTC-USD').
            df_current_history: DataFrame with current historical OHLCV data. 
                                Must have enough rows for indicator calculation.
                                (e.g., ~60 rows for indicators to mature before taking the latest readings)
        Returns:
            A 1D NumPy array of shape (31,) representing the normalized observation, or None if error.
        """
        if symbol not in self.norm_stats:
            logging.error(f"Normalization stats not found for {symbol}. Cannot normalize features.")
            return None
        
        stats = self.norm_stats[symbol]
        # Ensure means and stds are Series for proper alignment if they came from single-row DataFrame initially
        means = pd.Series(stats['means']) if isinstance(stats['means'], dict) else stats['means']
        stds = pd.Series(stats['stds']) if isinstance(stats['stds'], dict) else stats['stds']

        # Min rows needed for indicator calculation before taking the latest reading.
        # The input df_current_history should provide at least this many rows.
        min_rows_for_indicator_calc = 60 # Should be consistent with buffer in _calculate_and_store_norm_stats / _add_all_technical_indicators
        if df_current_history is None or len(df_current_history) < min_rows_for_indicator_calc:
            logging.warning(f"Input df_current_history for {symbol} has {len(df_current_history) if df_current_history is not None else 0} rows, need at least {min_rows_for_indicator_calc} for indicator calc. Cannot get features.")
            return None

        # 1. Add all technical indicators using the helper
        # df_current_history should be OHLCV data.
        df_with_indicators = self._add_all_technical_indicators(df_current_history.copy())

        # 2. Select the LATEST row of technical features after indicators are calculated.
        # The indicators themselves (like SMA_30) incorporate the lookback period.
        if df_with_indicators.empty or not all(col in df_with_indicators.columns for col in TECHNICAL_INDICATOR_COLUMNS):
            logging.warning(f"Not enough data or missing columns for {symbol} after adding indicators. Columns: {df_with_indicators.columns.tolist()}")
            return None
            
        latest_technical_features_unnormalized = df_with_indicators[TECHNICAL_INDICATOR_COLUMNS].iloc[-1].copy()

        # 3. Normalize the selected latest technical features
        # Ensure all columns expected by norm_stats (means/stds Series) are present
        for col in TECHNICAL_INDICATOR_COLUMNS:
            if col not in latest_technical_features_unnormalized.index: # It's a Series now
                logging.error(f"Missing technical feature column '{col}' in latest_technical_features_unnormalized for {symbol}. Adding placeholder 0.0.")
                latest_technical_features_unnormalized[col] = 0.0
            if col not in means.index or col not in stds.index: # Check against Series index
                logging.error(f"Normalization stats (mean/std Series) missing for feature '{col}' for {symbol}. Using 0 for mean, 1 for std.")
                # Ensure the key exists in means/stds Series before assignment if it's truly missing
                if col not in means.index: means[col] = 0.0
                if col not in stds.index: stds[col] = 1.0
        
        # Reorder to match means/stds (which should be in TECHNICAL_INDICATOR_COLUMNS order)
        latest_technical_features_unnormalized = latest_technical_features_unnormalized.reindex(TECHNICAL_INDICATOR_COLUMNS, fill_value=0.0)
        means = means.reindex(TECHNICAL_INDICATOR_COLUMNS, fill_value=0.0)
        stds = stds.reindex(TECHNICAL_INDICATOR_COLUMNS, fill_value=1.0).replace(0, 1e-8) # ensure no zero std

        normalized_technical_features_latest = (latest_technical_features_unnormalized - means) / stds
        normalized_technical_features_latest.fillna(0.0, inplace=True) # Fill any NaNs from division by tiny std

        # 4. Get current portfolio state (4 features)
        base_asset = symbol.split('-')[0]
        current_balance_usd = self.get_balance_usd() 
        current_holding_crypto = self.get_holding_crypto(base_asset) # This should return Decimal
        
        latest_close_price_series = df_current_history['close'] if 'close' in df_current_history else pd.Series([0.0])
        latest_close_price = latest_close_price_series.iloc[-1] if not latest_close_price_series.empty else 0.0
        
        total_portfolio_value = current_balance_usd + (current_holding_crypto * Decimal(str(latest_close_price)))

        portfolio_features = np.array([
            float(current_balance_usd),
            float(current_holding_crypto),
            float(latest_close_price), 
            float(total_portfolio_value)
        ], dtype=np.float32)

        # 5. Concatenate technical and portfolio features
        # Ensure normalized_technical_features_latest is a NumPy array for concatenation
        final_observation = np.concatenate((normalized_technical_features_latest.values, portfolio_features))

        if final_observation.shape != self.EXPECTED_OBS_SHAPE:
            logging.error(f"Shape mismatch for {symbol}: Expected {self.EXPECTED_OBS_SHAPE}, got {final_observation.shape}. Tech features: {normalized_technical_features_latest.values.shape}, Portfolio: {portfolio_features.shape}")
            return None 

        logging.debug(f"Generated normalized features for {symbol}. Shape: {final_observation.shape}")
        return final_observation

    def get_current_price(self, symbol: str) -> Optional[Decimal]:
        # Implement this method as needed
        pass

    def _get_signal(self, symbol: str, analysis_data: pd.DataFrame):
        """
        Determine trading signal for a given cryptocurrency
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC-USD')
            analysis_data (pd.DataFrame): DataFrame containing historical data for analysis.
        
        Returns:
            str: Trading signal (buy/sell/hold)
        """
        signal = 'hold'  # Default signal

        if self.trading_strategy == 'rl' and self.rl_agent is not None:
            logging.debug(f"Generating signal for {symbol} using RL agent.")
            try:
                # 1. Prepare features
                obs = self._get_normalized_features(symbol, analysis_data)
                if obs is None:
                    logging.warning(f"Could not generate normalized features for {symbol}. Defaulting to 'hold'.")
                    return 'hold'

                # Check observation shape (optional but good practice)
                if obs.shape[0] != EXPECTED_OBS_SHAPE[0]:
                    logging.warning(f"Observation shape mismatch for {symbol}. Expected {EXPECTED_OBS_SHAPE}, Got {obs.shape}. Defaulting to 'hold'.")
                    # Potentially pad or handle this case depending on model requirements
                    return 'hold'

                # 2. Predict action
                action, _ = self.rl_agent.predict(obs, deterministic=True)
                signal = self.actions_map.get(action.item(), 'hold') # Use .item() for numpy int
                logging.info(f"[{symbol}] RL agent signal: {signal} (action_int: {action.item()})")

                # --- Log Experience (Action Determined) ---
                self._log_experience(
                    symbol=symbol,
                    state=obs, # Log the state used for prediction
                    action=signal,
                    reward=0.0, # Placeholder for now
                    pnl_change=0.0, # Placeholder for now
                    portfolio_value=0.0 # Placeholder for now
                )
                return signal
            except Exception as e:
                logging.exception(f"Error during RL signal generation for {symbol}: {e}")
                return 'hold' # Default to hold on error
        elif self.trading_strategy == 'rsi': # Example: Add other strategies
            logging.debug(f"Generating signal for {symbol} using RSI strategy (placeholder).")
            # Placeholder for RSI strategy logic
            # rsi_value = analysis_data['RSI_14'].iloc[-1]
            # if rsi_value < 30: return 'buy'
            # elif rsi_value > 70: return 'sell'
            return 'hold'
        else:
            logging.warning(f"Unknown or unsupported trading strategy: '{self.trading_strategy}'. Defaulting to 'hold' for {symbol}.")
            return 'hold'
 
     # --- Trade Execution ---
    def _execute_trade(self, symbol: str, signal: str, latest_price: Decimal):
        logging.debug(f"Executing trade for {symbol}...")
        """
        Execute trade based on signal
        
        Args:
            symbol (str): Trading symbol
            signal (str): Trading signal
            latest_price (Decimal): The latest price used for signal generation & quantity calculation.
        """
        # Ensure trading is enabled
        if not self.enable_trading:
            logging.info("Trading is disabled. Skipping trade execution.")
            return

        # Ensure API client is available
        if not self.broker:
            logging.error("Broker not initialized. Cannot execute trade.")
            return

        # 1. Determine Side and Calculate Quantity
        side = None
        quantity = Decimal('0.0')
        trade_amount = Decimal(str(self.trade_amount_usd))
        symbol_base = symbol.split('-')[0]

        try:
            if latest_price <= 0:
                logging.error(f"Invalid latest price ({latest_price}) for {symbol}. Skipping trade.")
                return

            # Get current holdings
            current_holding_qty = Decimal(self.holdings.get(symbol_base, Decimal('0.0')))

            if signal == 'buy_all':
                side = 'buy'
                quantity = trade_amount / latest_price
            elif signal == 'buy_half':
                side = 'buy'
                quantity = (trade_amount / Decimal('2')) / latest_price
            elif signal == 'sell_all':
                side = 'sell'
                quantity = current_holding_qty
            elif signal == 'sell_half':
                side = 'sell'
                quantity = current_holding_qty / Decimal('2')

        except Exception as e:
            logging.exception(f"Error calculating quantity for {symbol} signal {signal}: {e}")
            return

        # 2. Validate Trade
        if side is None or quantity <= Decimal('0.00000001'): # Check against a minimum tradeable quantity
            logging.warning(f"Invalid trade parameters for {symbol}: side={side}, calculated quantity={quantity:.8f}. Signal was '{signal}'. Skipping trade.")
            return

        if side == 'sell' and quantity > current_holding_qty:
            logging.warning(f"Attempting to sell {quantity:.8f} {symbol_base}, but only hold {current_holding_qty:.8f}. Adjusting to sell max available.")
            quantity = current_holding_qty
            if quantity <= Decimal('0.00000001'): # Double check after adjustment
                logging.warning(f"No holdings ({current_holding_qty:.8f}) of {symbol_base} to sell after adjustment. Skipping trade.")
                return

        # 3. Prepare and Place Order (Market Order)
        order_details = self.broker.place_market_order(
            symbol=symbol,      # e.g., 'BTC-USD'
            side=signal,        # 'buy' or 'sell'
            quantity=quantity   # Decimal quantity
        )

        if order_details and order_details.get('status') not in ['failed', 'rejected', 'cancelled']:
            # Basic check for success, might need refinement based on states
            order_id = order_details.get('order_id')
            logging.info(f"Trade submitted successfully via broker. Order ID: {order_id}")
        else:
            logging.error(f"Trade submission failed or was rejected by broker. Response: {order_details}")

    def get_holdings(self):
        """Fetches current crypto holdings using the broker client and updates self.holdings."""
        if not self.broker:
            logging.warning("Attempted to get holdings, but broker is not initialized.")
            self.holdings = {}
            return

        logging.info("Fetching current holdings via broker client...")
        try:
            holdings_data = self.broker.get_holdings()
            if holdings_data is not None: # Expects Dict[str, Decimal] or None
                self.holdings = holdings_data
                logging.info(f"Successfully retrieved {len(self.holdings)} holdings: {list(self.holdings.keys())}")
            else:
                logging.warning("Received no data or empty response when fetching holdings.")
                self.holdings = {}

        except Exception as e:
            logging.error(f"Failed to fetch holdings: {e}", exc_info=True)
            self.holdings = {} # Ensure holdings is defined even on error

    def _fetch_latest_data(self):
        """Fetches the latest market data for tracked symbols."""
        logging.debug("Fetching latest market data...")
        # Implement this method as needed

    def run_trading_cycle(self):
        """Executes one cycle of fetching data, analyzing, and potentially trading."""
        logging.info("Starting new trading cycle...")

        # 1. Fetch latest data (you might need a separate method for this)
        # self._fetch_latest_data() # Assuming this populates self.historical_data or similar
        # For now, let's assume _calculate_and_store_norm_stats got enough data initially
        # Or, perhaps fetch data inside analyze_symbol?

        # 2. Fetch current holdings
        if self.broker:
            self.get_holdings() # Update holdings each cycle

        # 3. Analyze each symbol and get signals
        signals = {}
        for symbol in self.config.SYMBOLS_TO_TRADE: # Use config list
            # Fetch or retrieve the necessary data for analysis
            # This might involve calling self.data_provider.get_historical_data again
            # or using data already stored in self.historical_data
            # For RL, we need at least lookback_window length of data ending now.
            # Placeholder: Assuming data is magically available for now
            logging.debug(f"Fetching data for analysis for {symbol}...")
            # Example Fetch (needs proper start/end dates):
            # analysis_data = self.data_provider.get_historical_data(symbol, ..., interval=self.interval)
            # For now, skipping actual data fetch in cycle, relying on initial load for demo
            # analysis_data = self.historical_data.get(symbol) # Use initially fetched if available

            # Fetch fresh data for analysis in each cycle
            analysis_days_needed = self.rl_lookback_window + 35 # Lookback + indicator buffer
            analysis_data = self.data_provider.fetch_price_history(symbol, analysis_days_needed)

            if analysis_data is None or analysis_data.empty:
                 logging.warning(f"No data available for {symbol} to generate signal. Skipping.")
                 signals[symbol] = 'hold' # Default to hold if no data
                 continue

            signals[symbol] = self._get_signal(symbol, analysis_data)

        # 4. Execute trades based on signals
        if self.enable_trading:
            for symbol, signal in signals.items():
                if signal != 'hold':
                    logging.info(f"Signal for {symbol}: {signal}. Attempting to execute trade...")
                    try:
                        # Get the latest price from the analysis data used for the signal
                        latest_price_for_trade = Decimal(str(analysis_data['close'].iloc[-1]))
                        if latest_price_for_trade <= 0:
                            logging.error(f"Invalid latest price ({latest_price_for_trade}) from analysis data for {symbol}. Skipping trade.")
                            continue

                        self._execute_trade(symbol, signal, latest_price_for_trade)
                    except Exception as e:
                        logging.exception(f"Error executing trade for {symbol} with signal {signal}:")
        else:
            logging.info("Trading is disabled. Skipping trade execution.")

        logging.info(f"Trading cycle finished. Signals: {signals}")

    def run(self):
        """Runs the main bot loop, scheduling trading cycles."""
        logging.info(f"Starting main bot run loop. Checking every {self.check_interval} minutes.")

        # Run the first cycle immediately
        self.run_trading_cycle()

        # Schedule subsequent cycles
        schedule.every(self.check_interval).minutes.do(self.run_trading_cycle)

        while True:
            schedule.run_pending()
            time.sleep(1)

    # --- Experience Logging --- #
    def _log_experience(self, symbol: str, state: np.ndarray, action: str, reward: float, pnl_change: float, portfolio_value: float):
        """Logs the agent's experience to a dedicated file for later processing."""
        try:
            timestamp = pd.Timestamp.now(tz='UTC').isoformat() # Use ISO format timestamp
            
            # Ensure state is serializable (convert numpy array to list)
            serializable_state = state.tolist() if isinstance(state, np.ndarray) else state

            experience_data = {
                'timestamp': timestamp,
                'symbol': symbol,
                'state': serializable_state, # Log the state used for prediction
                'action': action,
                'reward': reward, # Placeholder for now
                'pnl_change': pnl_change, # Placeholder for now
                'portfolio_value': portfolio_value # Placeholder for now
                # Add 'next_state' later if needed for certain RL updates
            }
            
            # Log as a JSON string
            self.experience_logger.info(json.dumps(experience_data))

        except Exception as e:
            logging.error(f"Error logging experience data: {e}", exc_info=True)

# Main execution block
if __name__ == "__main__":
    logging.info("Starting bot execution...")
    try:
        bot = RobinhoodCryptoBot()
        bot.run()
    except KeyboardInterrupt:
        logging.info("Bot execution stopped manually.")
        print("\nBot stopped.")
    except Exception as e:
        logging.exception("An unexpected error occurred during bot execution:")
        print(f"An unexpected error occurred: {e}")
    finally:
        logging.info("Bot execution finished.")
