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

# Default mapping from RL integer actions to trade signals
DEFAULT_RL_ACTIONS_MAP = {0: 'sell', 1: 'hold', 2: 'buy'}
# Define the expected observation space shape for the RL model
# This might include flattened historical data + current balance + current holding
# Example: (lookback_window * num_features) + 2
# Adjust this based on your actual feature engineering
EXPECTED_OBS_SHAPE = (60 * 4) + 2 # Based on model error: 60 lookback * 4 features + 2 state vars

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
                # self.fetch_all_historical_data() # Make sure this method exists <- Incorrect name
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
            # self.fetch_holdings() # Make sure this method exists <- Incorrect name
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
        Calculates and stores normalization statistics (mean, std) for each symbol
        based on the initial historical data fetched.
        """
        logging.info("Calculating initial normalization statistics for RL features...")
        self.norm_stats = {} # Reset stats

        min_required_data = self.rl_lookback_window + 35 # Lookback + buffer for indicators (e.g., MACD slow+signal)

        logging.info(f"Starting normalization loop for symbols: {self.symbols}")
        for symbol in self.symbols:
            logging.debug(f"Processing normalization stats for symbol: {symbol}")
            try:
                # Fetch historical data for the symbol
                lookback_days = 90 # Example: Fetch 90 days of data

                df = self.data_provider.fetch_price_history(symbol, lookback_days) # Correct call

                if df is None or df.empty:
                    logging.warning(f"No historical data found for {symbol} via fetch_price_history. Skipping normalization stat calculation.") # Updated log message
                    continue

                if len(df) < min_required_data:
                    logging.warning(f"Insufficient historical data for {symbol} ({len(df)} points, need {min_required_data}) to calculate robust normalization stats. Skipping.")
                    continue

                try:
                    # Work on a copy to avoid modifying the original historical data
                    df_processed = df[OHLCV_COLS].copy()

                    # Add indicators
                    df_processed.ta.rsi(length=14, append=True)
                    df_processed.ta.macd(fast=12, slow=26, signal=9, append=True)
                    df_processed.ta.bbands(length=20, std=2, append=True)

                    # Select only feature columns AFTER adding indicators
                    df_processed = df_processed[FEATURE_COLS].copy()

                    # Drop rows with NaN values (generated by indicators needing warmup)
                    df_processed.dropna(inplace=True)

                    # Check if enough data remains AFTER dropping NaNs
                    if len(df_processed) < self.rl_lookback_window:
                        logging.warning(f"Insufficient data for {symbol} after calculating indicators and dropping NaNs ({len(df_processed)} points, need {self.rl_lookback_window}). Cannot calculate normalization stats.")
                        continue

                    # Calculate mean and std deviation
                    means = df_processed.mean()
                    stds = df_processed.std()

                    # Avoid division by zero if std is 0 for any feature
                    stds[stds == 0] = 1e-8 # Replace 0 std with a very small number

                    self.norm_stats[symbol] = {'means': means, 'stds': stds}
                    logging.info(f"Successfully calculated normalization stats for {symbol}.")

                except Exception as e:
                    logging.exception(f"Error calculating normalization stats for {symbol}:")
                    # Ensure partial results aren't stored if an error occurs
                    if symbol in self.norm_stats:
                        del self.norm_stats[symbol]

            except Exception as e:
                logging.exception(f"Error fetching historical data for {symbol}:")
                # Ensure partial results aren't stored if an error occurs
                if symbol in self.norm_stats:
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

    def _get_normalized_features(self, symbol, df):
        """Adds indicators, normalizes a slice, and ensures correct shape."""
        if symbol not in self.norm_stats:
            logging.error(f"Normalization stats not found for {symbol}. Cannot normalize.")
            return None

        stats = self.norm_stats[symbol]
        try:
            # Check: Ensure input df has enough data for lookback + indicators
            min_required_data = self.rl_lookback_window + 35 # ~65 rows
            if df is None or len(df) < min_required_data:
                logging.warning(f"Input DataFrame for {symbol} has only {len(df)} rows, need {min_required_data} for reliable indicator calculation + lookback. Cannot normalize.")
                return None

            # --- Revised Indicator Calculation Logic ---
            # 1. Take a larger slice for indicator calculation (includes warm-up)
            df_indicator_input = df.iloc[-min_required_data:].copy()
            logging.debug(f"Shape of df_indicator_input for {symbol}: {df_indicator_input.shape}")

            # 2. Ensure necessary columns exist in this larger slice
            if not all(col in df_indicator_input.columns for col in OHLCV_COLS):
                logging.error(f"Indicator input slice for {symbol} is missing required OHLCV columns.")
                return None

            # 3. Add indicators to the larger slice
            logging.debug(f"Calculating indicators on {len(df_indicator_input)} rows for {symbol}...")
            df_indicator_input.ta.rsi(length=14, append=True)
            df_indicator_input.ta.macd(fast=12, slow=26, signal=9, append=True)
            df_indicator_input.ta.bbands(length=20, std=2, append=True)
            logging.debug(f"Indicators calculated. Shape after indicators: {df_indicator_input.shape}")
            logging.debug(f"Tail of df_indicator_input after indicators:\n{df_indicator_input.tail()}")

            # 4. Take the final lookback window slice *after* indicators are calculated
            df_final_slice = df_indicator_input.iloc[-self.rl_lookback_window:].copy()
            logging.debug(f"Shape of final slice for normalization: {df_final_slice.shape}")
            # --- End Revised Logic ---

            # 5. Select features and handle potential NaNs in the final slice
            #    (NaNs might be present at the start due to indicator warm-up)
            logging.debug(f"Selecting FEATURE_COLS: {FEATURE_COLS}") # Log the columns we intend to select
            if not all(col in df_final_slice.columns for col in FEATURE_COLS):
                 missing_cols = [col for col in FEATURE_COLS if col not in df_final_slice.columns]
                 logging.error(f"Final slice for {symbol} is missing required FEATURE_COLS: {missing_cols}. Available: {df_final_slice.columns.tolist()}")
                 return None
            df_final_slice = df_final_slice[FEATURE_COLS].copy()
            logging.debug(f"Shape of df_final_slice after selecting features: {df_final_slice.shape}") # Log shape after selection
            logging.debug(f"Columns of df_final_slice after selecting features: {df_final_slice.columns.tolist()}") # Log columns after selection

            original_rows = len(df_final_slice)
            df_final_slice.dropna(inplace=True)
            rows_after_dropna = len(df_final_slice)
            if rows_after_dropna < original_rows:
                 logging.debug(f"Dropped {original_rows - rows_after_dropna} rows with NaNs from final slice for {symbol}.")

            # 6. Check final shape AFTER dropna
            expected_rows = self.rl_lookback_window
            if df_final_slice.shape[0] != expected_rows:
                 logging.warning(f"Final data slice shape for {symbol} after indicators and dropna is {df_final_slice.shape}, expected ({expected_rows}, {len(FEATURE_COLS)}). Cannot normalize.")
                 return None

            # 7. Normalize using stored stats
            df_normalized = (df_final_slice - stats['means']) / stats['stds']
            logging.debug(f"Shape of normalized data BEFORE RETURN for {symbol}: {df_normalized.shape}") # Log shape before return

            # 8. Flatten the DataFrame to a 1D NumPy array for the RL agent
            flattened_features = df_normalized.values.flatten()
            logging.debug(f"Shape of FLATTENED normalized data for {symbol}: {flattened_features.shape}") # Log shape before return

            # --- Append Portfolio State Features --- 
            # Fetch current holdings and balance needed for the observation space (expected shape 242)
            # NOTE: This assumes the model was trained with these 2 extra features appended.
            # NOTE: These values are NOT normalized here, which might differ from training!
            try:
                symbol_base = symbol.split('-')[0] # e.g., 'BTC' from 'BTC-USD'
                current_holding_qty_decimal = self.holdings.get(symbol_base, Decimal('0.0'))
                current_holding_qty = float(current_holding_qty_decimal)
                current_balance_usd = float(self.holdings.get('USD', Decimal('0.0')))

            except Exception as e:
                logging.exception(f"Error accessing self.holdings within _get_normalized_features for {symbol}: {e}. Setting state features to 0.")
                current_holding_qty = 0.0
                current_balance_usd = 0.0

            # Append the two state features
            state_features = np.array([current_balance_usd, current_holding_qty])
            final_observation = np.concatenate((flattened_features, state_features))
            logging.debug(f"Shape after appending state features for {symbol}: {final_observation.shape}")

            # --- Final Shape Check ---
            # Now expecting 240 (market) + 2 (state) = 242
            expected_final_size = (self.rl_lookback_window * len(FEATURE_COLS)) + 2 
            if final_observation.shape[0] != expected_final_size:
                logging.error(f"Final observation shape {final_observation.shape} does not match expected {expected_final_size}. Returning None.")
                return None

            return final_observation

        except Exception as e:
             # Log the specific slice that caused the error if possible
             logging.exception(f"Error adding indicators/normalizing data slice for {symbol}:")
             return None

    def _get_signal(self, symbol: str, analysis_data: pd.DataFrame) -> str:
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
                if obs.shape[0] != EXPECTED_OBS_SHAPE:
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
