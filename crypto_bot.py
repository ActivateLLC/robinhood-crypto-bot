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

# Load environment variables
load_dotenv()

# Local imports
from config import * # Load all config variables
from alt_crypto_data import AltCryptoDataProvider # Import only the class
from robinhood_api_client import CryptoAPITrading # Import the API helper class

# Define constants used for RL feature processing directly here
OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume'] # Standard OHLCV columns
# Features used for RL model input (MUST match training environment)
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume', # OHLCV
    'RSI_14', # RSI
    'MACD_12_26_9', # MACD Line
    #'MACDh_12_26_9', # MACD Histogram (Removed)
    #'MACDs_12_26_9', # MACD Signal Line (Removed)
    #'BBL_20_2.0',    # Bollinger Lower Band (Removed)
    'BBM_20_2.0',    # Bollinger Middle Band
    #'BBU_20_2.0'     # Bollinger Upper Band (Removed)
]

# Setup basic logging first in case _setup_logging fails or isn't defined yet
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RobinhoodCryptoBot:
    """
    Automated cryptocurrency trading bot for Robinhood using RL or traditional strategies.
    """
    def __init__(self):
        # Basic Setup and Config Loading
        self.log_level = LOG_LEVEL
        self.log_file = LOG_FILE
        # self._setup_logging() # Let's keep basicConfig for now, setup can refine later if needed
        # Use basicConfig for initial setup, _setup_logging can refine it later if needed
        # logging.basicConfig(level=self.log_level, format='%(asctime)s - %(levelname)s - %(message)s', filename=self.log_file, filemode='w') # Use filemode='w' to overwrite log on each run
        logging.info("--- Initializing Crypto Bot --- ") # This should now work

        logging.info("Loading configuration...")
        self.symbols = SYMBOLS
        self.enable_trading = ENABLE_TRADING
        self.trade_amount_usd = TRADE_AMOUNT_USD
        self.lookback_days = LOOKBACK_DAYS
        self.check_interval = INTERVAL_MINUTES
        self.trading_strategy = TRADING_STRATEGY
        self.rl_model_path = RL_MODEL_PATH
        self.interval = self.check_interval # Assign self.interval = self.check_interval
        self.plot_enabled = PLOT_ENABLED # Added from view
        self.plot_output_dir = PLOT_OUTPUT_DIR # Added from view

        # Initialize placeholders
        self.api_client = None
        self.rl_agent = None
        self.data_provider = None
        self.symbol_map = {}
        self.historical_data = {}
        self.holdings = {'USD': {'quantity': Decimal('0')}}
        self.rl_lookback_window = RL_LOOKBACK_WINDOW # Added from view
        self.rl_features_mean = None # Added from view
        self.rl_features_std = None # Added from view
        self.norm_stats = {} # Initialize dict to store normalization stats per symbol

        logging.info(f"Configuration loaded: Symbols={self.symbols}, Strategy={self.trading_strategy}, Trading Enabled={self.enable_trading}")

        # --- Complex Initializations ---

        # 1. Conditionally load RL agent
        if self.trading_strategy == 'RL':
            logging.info(f"RL strategy selected. Attempting to load model from: {self.rl_model_path}")
            try:
                from stable_baselines3 import PPO # Or your specific agent
                if os.path.exists(self.rl_model_path):
                    self.rl_agent = PPO.load(self.rl_model_path)
                    logging.info("RL agent loaded successfully.")
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
                    logging.error(f"RL model file not found at {self.rl_model_path}. RL trading disabled.")
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

        # 2. Initialize API Client
        try:
            logging.info("Initializing API client...")
            if ROBINHOOD_API_KEY and ROBINHOOD_PRIVATE_KEY:
                # Pass the required keys to the constructor <- No, keys are read inside the class now
                # self.api_client = CryptoAPITrading(api_key=ROBINHOOD_API_KEY, base64_private_key=ROBINHOOD_PRIVATE_KEY)
                self.api_client = CryptoAPITrading() # Call without arguments
                # Optional: Verify connection
                # account_info = self.api_client.get_account()
                # if account_info and 'id' in account_info:
                #     logging.info(f"Successfully connected to Robinhood API. Account ID: {account_info['id']}")
                # else:
                #     logging.error(f"Failed to verify Robinhood API connection. Disabling trading. Response: {account_info}")
                #     self.enable_trading = False
                #     self.api_client = None
            else:
                logging.warning("API Key or Private Key missing. Cannot initialize API client. Trading disabled.")
                self.enable_trading = False
                self.api_client = None # Ensure it's None if keys are missing
        except Exception as e:
            logging.exception(f"Error initializing API client: {e}. Trading disabled.")
            self.enable_trading = False
            self.api_client = None

        # 3. Initialize Data Provider
        try:
            logging.info(f"Initializing data provider (Preference: {DATA_PROVIDER_PREFERENCE})...")
            self.data_provider = AltCryptoDataProvider() # Still need this if RL uses it?
            logging.info("Data provider initialized.")

            # 4. Fetch initial historical data and calculate normalization stats if using RL
            if self.trading_strategy == 'RL':
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
        if self.api_client:
            logging.info("Fetching initial holdings...")
            # self.fetch_holdings() # Make sure this method exists <- Incorrect name
            self.get_holdings() # <-- Correct method name (matches API client method)
        else:
            logging.warning("Skipping initial holdings fetch: API client not available or trading disabled.")
            self.holdings = {} # Initialize as empty if not fetched
        logging.info(f"RobinhoodCryptoBot __init__ finished. Trading Enabled: {self.enable_trading}")

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
            return df_normalized

        except Exception as e:
             # Log the specific slice that caused the error if possible
             logging.exception(f"Error adding indicators/normalizing data slice for {symbol}:")
             return None

    def _get_signal(self, symbol: str) -> str:
        """
        Determine trading signal for a given cryptocurrency
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC-USD')
        
        Returns:
            str: Trading signal (buy/sell/hold)
        """
        # Convert symbol for analysis if needed
        analysis_symbol = self.api_client.convert_symbol_for_analysis(symbol)
        
        # Perform technical analysis
        signal = self._analyze_market(analysis_symbol)
        
        return signal

    def _execute_trade(self, symbol: str, signal: str, amount: float):
        """
        Execute trade based on signal
        
        Args:
            symbol (str): Trading symbol
            signal (str): Trading signal
            amount (float): Trade amount in USD
        """
        # Convert symbol for Robinhood trading execution
        trading_symbol = self.api_client.convert_symbol_for_trading(symbol)
        
        if signal == 'buy':
            self.api_client.place_order(
                side='buy', 
                symbol=trading_symbol, 
                amount_in_dollars=amount
            )
        elif signal == 'sell':
            self.api_client.place_order(
                side='sell', 
                symbol=trading_symbol, 
                amount_in_dollars=amount
            )

    def get_holdings(self):
        """Fetches current crypto holdings using the API client and updates self.holdings."""
        if not self.api_client:
            logging.warning("Attempted to get holdings, but API client is not initialized.")
            self.holdings = {}
            return

        logging.info("Fetching current holdings via API client...")
        try:
            # Pass asset codes if needed, otherwise get all
            # holdings_data = self.api_client.get_holdings(*[s.split('-')[0] for s in self.symbols])
            holdings_data = self.api_client.get_holdings() # Get all holdings by default

            if holdings_data and 'results' in holdings_data:
                # Process the holdings data - structure depends on API response
                # Example: Assuming 'results' is a list of dicts with 'asset_code' and 'quantity'
                self.holdings = {
                    holding['asset_code']: float(holding['quantity'])
                    for holding in holdings_data['results']
                    if 'asset_code' in holding and 'quantity' in holding
                }
                logging.info(f"Successfully retrieved {len(self.holdings)} holdings: {list(self.holdings.keys())}")
            elif holdings_data:
                 # Handle cases where structure might be different or empty
                 logging.warning(f"Received holdings data, but 'results' key missing or empty: {holdings_data}")
                 self.holdings = {}
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
        if self.api_client:
            self.get_holdings() # Update holdings each cycle

        # 3. Analyze each symbol and get signals
        signals = {}
        for symbol in self.symbols:
            # Fetch or retrieve the necessary data for analysis
            # This might involve calling self.data_provider.get_historical_data again
            # or using data already stored in self.historical_data
            # For RL, we need at least lookback_window length of data ending now.
            # Placeholder: Assuming data is magically available for now
            # In a real scenario, you'd fetch recent data here.
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
                        # 1. Get latest price
                        latest_price = analysis_data['close'].iloc[-1]
                        if latest_price <= 0:
                            logging.error(f"Invalid latest price ({latest_price}) for {symbol}. Cannot execute trade.")
                            continue

                        # 2. Determine side and quantity
                        symbol_base = symbol.split('-')[0]
                        trade_amount_usd = float(os.getenv('TRADE_AMOUNT_USD', '10')) # Get trade amount from config
                        current_holding_qty = self.holdings.get(symbol_base, 0.0)

                        side = None
                        quantity = 0.0

                        if signal == 'buy_all':
                            side = 'buy'
                            quantity = trade_amount_usd / latest_price
                        elif signal == 'buy_half':
                            side = 'buy'
                            quantity = (trade_amount_usd / 2) / latest_price
                        elif signal == 'sell_all':
                            side = 'sell'
                            quantity = current_holding_qty
                        elif signal == 'sell_half':
                            side = 'sell'
                            quantity = current_holding_qty / 2

                        # 3. Validate trade
                        if side is None or quantity <= 0:
                            logging.warning(f"Invalid trade parameters for {symbol}: side={side}, quantity={quantity}. Skipping trade.")
                            continue
                        if side == 'sell' and quantity > current_holding_qty:
                            logging.warning(f"Attempting to sell {quantity} {symbol_base}, but only hold {current_holding_qty}. Adjusting to sell max available.")
                            quantity = current_holding_qty
                            if quantity <= 0: # Double check after adjustment
                                logging.warning(f"No holdings ({current_holding_qty}) of {symbol_base} to sell. Skipping trade.")
                                continue

                        # 4. Prepare and place order (Market Order for simplicity)
                        client_order_id = str(uuid.uuid4())
                        order_type = 'market'
                        # Market order config specifies the quantity for buy/sell
                        order_config = {
                            "quantity": f"{quantity:.8f}" # Format quantity as string with precision
                        }

                        logging.info(f"Placing {order_type} order: {side} {quantity:.8f} {symbol} (ID: {client_order_id})")
                        order_result = self.api_client.place_order(
                            client_order_id=client_order_id,
                            side=side,
                            order_type=order_type,
                            symbol=symbol,
                            order_config=order_config
                        )

                        if order_result:
                            logging.info(f"Order placement API call successful for {symbol}. Result: {order_result}")
                            # TODO: Optionally, fetch holdings again to confirm trade execution status
                        else:
                            logging.error(f"Order placement API call failed for {symbol}.")

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
