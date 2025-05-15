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
from datetime import datetime
from crew_agents.src.crypto_agents import CryptoTradingAgents # Added for CrewAI

# Load environment variables
load_dotenv()

# Local imports
from config import load_config, setup_logging, Config, DEFAULT_RL_ACTIONS_MAP
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

# Define the expected observation space shape for the RL model
# This might include flattened historical data + current balance + current holding
# Example: (lookback_window * num_features) + 2
# Adjust this based on your actual feature engineering
EXPECTED_OBS_SHAPE = (31,) # Based on model error: 60 lookback * 4 features + 2 state vars

# Custom JSON encoder for Decimal types
def decimal_default_json_serializer(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

# Helper function for structured JSON logging
def _log_structured_event(level: int, event_type: str, message: str, details: Optional[Dict[str, Any]] = None, severity: Optional[str] = None, exc_info=None):
    """Logs an event in a structured JSON format."""
    # Determine severity based on logging level if not provided
    if severity is None:
        if level >= logging.ERROR:
            severity = "ERROR"
        elif level >= logging.WARNING:
            severity = "WARNING"
        elif level >= logging.INFO:
            severity = "INFO"
        else:
            severity = "DEBUG"

    log_entry = {
        "timestamp": pd.Timestamp.now(tz='UTC').isoformat(),
        "source_component": "crypto_bot",
        "event_type": event_type,
        "message": message,
        "severity": severity
    }
    if details:
        log_entry["details"] = details
    
    logger_name = Config.LOGGER_NAME if hasattr(Config, 'LOGGER_NAME') else 'crypto_bot_logger'
    logger = logging.getLogger(logger_name)
    
    # Use the custom serializer for Decimal objects
    logger.log(level, json.dumps(log_entry, default=decimal_default_json_serializer), exc_info=exc_info)

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
        self.trading_strategy = self.config.TRADING_STRATEGY.upper() # Ensure uppercase
        self.rl_model_path = self.config.RL_MODEL_PATH
        self.rl_lookback_window = self.config.RL_LOOKBACK_WINDOW
        self.EXPECTED_OBS_SHAPE = (31,) # Expected shape for RL agent observations
        self.interval = self.check_interval # Assign self.interval = self.check_interval
        self.plot_enabled = self.config.PLOT_ENABLED # Added from view
        self.plot_output_dir = self.config.PLOT_OUTPUT_DIR # Added from view

        # Initialize placeholders
        self.broker: Optional[BaseBroker] = None
        self.rl_agent = None
        self.data_provider = None
        self.symbol_map = {}
        self.historical_data = {}
        self.portfolio_state: Dict[str, Dict[str, Decimal]] = { 
            # symbol_base: {'quantity': Decimal, 'average_cost': Decimal, 'initial_investment': Decimal}
        }
        self.current_balance: Decimal = Decimal('0.0') # Initialize current_balance
        self.data_cache = {}
        self.crew_ai_decision_maker = None # Added for CrewAI

        # Action mapping for RL agent
        logging.info(f"DEFAULT_RL_ACTIONS_MAP from config: {DEFAULT_RL_ACTIONS_MAP}")
        self.action_map = DEFAULT_RL_ACTIONS_MAP # Default fallback
        logging.info(f"Initial self.action_map set to DEFAULT: {self.action_map}")

        if self.trading_strategy == 'rl' and self.rl_agent:
            try:
                # Attempt to get action_map from the environment
                env = self.rl_agent.get_env() # Get the vectorized environment
                if env is not None:
                    # Try to access action_map directly from the first unwrapped environment
                    # This handles cases where the environment might be wrapped (e.g., by DummyVecEnv)
                    unwrapped_env = env.envs[0] if hasattr(env, 'envs') else env
                    logging.debug(f"Unwrapped env type: {type(unwrapped_env)}")

                    if hasattr(unwrapped_env, 'action_map') and unwrapped_env.action_map:
                        self.action_map = unwrapped_env.action_map
                        logging.info(f"Using action_map from unwrapped_env.action_map: {self.action_map}")
                    elif hasattr(unwrapped_env, 'action_map_env') and unwrapped_env.action_map_env: # Check for action_map_env as a fallback
                        self.action_map = unwrapped_env.action_map_env
                        logging.info(f"Using action_map from unwrapped_env.action_map_env: {self.action_map}")
                    else:
                        logging.warning("RL agent's environment does not have 'action_map' or 'action_map_env'. Using default.")
                else:
                    logging.warning("Could not get environment from RL agent. Using default action_map.")
            except AttributeError as e:
                logging.warning(f"Error accessing action_map from RL agent's environment: {e}. Using default.")
            except Exception as e:
                logging.error(f"Unexpected error when trying to get action_map: {e}. Using default action_map.")
        
        logging.info(f"Final self.action_map in __init__: {self.action_map}")

        # Initialize experience logger (if not already done or needs specific config)
        # Assuming setup_logging handles the 'experience_logger' based on config
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
                    error_message = f"RL model file not found at {self.rl_model_path}. RL agent not loaded."
                    logging.error(error_message)
                    _log_structured_event(
                        level=logging.CRITICAL,
                        event_type="MODEL_LOAD_ERROR",
                        message=error_message,
                        details={"model_path": self.rl_model_path, "reason": "File not found"},
                        severity="CRITICAL"
                    )
            except ImportError as e:
                error_message = "Failed to import PPO from stable_baselines3. RL agent not loaded."
                logging.exception(error_message) # Keep exception for stack trace
                _log_structured_event(
                    level=logging.CRITICAL,
                    event_type="MODEL_LOAD_ERROR",
                    message=error_message,
                    details={"error": str(e), "reason": "ImportError"},
                    severity="CRITICAL"
                )
            except Exception as e:
                error_message = f"Error loading RL model from {self.rl_model_path}."
                logging.exception(error_message) # Keep exception for stack trace
                _log_structured_event(
                    level=logging.CRITICAL,
                    event_type="MODEL_LOAD_ERROR",
                    message=error_message,
                    details={
                        "model_path": self.rl_model_path,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    },
                    severity="CRITICAL"
                )
    
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
                error_message = "Broker initialization failed. Trading will be disabled."
                logging.error(error_message) # Keep a simple log too
                _log_structured_event(
                    level=logging.CRITICAL,
                    event_type="BROKER_INIT_FAILURE_CRITICAL", # More specific event type
                    message=error_message,
                    details={"reason": "Broker object is None"},
                    severity="CRITICAL"
                )
                self.enable_trading = False # Explicitly disable trading
        else:
            logging.info("Trading is disabled in the configuration. Broker will not be initialized.")

        # If broker initialization was successful, try to connect and fetch initial balance
        if self.broker and self.enable_trading: # Check enable_trading again in case it was set to False by unsupported type
            logging.info(f"Broker {broker_type} initialized. Attempting to connect and fetch account info...")
            if self.broker.connect():
                logging.info(f"Broker {broker_type} connected successfully.")
                account_info_list = self.broker.get_account_info()
                
                usd_balance_found = False
                if account_info_list and isinstance(account_info_list, list):
                    for account in account_info_list:
                        if isinstance(account, dict) and account.get('currency_code') == 'USD':
                            balance_str = account.get('balance')
                            if balance_str is not None:
                                try:
                                    self.current_balance = float(balance_str)
                                    logging.info(f"Initial account balance fetched: ${self.current_balance:.2f} USD")
                                    _log_structured_event(
                                        level=logging.INFO,
                                        event_type="ACCOUNT_BALANCE_FETCHED",
                                        message="Successfully fetched initial account balance.",
                                        details={"balance_usd": float(self.current_balance), "broker": broker_type}
                                    )
                                    usd_balance_found = True
                                    break # Found USD balance, exit loop
                                except ValueError:
                                    logging.error(f"Could not convert USD balance '{balance_str}' to float. Account: {account}")
                            else:
                                logging.warning(f"USD account found, but 'balance' key is missing or None. Account: {account}")
                        elif isinstance(account, dict) and 'error' in account:
                            # Log if an individual item in the list is an error dictionary
                            logging.error(f"Error item in account_info_list: {account.get('details', account.get('error'))}")
                elif account_info_list and isinstance(account_info_list, dict) and 'error' in account_info_list:
                    # This case handles if get_account_info itself returned a single error dict (not a list of errors)
                    # This might be redundant if RobinhoodBroker always returns a list, even for single errors, but good for robustness.
                    logging.error(f"Failed to fetch account balance: API error from broker.get_account_info(): {account_info_list.get('details', account_info_list.get('error'))}")
                
                if not usd_balance_found:
                    logging.error(f"Failed to fetch or parse USD account balance after successful connection to {broker_type}. Trading will be disabled.")
                    _log_structured_event(
                        level=logging.ERROR,
                        event_type="ACCOUNT_BALANCE_FETCH_FAILED",
                        message="Failed to fetch or parse USD account balance post-connection.",
                        details={"broker": broker_type, "account_info_raw": account_info_list if isinstance(account_info_list, (list, dict)) else str(account_info_list)},
                        severity="ERROR"
                    )
                    self.enable_trading = False
            else:
                logging.error(f"Failed to connect to broker {broker_type}. Trading will be disabled.")
                _log_structured_event(
                    level=logging.CRITICAL, # This is critical as trading cannot proceed
                    event_type="BROKER_CONNECT_FAILED",
                    message=f"Failed to connect to broker: {broker_type}.",
                    details={"broker_type": broker_type},
                    severity="CRITICAL"
                )
                self.enable_trading = False # Disable trading if connection fails
        elif self.enable_trading: # This means self.broker is None but enable_trading was true initially
            # This case is mostly covered by the self.broker is None check below, 
            # but adding explicit log if somehow enable_trading is true but broker is None here.
            logging.error(f"Broker object is None despite trading being enabled for {broker_type}. Trading disabled.")
            self.enable_trading = False

        # Final check on broker initialization and connection status
        if self.broker is None and self.enable_trading: # Catches cases where _initialize_broker returned None
            error_message = "Broker object is None despite trading being enabled. Trading will be disabled."
            logging.error(error_message) # Keep a simple log too
            _log_structured_event(
                level=logging.CRITICAL,
                event_type="BROKER_INIT_FAILURE_CRITICAL", # More specific event type
                message=error_message,
                details={"reason": "Broker object is None"},
                severity="CRITICAL"
            )
            self.enable_trading = False # Explicitly disable trading

        # 3. Initialize Data Provider
        try:
            logging.info(f"Initializing data provider (Preference: {self.config.DATA_PROVIDER_PREFERENCE})...")
            if self.config.DATA_PROVIDER_PREFERENCE == 'yfinance':
                try:
                    from data_providers.yfinance_data_provider import YFinanceDataProvider
                    self.data_provider = YFinanceDataProvider(symbols_to_trade=self.config.SYMBOLS_TO_TRADE)
                    logging.info("Dedicated YFinanceDataProvider initialized successfully.")
                except ModuleNotFoundError:
                    _log_structured_event(
                        logging.WARNING,
                        "YFINANCE_PROVIDER_MISSING_FALLBACK_WARNING",
                        f"DATA_PROVIDER_PREFERENCE is 'yfinance', but YFinanceDataProvider module is missing. Falling back to AltCryptoDataProvider.",
                        details={"preference": "yfinance", "fallback_provider": "AltCryptoDataProvider"}
                    )
                    from alt_crypto_data import AltCryptoDataProvider
                    self.data_provider = AltCryptoDataProvider()
                    logging.info("AltCryptoDataProvider initialized as fallback for 'yfinance' preference.")
            elif self.config.DATA_PROVIDER_PREFERENCE == 'alt_crypto_data':
                from alt_crypto_data import AltCryptoDataProvider
                self.data_provider = AltCryptoDataProvider()
                logging.info("AltCryptoDataProvider initialized successfully based on preference.")
            else:
                _log_structured_event(
                    logging.WARNING,
                    "INVALID_DATA_PROVIDER_PREFERENCE_DEFAULT_FALLBACK_WARNING",
                    f"Invalid data provider preference: {self.config.DATA_PROVIDER_PREFERENCE}. Defaulting to AltCryptoDataProvider.",
                    details={"invalid_preference": self.config.DATA_PROVIDER_PREFERENCE, "default_provider": "AltCryptoDataProvider"}
                )
                from alt_crypto_data import AltCryptoDataProvider
                self.data_provider = AltCryptoDataProvider()
                logging.info("AltCryptoDataProvider initialized as default fallback due to invalid preference.")
            # logging.info("Data Provider initialized successfully.") # This message is now more specific within branches
        except Exception as e:
            error_message = "Failed to initialize AltCryptoDataProvider."
            logging.exception(error_message)
            _log_structured_event(
                level=logging.CRITICAL,
                event_type="DATA_PROVIDER_INIT_ERROR",
                message=error_message,
                details={"error_type": type(e).__name__, "error_message": str(e)},
                severity="CRITICAL"
            )
            self.data_provider = None # Ensure it's None
            # If data provider fails, bot cannot function. Consider disabling trading or halting.
            logging.critical("CRITICAL: Data provider failed to initialize. Bot cannot fetch market data. Trading disabled.")
            self.enable_trading = False # Disable trading if data provider fails

        # 4. Fetch initial historical data and calculate normalization stats if RL strategy is used
        # Only proceed if data_provider was initialized successfully
        if self.data_provider and self.trading_strategy == 'rl':
            try:
                self._calculate_and_store_norm_stats() # This method now handles its own internal logging
                if not self.norm_stats: # Check if any stats were actually calculated
                    # This is a more general check after the method call, if the method itself didn't make it critical
                    # _calculate_and_store_norm_stats will log if it fails for ALL symbols
                    pass # The method itself logs if it fails for all symbols
            except Exception as e: # Catch unexpected errors from the stats calculation call itself
                error_message = "Critical error during initial calculation of normalization statistics."
                logging.exception(error_message)
                _log_structured_event(
                    level=logging.CRITICAL,
                    event_type="NORMALIZATION_STATS_ERROR",
                    message=error_message,
                    details={"error_type": type(e).__name__, "error_message": str(e)},
                    severity="CRITICAL"
                )
                # Depending on bot logic, might disable trading or RL strategy
                logging.critical("Normalization stats calculation failed critically. RL strategy may be impaired.")
                # self.trading_strategy = 'hold' # Example fallback

        # 5. Fetch initial holdings (Requires API Client)
        if self.broker:
            logging.info("Fetching initial holdings...")
            self.get_holdings() # This will now populate self.portfolio_state
        else:
            logging.warning("Skipping initial holdings fetch: API client not available or trading disabled.")
            # self.portfolio_state remains as is or empty if never populated
        logging.info(f"RobinhoodCryptoBot __init__ finished. Trading Enabled: {self.enable_trading}")

        if self.trading_strategy == 'CREWAI':
            _log_structured_event(logging.INFO, "CREWAI_INIT_STARTED", "Initializing CryptoTradingAgents for CREWAI strategy.")
            try:
                agent_symbol = self.symbols[0] if self.symbols else 'BTC-USD' # Default or first symbol
                self.crew_ai_decision_maker = CryptoTradingAgents(symbol=agent_symbol)
                _log_structured_event(logging.INFO, "CREWAI_INIT_SUCCESS", f"CryptoTradingAgents initialized for symbol: {agent_symbol}")
            except ImportError as ie:
                _log_structured_event(
                    logging.CRITICAL,
                    "CREWAI_IMPORT_ERROR",
                    f"Failed to import CryptoTradingAgents: {ie}. Ensure crew_agents.src is in PYTHONPATH or accessible.", exc_info=True
                )
                self.enable_trading = False # Disable trading if essential component fails
            except Exception as e:
                _log_structured_event(
                    logging.ERROR,
                    "CREWAI_INIT_FAILED",
                    f"Failed to initialize CryptoTradingAgents: {e}", exc_info=True
                )
                self.enable_trading = False # Disable trading

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
            error_message = f"Failed to initialize {broker_type}."
            logging.exception(error_message) # Keep existing exception log for stack trace
            _log_structured_event(
                level=logging.CRITICAL,
                event_type="BROKER_INIT_ERROR",
                message=error_message,
                details={
                    "broker_type": broker_type,
                    "config_keys": list(broker_config.keys()), # Log keys, not values, for security
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                severity="CRITICAL"
            )
            return None

    def _calculate_and_store_norm_stats(self):
        """
        Calculates and stores normalization statistics (mean, std) for the 27 technical indicators.
        """
        logging.info("Calculating initial normalization statistics for 27 RL technical indicators...")
        self.norm_stats = {} # Reset stats

        min_rows_for_ohlcv_fetch = 90 
        logging.info(f"Starting normalization stats calculation for symbols: {self.symbols}")
        
        for symbol in self.symbols:
            logging.debug(f"Processing normalization stats for symbol: {symbol}")
            try:
                df_ohlcv_history = self.data_provider.fetch_price_history(symbol, days=min_rows_for_ohlcv_fetch)

                if df_ohlcv_history is None or df_ohlcv_history.empty:
                    error_message = f"No historical OHLCV data found for {symbol} during norm_stats calculation."
                    logging.warning(error_message)
                    _log_structured_event(
                        level=logging.WARNING,
                        event_type="DATA_FETCH_ERROR",
                        message=error_message,
                        details={"symbol": symbol, "context": "normalization_stats"},
                        severity="WARNING"
                    )
                    continue
                
                if not all(col.lower() in [c.lower() for c in df_ohlcv_history.columns] for col in OHLCV_COLS):
                    error_message = f"Fetched historical data for {symbol} is missing one or more OHLCV columns. Skipping normalization."
                    logging.warning(error_message)
                    _log_structured_event(
                        level=logging.WARNING,
                        event_type="DATA_PROCESSING_ERROR",
                        message=error_message,
                        details={"symbol": symbol, "missing_columns": True, "context": "normalization_stats"},
                        severity="WARNING"
                    )
                    continue
                
                if len(df_ohlcv_history) < 60:
                    error_message = f"Insufficient historical OHLCV data points for {symbol} ({len(df_ohlcv_history)} points, need ~60). Skipping normalization."
                    logging.warning(error_message)
                    _log_structured_event(
                        level=logging.WARNING,
                        event_type="DATA_INSUFFICIENT_ERROR",
                        message=error_message,
                        details={"symbol": symbol, "rows_found": len(df_ohlcv_history), "rows_needed": 60, "context": "normalization_stats"},
                        severity="WARNING"
                    )
                    continue

                df_with_indicators = self._add_all_technical_indicators(df_ohlcv_history.copy())
                df_technical_features = df_with_indicators[TECHNICAL_INDICATOR_COLUMNS].copy()
                df_technical_features.dropna(inplace=True)

                min_data_points_for_stats = max(30, self.rl_lookback_window) 
                if len(df_technical_features) < min_data_points_for_stats:
                    error_message = f"Insufficient data for {symbol} after indicator calculation & NaN drop ({len(df_technical_features)} points, need {min_data_points_for_stats}). Cannot calc norm stats."
                    logging.warning(error_message)
                    _log_structured_event(
                        level=logging.WARNING,
                        event_type="DATA_INSUFFICIENT_ERROR",
                        message=error_message,
                        details={"symbol": symbol, "rows_found": len(df_technical_features), "rows_needed": min_data_points_for_stats, "context": "normalization_stats_post_indicators"},
                        severity="WARNING"
                    )
                    continue

                means = df_technical_features.mean()
                stds = df_technical_features.std()
                stds[stds == 0] = 1e-8

                self.norm_stats[symbol] = {'means': means, 'stds': stds}
                logging.info(f"Successfully calculated normalization stats for {symbol} using {len(df_technical_features)} data points.")

            except Exception as e:
                error_message = f"Error calculating/storing normalization stats for {symbol}."
                logging.exception(error_message) # Keep for stack trace
                _log_structured_event(
                    level=logging.ERROR,
                    event_type="NORMALIZATION_STATS_ERROR",
                    message=error_message,
                    details={
                        "symbol": symbol,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    },
                    severity="ERROR"
                )
                if symbol in self.norm_stats:
                    del self.norm_stats[symbol]

        logging.info("Finished calculating initial normalization statistics.")
        if not self.norm_stats:
            error_message = "CRITICAL: Failed to calculate normalization statistics for ANY symbol. RL strategy will likely fail or be significantly impaired."
            logging.error(error_message)
            _log_structured_event(
                level=logging.CRITICAL,
                event_type="NORMALIZATION_STATS_FAILURE_ALL_SYMBOLS",
                message=error_message,
                severity="CRITICAL"
            )
        else:
             symbols_with_stats = list(self.norm_stats.keys())
             logging.info(f"Normalization stats calculated for: {symbols_with_stats}")
             missing_stats = [s for s in self.symbols if s not in self.norm_stats]
             if missing_stats:
                 logging.warning(f"Could not calculate normalization stats for: {missing_stats}")
                 # Optionally, log a structured event for this too, though it's less critical than ALL failing
                 _log_structured_event(
                     level=logging.WARNING,
                     event_type="NORMALIZATION_STATS_MISSING_SOME_SYMBOLS",
                     message=f"Failed to calculate normalization stats for some symbols: {missing_stats}",
                     details={"missing_symbols": missing_stats, "successful_symbols": symbols_with_stats},
                     severity="WARNING"
                 )

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
                # Fill NaNs that might have been introduced by _add_all_technical_indicators before normalization
                # This is critical if _add_all_technical_indicators doesn't perfectly fill its own NaNs for the selected columns
                df_processed[col_name].fillna(method='bfill', inplace=True) # Backfill first
                df_processed[col_name].fillna(0.0, inplace=True) # Then fill remaining NaNs with 0
                
        return df_processed

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
        MIN_ROWS_FOR_FEATURES = self.config.RL_MIN_ROWS_FOR_FEATURES # Corrected access

        if self.trading_strategy == 'rl' and self.rl_agent is not None:
            logging.debug(f"Generating signal for {symbol} using RL agent.")
            obs = None
            try:
                # --- Start: Inlined _get_normalized_features logic ---
                if analysis_data is None or analysis_data.empty or len(analysis_data) < MIN_ROWS_FOR_FEATURES:
                    _log_structured_event(
                        logging.WARNING,
                        "DATA_INSUFFICIENT_ERROR",
                        f"Insufficient analysis_data for {symbol}. Got {len(analysis_data) if analysis_data is not None else 0} rows, need {MIN_ROWS_FOR_FEATURES}. Defaulting to 'hold'.",
                        details={"symbol": symbol, "data_rows": len(analysis_data) if analysis_data is not None else 0, "required_rows": MIN_ROWS_FOR_FEATURES}
                    )
                    return 'hold'

                df_with_indicators = self._add_all_technical_indicators(analysis_data)
                if df_with_indicators.empty or df_with_indicators[TECHNICAL_INDICATOR_COLUMNS].isnull().all().all():
                    _log_structured_event(
                        logging.ERROR,
                        "INDICATOR_CALCULATION_FAILED",
                        f"_add_all_technical_indicators returned empty or all-NaN DataFrame for {symbol}. Defaulting to 'hold'.",
                        details={"symbol": symbol}
                    )
                    return 'hold'

                if symbol not in self.norm_stats or not all(k in self.norm_stats[symbol] for k in ['means', 'stds']):
                    _log_structured_event(
                        logging.ERROR,
                        "NORMALIZATION_STATS_MISSING",
                        f"Normalization stats (mean/std) not found for {symbol}. Defaulting to 'hold'.",
                        details={"symbol": symbol, "available_stats_keys": list(self.norm_stats.get(symbol, {}).keys())}
                    )
                    return 'hold'

                # Select only the technical indicator columns for normalization
                features_df = df_with_indicators[TECHNICAL_INDICATOR_COLUMNS].copy()
                if features_df.empty:
                    _log_structured_event(
                        logging.ERROR,
                        "DATA_PROCESSING_ERROR",
                        f"No technical indicator columns found or data is empty after selection for {symbol}. Defaulting to 'hold'.",
                        details={"symbol": symbol}
                    )
                    return 'hold'
                
                # Fill any NaNs that might have been introduced by _add_all_technical_indicators before normalization
                # This is critical if _add_all_technical_indicators doesn't perfectly fill its own NaNs for the selected columns
                features_df.fillna(method='bfill', inplace=True)
                features_df.fillna(0.0, inplace=True) # Fill any remaining NaNs at the beginning with 0

                mean_stats = pd.Series(self.norm_stats[symbol]['means'])[TECHNICAL_INDICATOR_COLUMNS]
                std_stats = pd.Series(self.norm_stats[symbol]['stds'])[TECHNICAL_INDICATOR_COLUMNS]
                
                # Ensure alignment if series come from dicts
                mean_stats = mean_stats.reindex(features_df.columns)
                std_stats = std_stats.reindex(features_df.columns)

                normalized_features = (features_df - mean_stats) / (std_stats + 1e-7) # Epsilon for division by zero
                
                if normalized_features.isnull().any().any():
                    _log_structured_event(
                        logging.WARNING, # Warning as we might proceed with some NaNs if not caught earlier
                        "DATA_PROCESSING_ERROR",
                        f"NaNs found in normalized features for {symbol} despite fill efforts. Check std_devs. Defaulting to 'hold'.",
                        details={"symbol": symbol}
                    )
                    # Consider returning 'hold' if NaNs are critical
                    # For now, we'll let the model try, but this is a data quality issue.

                normalized_latest_indicators = normalized_features.iloc[-1].values.astype(np.float32)

                # --- Placeholder for additional features (balance, holdings, PNL, price change) ---
                base_asset = symbol.split('-')[0]
                current_balance_usd = self.current_balance # Assuming self.current_balance is in USD
                
                # Example: Normalize balance by dividing by a large constant (e.g., typical portfolio size)
                # This should ideally use stats from _calculate_and_store_norm_stats if these features were included there
                balance_norm_divisor = self.config.get('rl_balance_norm_divisor', 10000.0)
                current_balance_normalized = float(current_balance_usd / Decimal(str(balance_norm_divisor)))
                _log_structured_event(logging.DEBUG, "PLACEHOLDER_NORMALIZATION_USED", f"Using placeholder normalization for balance for {symbol}. Divisor: {balance_norm_divisor}", details={"symbol": symbol, "feature": "balance", "divisor": balance_norm_divisor})

                current_asset_state = self.portfolio_state.get(base_asset, {})
                current_holding_qty = current_asset_state.get('quantity', Decimal('0.0'))
                average_cost_basis = current_asset_state.get('average_cost', Decimal('0.0'))
                initial_investment = current_asset_state.get('initial_investment', Decimal('0.0'))

                holding_norm_divisor = self.config.get('rl_holding_norm_divisor', 100.0) # e.g. max expected holding qty for a less valuable asset
                current_holding_qty_normalized = float(current_holding_qty / Decimal(str(holding_norm_divisor)))
                _log_structured_event(logging.DEBUG, "PLACEHOLDER_NORMALIZATION_USED", f"Using placeholder normalization for holding quantity for {symbol}. Divisor: {holding_norm_divisor}", details={"symbol": symbol, "feature": "holding_qty", "divisor": holding_norm_divisor})

                # Calculate PNL and Price Change if position is held
                pnl_normalized = 0.0
                price_change_normalized = 0.0

                if not analysis_data['close'].empty:
                    current_price = Decimal(str(analysis_data['close'].iloc[-1]))
                    if current_holding_qty > Decimal('1e-8') and average_cost_basis > Decimal('1e-8'): # Position exists and has valid cost
                        unrealized_pnl = (current_price - average_cost_basis) * current_holding_qty
                        price_change_percentage = ((current_price - average_cost_basis) / average_cost_basis) # This is a ratio, e.g., 0.05 for 5%
                        
                        # Normalize PNL (e.g., as a ratio of initial investment for this position)
                        if initial_investment > Decimal('1e-8'):
                            pnl_normalized = float(unrealized_pnl / initial_investment)
                        else: # Should not happen if holding qty > 0 and avg_cost > 0, but defensive
                            pnl_normalized = 0.0 
                        
                        # Normalize Price Change (already a ratio, can be scaled or clipped if necessary)
                        # For example, clipping to +/- 100% (ratio +/- 1.0) then using as is.
                        price_change_normalized = float(max(-1.0, min(1.0, price_change_percentage))) # Clipped to +/- 100%
                        
                        _log_structured_event(logging.DEBUG, "PNL_PRICE_CHANGE_CALCULATED", 
                                            f"{symbol}: PNL_Norm={pnl_normalized:.4f}, PriceChange_Norm={price_change_normalized:.4f}",
                                            details={"symbol":symbol, "current_price":float(current_price), "avg_cost":float(average_cost_basis), 
                                                     "qty":float(current_holding_qty), "unrealized_pnl":float(unrealized_pnl),
                                                     "price_change_perc_raw": float(price_change_percentage), "initial_investment_for_pos": float(initial_investment)})
                    else:
                        _log_structured_event(logging.DEBUG, "PNL_PRICE_CHANGE_ZERO_NO_POSITION",
                                            f"{symbol}: PNL and PriceChange are 0 (no tracked position or zero cost basis).",
                                            details={"symbol":symbol, "current_price":float(current_price), "avg_cost":float(average_cost_basis), "qty":float(current_holding_qty)})
                else:
                    _log_structured_event(logging.WARNING, "PNL_PRICE_CHANGE_ZERO_NO_CURRENT_PRICE",
                                        f"{symbol}: PNL and PriceChange are 0 (no current price from analysis_data).",
                                        details={"symbol":symbol})

                additional_features = np.array([
                    current_balance_normalized,
                    current_holding_qty_normalized,
                    pnl_normalized,  # Replaces 0.0 placeholder
                    price_change_normalized  # Replaces 0.0 placeholder
                ])

                # --- End: Inlined _get_normalized_features logic ---

            except Exception as e_feature_gen:
                _log_structured_event(
                    logging.ERROR,
                    "FEATURE_GENERATION_ERROR",
                    f"Error during feature generation for RL agent for {symbol}: {e_feature_gen}",
                    details={"symbol": symbol, "error_message": str(e_feature_gen), "exc_info": True}
                )
                obs = None # Ensure obs is None if feature generation fails

            if obs is None:
                _log_structured_event(
                    logging.WARNING,
                    "FEATURE_GENERATION_FAILED",
                    f"Could not generate normalized features for {symbol}. Defaulting to 'hold'.",
                    details={"symbol": symbol}
                )
                return 'hold'

            if obs.shape[0] != EXPECTED_OBS_SHAPE[0]:
                _log_structured_event(
                    logging.ERROR,
                    "MODEL_INPUT_SHAPE_MISMATCH",
                    f"Observation shape mismatch for {symbol}. Expected {EXPECTED_OBS_SHAPE}, Got {obs.shape}. Defaulting to 'hold'.",
                    details={"symbol": symbol, "expected_shape": str(EXPECTED_OBS_SHAPE), "actual_shape": str(obs.shape)}
                )
                return 'hold'

            try:
                action_raw, _ = self.rl_agent.predict(obs, deterministic=True)
                action_int = action_raw.item() # Convert numpy array to Python int
                signal = self.action_map.get(action_int, 'hold')
                
                if signal == 'hold' and action_int not in self.action_map:
                    _log_structured_event(
                        logging.WARNING,
                        "INVALID_ACTION_MAPPING",
                        f"RL agent predicted action {action_int} for {symbol} which is not in action_map. Defaulting to 'hold'.",
                        details={"symbol": symbol, "predicted_action_int": action_int, "action_map": str(self.action_map)}
                    )
                
                logging.info(f"[{symbol}] RL agent signal: {signal} (action_int: {action_int})")

                self._log_experience(
                    symbol=symbol,
                    state=obs,
                    action=signal,
                    reward=0.0, # Placeholder
                    pnl_change=0.0, # Placeholder
                    portfolio_value=float(self.current_balance) # Log current portfolio value
                )
                return signal
            except Exception as e_predict:
                _log_structured_event(
                    logging.ERROR,
                    "MODEL_INFERENCE_ERROR",
                    f"Error during RL agent prediction for {symbol}: {e_predict}",
                    details={"symbol": symbol, "error_message": str(e_predict), "exc_info": True}
                )
                return 'hold'
                
        elif self.trading_strategy == 'rsi':
            logging.debug(f"Generating signal for {symbol} using RSI strategy (placeholder).")
            # Placeholder for RSI strategy logic
            # rsi_value = analysis_data['RSI_14'].iloc[-1]
            # if rsi_value < 30: return 'buy'
            # elif rsi_value > 70: return 'sell'
            return 'hold'
        elif self.trading_strategy == 'CREWAI':
            if self.crew_ai_decision_maker:
                _log_structured_event(logging.INFO, "CREWAI_GET_SIGNAL_STARTED", 
                                      f"Attempting to get signal from CrewAI for {symbol}", 
                                      details={"symbol": symbol})
                try:
                    # Ensure the agent's symbol matches the current symbol, or re-initialize if necessary.
                    # This basic re-initialization might be slow for frequent symbol changes.
                    # A more robust solution would involve a pool of agents or an agent that can handle multiple symbols.
                    if self.crew_ai_decision_maker.symbol != symbol:
                        _log_structured_event(logging.WARNING, "CREWAI_SYMBOL_MISMATCH",
                                              f"CrewAI decision maker initialized with {self.crew_ai_decision_maker.symbol} but getting signal for {symbol}. Re-initializing for current symbol.",
                                              details={"init_symbol": self.crew_ai_decision_maker.symbol, "current_symbol": symbol})
                        self.crew_ai_decision_maker = CryptoTradingAgents(symbol=symbol) # Re-init

                    # Pass the historical_data (which is analysis_data here, containing all indicators)
                    crew_analysis_output = self.crew_ai_decision_maker.analyze_market(historical_data=analysis_data) 
                    
                    signal = crew_analysis_output.get('recommended_action', 'hold').lower()
                    
                    _log_structured_event(logging.INFO, "CREWAI_SIGNAL_RECEIVED", 
                                          f"Signal received from CrewAI for {symbol}: {signal}", 
                                          details={"symbol": symbol, "signal": signal, "crew_output_keys": list(crew_analysis_output.keys()) if crew_analysis_output else []})
                except Exception as e:
                    _log_structured_event(
                        logging.ERROR,
                        "CREWAI_GET_SIGNAL_ERROR",
                        f"Error getting signal from CrewAI for {symbol}: {e}", 
                        details={"symbol": symbol}, exc_info=True
                    )
                    signal = 'hold' # Fallback on error
            else:
                _log_structured_event(logging.WARNING, "CREWAI_NOT_INITIALIZED",
                                      f"CrewAI strategy selected but decision maker not initialized/available for {symbol}. Defaulting to hold.",
                                      details={"symbol": symbol})
                signal = 'hold'

        else:
            _log_structured_event(logging.WARNING, "UNKNOWN_STRATEGY",
                                  f"Unknown trading strategy: {self.trading_strategy} for {symbol}. Defaulting to hold.")
            signal = 'hold'

        return signal
 
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
        trade_params = {
            "symbol": symbol,
            "signal": signal,
            "latest_price": float(latest_price) # For JSON serialization
        }

        if not self.enable_trading:
            _log_structured_event(
                logging.INFO,
                "TRADE_EXECUTION_DISABLED",
                "Trading is disabled. Skipping trade execution.",
                details=trade_params
            )
            return

        if not self.broker:
            _log_structured_event(
                logging.ERROR,
                "BROKER_UNAVAILABLE_ERROR",
                "Broker not initialized. Cannot execute trade.",
                details=trade_params
            )
            return

        side = None
        quantity = Decimal('0.0')
        trade_amount = Decimal(str(self.trade_amount_usd))
        symbol_base = symbol.split('-')[0]
        current_holding_qty = Decimal(self.portfolio_state.get(symbol_base, {}).get('quantity', Decimal('0.0'))) # Updated to use portfolio_state

        trade_params.update({
            "trade_amount_usd": float(trade_amount),
            "current_holding_qty": float(current_holding_qty)
        })

        try:
            if latest_price <= 0:
                _log_structured_event(
                    logging.ERROR,
                    "INVALID_PRICE_ERROR",
                    f"Invalid latest price ({latest_price}) for {symbol}. Skipping trade.",
                    details=trade_params
                )
                return

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
            # else: signal is 'hold' or unknown, side remains None

        except Exception as e:
            _log_structured_event(
                logging.ERROR,
                "QUANTITY_CALCULATION_ERROR",
                f"Error calculating quantity for {symbol} signal {signal}: {e}",
                details={"error_message": str(e), "exc_info": True},
                **trade_params
            )
            return

        trade_params["calculated_side"] = side
        trade_params["calculated_quantity"] = float(quantity)

        min_tradeable_qty_str = self.config.get('min_tradeable_quantity', '0.00000001')
        min_tradeable_qty = Decimal(min_tradeable_qty_str)

        if side is None or quantity <= min_tradeable_qty:
            _log_structured_event(
                logging.WARNING,
                "INVALID_TRADE_PARAMETERS_WARNING",
                f"Invalid trade parameters for {symbol}: side={side}, calculated quantity={quantity:.8f} (min: {min_tradeable_qty}). Signal was '{signal}'. Skipping trade.",
                details={"min_tradeable_qty": float(min_tradeable_qty)},
                **trade_params
            )
            return

        if side == 'sell' and quantity > current_holding_qty:
            original_sell_qty = quantity
            quantity = current_holding_qty
            _log_structured_event(
                logging.WARNING,
                "INSUFFICIENT_HOLDINGS_FOR_SELL_WARNING",
                f"Attempting to sell {original_sell_qty:.8f} {symbol_base}, but only hold {current_holding_qty:.8f}. Adjusting to sell max available ({quantity:.8f}).",
                details={"original_calculated_quantity": float(original_sell_qty), "adjusted_quantity": float(quantity)},
                **trade_params
            )
            trade_params["calculated_quantity"] = float(quantity) # Update for subsequent logs
            
            if quantity <= min_tradeable_qty:
                _log_structured_event(
                    logging.WARNING,
                    "NO_HOLDINGS_TO_SELL_WARNING",
                    f"No holdings ({current_holding_qty:.8f}) of {symbol_base} to sell after adjustment (min_tradeable: {min_tradeable_qty}). Skipping trade.",
                    details={"min_tradeable_qty": float(min_tradeable_qty)},
                    **trade_params
                )
                return

        order_details = None
        try:
            logging.info(f"Attempting to place {side} order for {quantity:.8f} {symbol} at market price.")
            # Ensure quantity is correctly typed for broker method if it's strict
            # Most SDKs handle Decimal, float, or string representations.
            order_details = self.broker.place_market_order(
                symbol=symbol,      
                side=side,        
                quantity=quantity   
            )

            if order_details and order_details.get('status') not in ['failed', 'rejected', 'cancelled', None, ''] and order_details.get('order_id'): # Basic check for success
                _log_structured_event(
                    logging.INFO,
                    "TRADE_SUBMITTED_SUCCESS",
                    f"Trade submitted successfully via broker for {symbol}. Side: {side}, Qty: {quantity:.8f}. Order ID: {order_details.get('order_id')}",
                    details={"order_details": order_details},
                    **trade_params
                )
                action_taken = True 

                if order_details and order_details.get('status') not in ['failed', 'rejected', 'cancelled', None, ''] and order_details.get('order_id'): # Basic check for success
                    _log_structured_event(
                        logging.INFO,
                        "TRADE_SUBMITTED_SUCCESS",
                        f"Trade submitted successfully via broker for {symbol}. Side: {side}, Qty: {quantity:.8f}. Order ID: {order_details.get('order_id')}",
                        details={"order_details": order_details},
                        **trade_params
                    )
                    # --- Update portfolio_state based on successful trade --- 
                    filled_price = latest_price # Assume fill at latest_price for now
                    filled_quantity = quantity # Use the requested quantity for calculation initially

                    # Try to use actual fill details if available from broker response
                    if order_details.get('avg_fill_price') and order_details.get('filled_qty'):
                        try:
                            # Ensure these are valid numbers before converting to Decimal
                            avg_fill_price_str = str(order_details['avg_fill_price'])
                            filled_qty_str = str(order_details['filled_qty'])
                            if avg_fill_price_str and filled_qty_str: # Check they are not empty or None
                                filled_price = Decimal(avg_fill_price_str)
                                filled_quantity = Decimal(filled_qty_str)
                                _log_structured_event(logging.INFO, "FILL_DETAILS_USED", f"Using actual fill details for {symbol}: Price={filled_price}, Qty={filled_quantity}", details=order_details)
                        except (InvalidOperation, TypeError, ValueError) as e_fill_parse: # Catch conversion errors
                            _log_structured_event(logging.WARNING, "FILL_DETAILS_PARSE_ERROR", f"Could not parse fill details from order_details for {symbol}: {e_fill_parse}. Reverting to estimated price/qty.", details=order_details)
                            # filled_price and filled_quantity remain as initially assumed

                    asset_base = symbol.split('-')[0]
                    current_asset_state = self.portfolio_state.get(
                        asset_base, 
                        {'quantity': Decimal('0'), 'average_cost': Decimal('0'), 'initial_investment': Decimal('0')}
                    )
                    
                    current_qty = current_asset_state['quantity']
                    current_avg_cost = current_asset_state['average_cost']
                    current_initial_investment = current_asset_state['initial_investment']

                    if side == 'buy':
                        new_total_qty = current_qty + filled_quantity
                        new_investment_this_trade = filled_price * filled_quantity
                        new_total_initial_investment = current_initial_investment + new_investment_this_trade
                        
                        if new_total_qty > Decimal('1e-12'): # Avoid division by zero if qty is effectively zero
                            new_avg_cost = new_total_initial_investment / new_total_qty
                        else:
                            new_avg_cost = Decimal('0') # Or keep old avg_cost if preferred when qty is zero
                        
                        self.portfolio_state[asset_base] = {
                            'quantity': new_total_qty,
                            'average_cost': new_avg_cost,
                            'initial_investment': new_total_initial_investment
                        }
                        _log_structured_event(logging.INFO, "PORTFOLIO_STATE_BUY_UPDATE", 
                                            f"Updated portfolio for BUY {asset_base}: Qty={new_total_qty:.8f}, AvgCost={new_avg_cost:.2f}, InitInv={new_total_initial_investment:.2f}", 
                                            details=self.portfolio_state[asset_base])
                    
                    elif side == 'sell':
                        proceeds_this_trade = filled_price * filled_quantity
                        # Reduce initial investment proportionally to the quantity sold relative to average cost
                        # This recognizes the cost of goods sold (COGS)
                        cost_of_goods_sold = current_avg_cost * filled_quantity 
                        # Cap COGS at current initial investment to prevent negative investment
                        cost_of_goods_sold = min(cost_of_goods_sold, current_initial_investment) 

                        new_total_qty = current_qty - filled_quantity
                        new_total_initial_investment = current_initial_investment - cost_of_goods_sold

                        # Average cost of remaining shares does not change
                        new_avg_cost = current_avg_cost 

                        if new_total_qty <= Decimal('1e-8'): # Effectively zero or less
                            _log_structured_event(logging.INFO, "PORTFOLIO_STATE_SELL_CLOSE", 
                                                f"Closing portfolio position for {asset_base} due to sell. Qty became {new_total_qty:.8f}. Proceeds: {proceeds_this_trade:.2f}, COGS: {cost_of_goods_sold:.2f}", 
                                                details={'asset': asset_base, 'remaining_qty': float(new_total_qty)})
                            # Option 1: Remove the asset from portfolio_state
                            # if asset_base in self.portfolio_state: del self.portfolio_state[asset_base]
                            # Option 2: Keep the entry but with zero quantity and adjusted investment (potentially zero or very small)
                            self.portfolio_state[asset_base] = {
                                'quantity': Decimal('0.0'), # Ensure it's explicitly zero
                                'average_cost': new_avg_cost, # Keep avg cost for record, or set to 0
                                'initial_investment': max(Decimal('0.0'), new_total_initial_investment) # Ensure not negative
                            }
                        else:
                            self.portfolio_state[asset_base] = {
                                'quantity': new_total_qty,
                                'average_cost': new_avg_cost, # Remains the same
                                'initial_investment': new_total_initial_investment
                            }
                            _log_structured_event(logging.INFO, "PORTFOLIO_STATE_SELL_UPDATE", 
                                                f"Updated portfolio for SELL {asset_base}: Qty={new_total_qty:.8f}, AvgCost={new_avg_cost:.2f}, InitInv={new_total_initial_investment:.2f}. Proceeds: {proceeds_this_trade:.2f}, COGS: {cost_of_goods_sold:.2f}",
                                                details=self.portfolio_state[asset_base])

            else:
                _log_structured_event(
                    logging.ERROR,
                    "BROKER_ORDER_REJECTED_ERROR",
                    f"Trade submission failed, was rejected, or no order_id returned by broker for {symbol}. Response: {order_details}",
                    details={"order_details": order_details},
                    **trade_params
                )
        except Exception as e_broker:
            _log_structured_event(
                logging.ERROR,
                "BROKER_API_ERROR",
                f"Exception during broker API call for {symbol} ({side} {quantity:.8f}): {e_broker}",
                details={"error_message": str(e_broker), "exc_info": True, "broker_response": order_details}, # Log what we got before the exception if any
                **trade_params
            )

    def get_holdings(self):
        """Fetches current crypto holdings using the broker client and updates self.portfolio_state."""
        _log_structured_event(logging.INFO, "HOLDINGS_FETCH_ATTEMPT", "Attempting to fetch current holdings.")

        if not self.broker:
            _log_structured_event(
                logging.ERROR,
                "BROKER_UNAVAILABLE_FOR_HOLDINGS_ERROR",
                "Broker is not initialized. Cannot fetch holdings."
            )
            # self.portfolio_state remains as is or empty if never populated
            return

        try:
            # broker_holdings is expected to be Dict[str (asset_code), str (quantity)]
            broker_holdings = self.broker.get_holdings()
            
            if broker_holdings is not None and isinstance(broker_holdings, dict):
                processed_assets = set()
                for asset_code, qty_str in broker_holdings.items():
                    processed_assets.add(asset_code)
                    try:
                        current_qty_from_broker = Decimal(str(qty_str))
                        if asset_code not in self.portfolio_state:
                            # New asset detected from broker, not tracked by bot's trades yet
                            self.portfolio_state[asset_code] = {
                                'quantity': current_qty_from_broker,
                                'average_cost': Decimal('0.0'),
                                'initial_investment': Decimal('0.0')
                            }
                            _log_structured_event(
                                logging.WARNING,
                                "UNTRACKED_ASSET_DETECTED",
                                f"Asset {asset_code} detected from broker but not previously tracked by bot. Cost basis unknown, initialized to 0.",
                                details={"asset_code": asset_code, "quantity": float(current_qty_from_broker)}
                            )
                        else:
                            # Asset already tracked, just update quantity from broker
                            # This assumes broker's quantity is the source of truth for current quantity
                            self.portfolio_state[asset_code]['quantity'] = current_qty_from_broker
                    except Exception as e_conv: # Broader exception for Decimal conversion or other issues
                        _log_structured_event(
                            logging.WARNING,
                            "INVALID_HOLDING_QUANTITY_FORMAT",
                            f"Could not process holding quantity '{qty_str}' for asset '{asset_code}': {e_conv}. Skipping this asset.",
                            details={"asset_code": asset_code, "invalid_quantity": str(qty_str)}
                        )
                
                # For assets in portfolio_state but NOT in broker_holdings (e.g., sold outside bot, or broker error for that asset)
                # set their quantity to zero, but keep avg_cost and initial_investment for potential PNL calc if needed for other logic.
                # Or, if bot assumes it controls all, then it implies it was sold.
                assets_in_state = list(self.portfolio_state.keys())
                for asset_code_in_state in assets_in_state:
                    if asset_code_in_state not in processed_assets and asset_code_in_state != 'USD': # Don't zero out USD if we ever put it here
                        _log_structured_event(
                            logging.INFO,
                            "ASSET_REMOVED_OR_ZEROED",
                            f"Asset {asset_code_in_state} was in portfolio_state but not in broker's current holdings. Setting quantity to 0.",
                            details={"asset_code": asset_code_in_state, "previous_quantity": float(self.portfolio_state[asset_code_in_state]['quantity'])}
                        )
                        self.portfolio_state[asset_code_in_state]['quantity'] = Decimal('0.0')
                        # Note: average_cost and initial_investment are kept. If quantity is 0, PNL should be 0.

                _log_structured_event(
                    logging.INFO,
                    "HOLDINGS_FETCH_SUCCESS",
                    f"Successfully processed {len(broker_holdings)} holdings from broker.",
                    details={
                        "portfolio_state_assets": list(self.portfolio_state.keys()),
                        "assets_from_broker_count": len(broker_holdings)
                    }
                )
            elif broker_holdings is None:
                _log_structured_event(
                    logging.WARNING,
                    "NO_HOLDINGS_DATA_RETURNED_WARNING",
                    "Broker returned None when fetching holdings. Assuming no crypto holdings or API issue."
                )
                # Set all quantities in portfolio_state to 0 if broker says none
                for asset_code_in_state in list(self.portfolio_state.keys()):
                    if asset_code_in_state != 'USD':
                        self.portfolio_state[asset_code_in_state]['quantity'] = Decimal('0.0')
            else: # Unexpected data type
                _log_structured_event(
                    logging.WARNING,
                    "UNEXPECTED_HOLDINGS_DATA_FORMAT_WARNING",
                    f"Broker returned unexpected data type for holdings: {type(broker_holdings)}. Expected dict or None.",
                    details={"received_type": str(type(broker_holdings))}
                )
                # Potentially zero out quantities in portfolio_state as data is unreliable
                for asset_code_in_state in list(self.portfolio_state.keys()):
                    if asset_code_in_state != 'USD':
                        self.portfolio_state[asset_code_in_state]['quantity'] = Decimal('0.0')

        except Exception as e:
            _log_structured_event(
                logging.ERROR,
                "BROKER_API_HOLDINGS_FETCH_ERROR",
                f"Failed to fetch holdings due to broker API error: {e}",
                details={"error_message": str(e), "exc_info": True}
            )
            # Don't wipe portfolio_state here, could be temporary API issue.
            # Let existing tracked values persist, but they might be stale.

    def _fetch_latest_data(self):
        """Fetches the latest market data for tracked symbols."""
        logging.debug("Fetching latest market data...")
        # Implement this method as needed

    def run_trading_cycle(self):
        """Executes one cycle of fetching data, analyzing, and potentially trading."""
        _log_structured_event(logging.INFO, "TRADING_CYCLE_START", "Starting new trading cycle.")
        cycle_start_time = datetime.now()

        if self.broker:
            _log_structured_event(logging.DEBUG, "HOLDINGS_UPDATE_START", "Attempting to update holdings.")
            try:
                self.get_holdings() # Update holdings each cycle
                _log_structured_event(
                    logging.DEBUG,
                    "HOLDINGS_UPDATE_COMPLETE",
                    "Holdings update process finished.",
                    details={"holdings": self.portfolio_state}
                )
            except Exception as e_holdings:
                _log_structured_event(
                    logging.ERROR,
                    "HOLDINGS_UPDATE_ERROR_IN_CYCLE",
                    f"Error updating holdings during trading cycle: {e_holdings}",
                    details={"error_message": str(e_holdings), "exc_info": True}
                )

        # Stores {'symbol': {'signal': str, 'latest_price': Decimal, 'analysis_data_present': bool}}
        symbol_actions = {}

        for symbol in self.config.SYMBOLS_TO_TRADE:
            action_details = {'signal': 'hold', 'latest_price': None, 'analysis_data_present': False}
            _log_structured_event(logging.DEBUG, "DATA_FETCH_FOR_SIGNAL_ATTEMPT",
                                f"Fetching data for {symbol} to generate signal.",
                                details={"symbol": symbol, "data_provider": self.data_provider.__class__.__name__})
            try:
                analysis_data = self.data_provider.fetch_price_history(symbol, days=self.rl_lookback_window + 35)

                if analysis_data is None or analysis_data.empty:
                    _log_structured_event(logging.WARNING, "DATA_UNAVAILABLE_FOR_SIGNAL_WARNING",
                                        f"No data available for {symbol} after fetch attempt. Skipping signal generation.",
                                        details={"symbol": symbol})
                    symbol_actions[symbol] = action_details # Record default 'hold' due to no data
                    continue # Skip to next symbol

                _log_structured_event(logging.DEBUG, "SIGNAL_GENERATION_INPUT_DATA_READY",
                                    f"Data ready for signal generation for {symbol}.",
                                    details={"symbol": symbol, "data_length": len(analysis_data)})

                action_details['analysis_data_present'] = True
                action_details['signal'] = self._get_signal(symbol, analysis_data)
                if not analysis_data['close'].empty:
                    action_details['latest_price'] = Decimal(str(analysis_data['close'].iloc[-1]))
                else:
                    _log_structured_event(logging.ERROR, "EMPTY_CLOSE_PRICE_IN_ANALYSIS_DATA",
                                        f"'close' column in analysis_data for {symbol} is empty. Cannot determine latest price.",
                                        details={"symbol": symbol})
                    action_details['latest_price'] = None # Explicitly set to None
                    action_details['signal'] = 'hold' # Fallback to hold if price is indeterminate

            except Exception as e_fetch_signal:
                _log_structured_event(logging.ERROR, "FETCH_OR_SIGNAL_ERROR_IN_CYCLE",
                                    f"Error during data fetch or signal generation for {symbol}: {e_fetch_signal}",
                                    details={"symbol": symbol, "error_message": str(e_fetch_signal), "exc_info": True})
                symbol_actions[symbol] = action_details # Store default 'hold'
            finally:
                symbol_actions[symbol] = action_details
                _log_structured_event(logging.DEBUG, "SIGNAL_PROCESSED_FOR_SYMBOL",
                                    f"Signal for {symbol}: {action_details['signal']}, Price: {action_details['latest_price']}",
                                    details={"symbol": symbol, "signal": action_details['signal'], "latest_price": str(action_details['latest_price']) if action_details['latest_price'] is not None else None})

        if self.enable_trading:
            _log_structured_event(logging.INFO, "TRADE_EXECUTION_PHASE_START", "Starting trade execution phase based on signals.")
            for symbol, details in symbol_actions.items():
                signal = details['signal']
                latest_price_for_trade = details['latest_price']

                if signal != 'hold':
                    trade_log_params = {"symbol": symbol, "signal": signal, "latest_price_for_trade": str(latest_price_for_trade) if latest_price_for_trade else None}
                    _log_structured_event(logging.INFO, "TRADE_EXECUTION_ATTEMPT", 
                                        f"Attempting trade for {symbol}: Signal={signal}, Price={latest_price_for_trade}", 
                                        details=trade_log_params)

                    if not details['analysis_data_present'] or latest_price_for_trade is None:
                        _log_structured_event(logging.ERROR, "MISSING_DATA_FOR_TRADE_EXECUTION_ERROR",
                                            f"Cannot execute trade for {symbol} due to missing analysis data or latest price.",
                                            details=trade_log_params)
                        continue # Skip to next symbol
                    
                    if latest_price_for_trade <= 0:
                        _log_structured_event(logging.ERROR, "INVALID_PRICE_FOR_TRADE_ERROR",
                                            f"Invalid latest price ({latest_price_for_trade}) for {symbol} from stored analysis. Skipping trade.",
                                            details=trade_log_params)
                        continue # Skip to next symbol
                    try:
                        self._execute_trade(symbol, signal, latest_price_for_trade)
                    except Exception as e_trade_exec:
                        _log_structured_event(logging.ERROR, "CYCLE_LEVEL_TRADE_EXECUTION_ERROR",
                                            f"Unhandled error executing trade for {symbol} with signal {signal} in cycle: {e_trade_exec}",
                                            details={"error_message": str(e_trade_exec), "exc_info": True},
                                            **trade_log_params)
        else:
            _log_structured_event(logging.INFO, "TRADING_EXECUTION_SKIPPED_DISABLED_INFO", 
                                "Trading is disabled for this cycle. Skipping trade execution phase.",
                                details={"signals_generated": {s: d['signal'] for s, d in symbol_actions.items()}})

        cycle_end_time = datetime.now()
        cycle_duration = (cycle_end_time - cycle_start_time).total_seconds()
        _log_structured_event(logging.INFO, "TRADING_CYCLE_COMPLETE", 
                            f"Trading cycle finished in {cycle_duration:.2f} seconds.",
                            details={
                                "cycle_duration_seconds": float(f"{cycle_duration:.2f}"),
                                "signals_generated": {s: d['signal'] for s, d in symbol_actions.items()},
                                "trades_attempted": 0 # Add this detail
                            })

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
