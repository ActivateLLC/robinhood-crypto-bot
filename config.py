# /Users/activate/Dev/robinhood-crypto-bot/config.py
import os
from dotenv import load_dotenv
import sys
import json

# Load environment variables from .env file
load_dotenv(override=True)

# --- Environment Variables (loaded via dotenv) ---
# --- Logging --- Use standard logging library config
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_FILE = os.getenv('LOG_FILE', 'logs/crypto_bot.log')
EXPERIENCE_LOG_FILE = os.getenv('EXPERIENCE_LOG_FILE', 'logs/live_experience.log')

# --- Trading Parameters --- 
ENABLE_TRADING = os.getenv('ENABLE_TRADING', 'False').lower() == 'true'
TRADE_AMOUNT_USD_RAW = os.getenv("TRADE_AMOUNT_USD", "50.0")
TRADE_AMOUNT_USD = float(TRADE_AMOUNT_USD_RAW.split('#')[0].strip()) # Example: $50 per buy trade
SYMBOLS_STR = os.getenv("SYMBOLS_TO_TRADE", "BTC-USD") # Read the correct env var
SYMBOLS = [symbol.strip().upper() for symbol in SYMBOLS_STR.split(',') if symbol.strip()]

# Interval between trading cycles in minutes
CHECK_INTERVAL_RAW = os.getenv("CHECK_INTERVAL", "15")
INTERVAL_MINUTES = int(CHECK_INTERVAL_RAW.split('#')[0].strip()) # Convert to integer after stripping

# Percentage of holdings to sell (1.0 = 100%, 0.5 = 50%)
TRADE_SELL_PERCENT = float(os.getenv("TRADE_SELL_PERCENT", "1.0"))

# Master switch to enable/disable actual trade execution via API
enable_trading_raw = os.getenv("ENABLE_TRADING", "False") # Read raw value, revert default
ENABLE_TRADING = enable_trading_raw.lower().strip() == "true" # Process raw value (added strip)

# --- Traditional Strategy Selection ---
# Strategy to use if RL model is disabled or fails (e.g., 'SMA', 'MACD', 'RSI_BASELINE')
# Load the strategy, strip any inline comments, and convert to upper case
TRADING_STRATEGY_RAW = os.getenv("TRADING_STRATEGY", "SMA")
TRADING_STRATEGY = TRADING_STRATEGY_RAW.split('#')[0].strip().upper()

# --- RL Model Configuration ---
# Enable/disable using the RL model for signals
ENABLE_RL_MODEL = os.getenv("ENABLE_RL_MODEL", "True").lower() == "true"

# Path to the trained RL model zip file
RL_MODEL_PATH_RAW = os.getenv("RL_MODEL_PATH", "models/ppo_crypto_final.zip")
RL_MODEL_PATH = RL_MODEL_PATH_RAW.split('#')[0].strip()

# Path to the normalization parameters file
RL_PARAMS_PATH_RAW = os.getenv("RL_PARAMS_PATH", "models/normalization_params.pkl")
RL_PARAMS_PATH = RL_PARAMS_PATH_RAW.split('#')[0].strip()

# Lookback window size used during RL training (MUST match training env)
RL_LOOKBACK_WINDOW = int(os.getenv('RL_LOOKBACK_WINDOW', '30'))

# --- Broker Credentials (Handle with care!) ---
# Load directly in the script or secure config manager, not hardcoded here
BROKER_TYPE = os.getenv('BROKER_TYPE', 'robinhood') # Example: 'robinhood' or 'ibkr'
ROBINHOOD_API_KEY = os.getenv('ROBINHOOD_API_KEY')
ROBINHOOD_BASE64_PRIVATE_KEY = os.getenv('BASE64_PRIVATE_KEY') # Corrected env var name

# --- Data Provider Settings ---
# Preference order for data providers ("yfinance", "coingecko")
DATA_PROVIDER_PREFERENCE = os.getenv("DATA_PROVIDER_PREFERENCE", "yfinance").lower()

# Number of days of historical data to fetch for indicators
LOOKBACK_DAYS_RAW = os.getenv("LOOKBACK_DAYS", "90") # Default to 90 if not set
LOOKBACK_DAYS = int(LOOKBACK_DAYS_RAW.split('#')[0].strip()) # Convert to integer after stripping

# yfinance specific settings
YFINANCE_PERIOD_RAW = os.getenv("YFINANCE_PERIOD", "60d") # Reduced for 5m interval
YFINANCE_PERIOD = YFINANCE_PERIOD_RAW.split('#')[0].strip()
YFINANCE_INTERVAL_RAW = os.getenv("YFINANCE_INTERVAL", "1h") # Default to 1 hour
YFINANCE_INTERVAL = YFINANCE_INTERVAL_RAW.split('#')[0].strip() # Data interval (e.g., 1m, 5m, 15m, 30m, 1h, 1d)

# CoinGecko specific settings
COINGECKO_DAYS = os.getenv("COINGECKO_DAYS", "90") # Match LOOKBACK_DAYS
COINGECKO_VS_CURRENCY = os.getenv("COINGECKO_VS_CURRENCY", "usd") # Currency for price data

# --- Plotting (Optional) ---
PLOT_ENABLED = os.getenv("PLOT_ENABLED", "False").lower() == "true"
PLOT_OUTPUT_DIR = os.getenv("PLOT_OUTPUT_DIR", "plots") # Directory to save plots

# --- Validation ---
if not SYMBOLS:
    raise ValueError("SYMBOLS environment variable cannot be empty.")
if ENABLE_TRADING and not ROBINHOOD_API_KEY:
    print("WARNING: ENABLE_TRADING is True, but Robinhood API key is missing in .env file. Trading will be disabled.")
    ENABLE_TRADING = False # Force disable if key is missing
if ENABLE_TRADING and not ROBINHOOD_BASE64_PRIVATE_KEY:
    print("WARNING: ENABLE_TRADING is True, but Robinhood Base64 Private Key is missing in .env file. Trading will be disabled.")
    ENABLE_TRADING = False # Force disable if private key is missing

print("--- Configuration Loaded ---")
print(f"Symbols: {SYMBOLS}")
print(f"Interval: {INTERVAL_MINUTES} minutes")
print(f"Enable Trading: {ENABLE_TRADING}")
print(f"Enable RL Model: {ENABLE_RL_MODEL}")
print(f"Traditional Strategy: {TRADING_STRATEGY}")
print(f"Log Level: {LOG_LEVEL}")
print(f"Data Provider: {DATA_PROVIDER_PREFERENCE}")
print(f"Broker Type: {BROKER_TYPE}")
print(f"Robinhood API Key: {ROBINHOOD_API_KEY}")
print(f"Robinhood Base64 Private Key: {ROBINHOOD_BASE64_PRIVATE_KEY}")
print("--------------------------")

# --- Configuration Class --- Ensures type safety and central access
from dataclasses import dataclass
from decimal import Decimal

@dataclass(frozen=True) # Make config immutable after creation
class Config:
    # Logging
    LOG_LEVEL: str
    LOG_FILE: str
    EXPERIENCE_LOG_FILE: str

    # Trading Parameters
    ENABLE_TRADING: bool
    TRADE_AMOUNT_USD: Decimal
    SYMBOLS_TO_TRADE: list
    TRADING_STRATEGY: str

    # Broker Config
    BROKER_TYPE: str
    ROBINHOOD_API_KEY: str
    ROBINHOOD_BASE64_PRIVATE_KEY: str

    # Data Fetching
    LOOKBACK_DAYS: int
    YFINANCE_PERIOD: str
    YFINANCE_INTERVAL: str

    # RL Model Configuration
    RL_MODEL_PATH: str
    RL_LOOKBACK_WINDOW: int
    RL_REWARD_FUNCTION: str
    INITIAL_BALANCE_USD: Decimal
    TRAINING_MODE: bool
    INTERVAL_MINUTES: int
    PLOT_ENABLED: bool
    PLOT_OUTPUT_DIR: str
    DATA_PROVIDER_PREFERENCE: str
    ENABLE_RL_MODEL: bool

# --- Config Loading Function ---
def load_config() -> Config:
    """Loads configuration from environment variables and returns a Config object."""
    load_dotenv(override=True) # Ensure .env is loaded

    # --- Load from Environment --- Add default values
    config_data = {
        'LOG_LEVEL': LOG_LEVEL, # Use constant
        'LOG_FILE': LOG_FILE,   # Use constant
        'EXPERIENCE_LOG_FILE': EXPERIENCE_LOG_FILE, # Added config value
        'ENABLE_TRADING': ENABLE_TRADING,
        'TRADE_AMOUNT_USD': Decimal(os.getenv('TRADE_AMOUNT_USD', '10.00')),
        'SYMBOLS_TO_TRADE': [symbol.strip() for symbol in os.getenv('SYMBOLS_TO_TRADE', 'BTC-USD').split(',')].copy(),
        'TRADING_STRATEGY': os.getenv('TRADING_STRATEGY', 'rl').lower(), # Default to 'rl'

        # Broker Config
        'BROKER_TYPE': os.getenv('BROKER_TYPE', 'robinhood').lower(),
        'ROBINHOOD_API_KEY': os.getenv('ROBINHOOD_API_KEY'), # Load keys into config
        'ROBINHOOD_BASE64_PRIVATE_KEY': os.getenv('BASE64_PRIVATE_KEY'), # Correct env name

        # Data Fetching
        'LOOKBACK_DAYS': int(os.getenv('LOOKBACK_DAYS', '90')),
        'YFINANCE_PERIOD': os.getenv('YFINANCE_PERIOD', '3mo'),
        'YFINANCE_INTERVAL': os.getenv('YFINANCE_INTERVAL', '1h'),

        'RL_MODEL_PATH': os.getenv('RL_MODEL_PATH', 'models/ppo_crypto_trader.zip'),
        'RL_LOOKBACK_WINDOW': int(os.getenv('RL_LOOKBACK_WINDOW', '60')),
        'RL_REWARD_FUNCTION': os.getenv('RL_REWARD_FUNCTION', 'simple_profit'),
        'INITIAL_BALANCE_USD': Decimal(os.getenv('INITIAL_BALANCE_USD', '1000.00')),
        'TRAINING_MODE': os.getenv('TRAINING_MODE', 'False').lower() == 'true',
        'INTERVAL_MINUTES': int(os.getenv('INTERVAL_MINUTES', '5')),
        'PLOT_ENABLED': os.getenv('PLOT_ENABLED', 'False').lower() == 'true',
        'PLOT_OUTPUT_DIR': os.getenv('PLOT_OUTPUT_DIR', 'plots'),
        'DATA_PROVIDER_PREFERENCE': os.getenv('DATA_PROVIDER_PREFERENCE', 'yfinance').lower(),
        'ENABLE_RL_MODEL': os.getenv('ENABLE_RL_MODEL', 'True').lower() == 'true',
    }

    # Basic validation (Example: Check if keys exist if trading enabled)
    if config_data['ENABLE_TRADING']:
        if config_data['BROKER_TYPE'] == 'robinhood':
            if not config_data['ROBINHOOD_API_KEY'] or not config_data['ROBINHOOD_BASE64_PRIVATE_KEY']:
                # Use print here as logging might not be set up yet
                print("ERROR: Robinhood API Key and Private Key are required in .env when ENABLE_TRADING=True and BROKER_TYPE=robinhood.")
                sys.exit(1) # Exit if required keys are missing
        # Add more validation as needed (e.g., for IBKR keys)
        elif config_data['BROKER_TYPE'] == 'ibkr':
            # Add checks for IBKR config from .env if needed
            pass

    return Config(**config_data)

# --- Logging Setup Function ---
import logging
from logging.handlers import RotatingFileHandler
import sys

def setup_logging(config: Config):
    """Configures logging based on the loaded configuration."""
    # --- Get the root logger --- #
    root_logger = logging.getLogger()

    # --- Clear existing handlers (optional, but good practice to avoid duplicates) --- #
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # --- Set the level on the root logger --- #
    log_level_name = config.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_name, logging.INFO) # Default to INFO if invalid
    root_logger.setLevel(log_level)

    # --- Create Formatter --- #
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- File Handler Setup --- #
    try:
        log_dir = os.path.dirname(config.LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"Created log directory: {log_dir}") # Use print as logging might not be fully ready

        log_file_path = os.path.abspath(config.LOG_FILE)
        # Use RotatingFileHandler for log rotation
        file_handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5) # 10MB per file, 5 backups
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file handler for {config.LOG_FILE}: {e}", file=sys.stderr)

    # --- Console Handler Setup (Optional but Recommended) --- #
    console_handler = logging.StreamHandler(sys.stdout) # Log to stdout
    console_handler.setFormatter(formatter)
    # Optionally set a different level for console output, e.g., INFO
    # console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # --- Log completion --- #
    # Use the root logger now that it's configured
    root_logger.info(f"Logging setup complete. Root logger level: {log_level_name}, File: {config.LOG_FILE}")

    # --- Setup Experience Logger (Separate Handler) --- #
    try:
        exp_log_dir = os.path.dirname(config.EXPERIENCE_LOG_FILE)
        if exp_log_dir and not os.path.exists(exp_log_dir):
            os.makedirs(exp_log_dir)
            print(f"Created experience log directory: {exp_log_dir}")

        exp_log_path = os.path.abspath(config.EXPERIENCE_LOG_FILE)
        experience_logger = logging.getLogger('experience_logger')
        
        # --- Clear existing handlers FIRST --- #
        if experience_logger.hasHandlers():
            experience_logger.handlers.clear()
        # ----------------------------------- #
        
        experience_logger.setLevel(logging.INFO) # Log all experience entries
        experience_logger.propagate = False # Do not send to root logger handlers

        # Use a simple formatter, or just log JSON directly
        formatter = logging.Formatter('%(message)s') # Simple formatter to just log the message (JSON string)
        exp_file_handler = RotatingFileHandler(exp_log_path, maxBytes=50*1024*1024, backupCount=3) # 50MB, 3 backups
        exp_file_handler.setFormatter(formatter) # Set the formatter

        experience_logger.addHandler(exp_file_handler)
        root_logger.info(f"Experience logger setup complete. File: {exp_log_path}")

    except Exception as e:
        print(f"Error setting up experience logger for {config.EXPERIENCE_LOG_FILE}: {e}", file=sys.stderr)

    # --- Force level for specific noisy loggers (if needed) --- #
    # Explicitly set the broker logger to DEBUG, overriding the root level if necessary
    broker_logger = logging.getLogger("brokers.robinhood_broker")
    broker_logger.setLevel(logging.DEBUG)
    # Confirm the level was set correctly *within* setup_logging
    root_logger.info(f"Set 'brokers.robinhood_broker' level. Effective level now: {broker_logger.getEffectiveLevel()}")
    # logging.getLogger("requests").setLevel(logging.WARNING)
    # logging.getLogger("urllib3").setLevel(logging.WARNING)
    # logging.getLogger("schedule").setLevel(logging.INFO)

# --- Example Usage (Optional) ---
if __name__ == '__main__':
    config = load_config()
    setup_logging(config)
