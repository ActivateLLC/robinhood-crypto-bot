# /Users/activate/Dev/robinhood-crypto-bot/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# --- Robinhood API Credentials ---
# IMPORTANT: Store your actual keys securely in the .env file
# DO NOT commit your keys to version control.
ROBINHOOD_API_KEY = os.getenv("ROBINHOOD_API_KEY", None)
ROBINHOOD_PRIVATE_KEY = os.getenv("ROBINHOOD_PRIVATE_KEY", None)

# --- Trading Parameters ---
# Comma-separated list of crypto trading pairs (e.g., "BTC,ETH-USD")
SYMBOLS_STR = os.getenv("SYMBOLS", "BTC")
SYMBOLS = [symbol.strip().upper() for symbol in SYMBOLS_STR.split(',') if symbol.strip()]

# Interval between trading cycles in minutes
CHECK_INTERVAL_RAW = os.getenv("CHECK_INTERVAL", "15")
INTERVAL_MINUTES = int(CHECK_INTERVAL_RAW.split('#')[0].strip()) # Convert to integer after stripping

# Base amount in USD for buy orders (if using amount-based orders)
TRADE_AMOUNT_USD_RAW = os.getenv("TRADE_AMOUNT_USD", "50.0")
TRADE_AMOUNT_USD = float(TRADE_AMOUNT_USD_RAW.split('#')[0].strip()) # Example: $50 per buy trade

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

# --- Data Provider Settings ---
# Preference order for data providers ("yfinance", "coingecko")
DATA_PROVIDER_PREFERENCE = os.getenv("DATA_PROVIDER_PREFERENCE", "yfinance").lower()

# Number of days of historical data to fetch for indicators
LOOKBACK_DAYS_RAW = os.getenv("LOOKBACK_DAYS", "90") # Default to 90 if not set
LOOKBACK_DAYS = int(LOOKBACK_DAYS_RAW.split('#')[0].strip()) # Convert to integer after stripping

# yfinance specific settings
YFINANCE_PERIOD_RAW = os.getenv("YFINANCE_PERIOD", "6mo") # Increased period to fetch more data points
YFINANCE_PERIOD = YFINANCE_PERIOD_RAW.split('#')[0].strip()
YFINANCE_INTERVAL_RAW = os.getenv("YFINANCE_INTERVAL", "1h") # Default to 1 hour
YFINANCE_INTERVAL = YFINANCE_INTERVAL_RAW.split('#')[0].strip() # Data interval (e.g., 1m, 5m, 15m, 30m, 1h, 1d)

# CoinGecko specific settings
COINGECKO_DAYS = os.getenv("COINGECKO_DAYS", "90") # Match LOOKBACK_DAYS
COINGECKO_VS_CURRENCY = os.getenv("COINGECKO_VS_CURRENCY", "usd") # Currency for price data

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FILE = os.getenv("LOG_FILE", "crypto_bot.log") # Name of the log file

# --- Plotting (Optional) ---
PLOT_ENABLED = os.getenv("PLOT_ENABLED", "False").lower() == "true"
PLOT_OUTPUT_DIR = os.getenv("PLOT_OUTPUT_DIR", "plots") # Directory to save plots

# --- Validation ---
if not SYMBOLS:
    raise ValueError("SYMBOLS environment variable cannot be empty.")
if ENABLE_TRADING and (not ROBINHOOD_API_KEY or not ROBINHOOD_PRIVATE_KEY):
    print("WARNING: ENABLE_TRADING is True, but Robinhood API keys are missing in .env file. Trading will be disabled.")
    ENABLE_TRADING = False # Force disable if keys are missing

print("--- Configuration Loaded ---")
print(f"Symbols: {SYMBOLS}")
print(f"Interval: {INTERVAL_MINUTES} minutes")
print(f"Enable Trading: {ENABLE_TRADING}")
print(f"Enable RL Model: {ENABLE_RL_MODEL}")
print(f"Traditional Strategy: {TRADING_STRATEGY}")
print(f"Log Level: {LOG_LEVEL}")
print(f"Data Provider: {DATA_PROVIDER_PREFERENCE}")
print("--------------------------")
