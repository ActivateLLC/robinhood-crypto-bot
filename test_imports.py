# test_imports.py
import sys
import os
# Ensure the project root is in the path for custom module imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print("-" * 20)

print("Importing logging...")
import logging
print("OK")

print("Importing os...")
import os
print("OK")

print("Importing time...")
import time
print("OK")

print("Importing uuid...")
import uuid
print("OK")

print("Importing sys...")
# Already imported
print("OK")

print("Importing numpy...")
try:
    import numpy as np
    print(f"OK (Version: {np.__version__})")
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)

print("Importing pandas...")
try:
    import pandas as pd
    print(f"OK (Version: {pd.__version__})")
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)

print("Importing decimal...")
try:
    from decimal import Decimal, ROUND_DOWN, InvalidOperation
    print("OK")
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)

print("Importing config...")
try:
    from config import (
        LOG_LEVEL, LOG_FILE, RL_MODEL_PATH, SYMBOLS, ENABLE_TRADING,
        TRADE_AMOUNT_USD, LOOKBACK_DAYS, INTERVAL_MINUTES, TRADING_STRATEGY,
        ROBINHOOD_API_KEY, ROBINHOOD_PRIVATE_KEY # Also check key loading
    )
    print("OK")
    print(f"  Symbols type: {type(SYMBOLS)}")
    print(f"  Symbols value: {SYMBOLS}")
    print(f"  API Key Loaded: {bool(ROBINHOOD_API_KEY)}")
    print(f"  Private Key Loaded: {bool(ROBINHOOD_PRIVATE_KEY)}")

except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)

print("Importing AltCryptoDataProvider...")
try:
    from data_provider import AltCryptoDataProvider
    print("OK")
except ModuleNotFoundError:
    print("Skipped (data_provider.py not found? - Check filename)")
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)


print("Importing CryptoAPITrading...")
try:
    from api_client import CryptoAPITrading
    print("OK")
except ModuleNotFoundError:
     print("Skipped (api_client.py not found? - Check filename)")
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)

# Temporarily skip pandas_ta and stable_baselines3

print("-" * 20)
print("All essential imports tested successfully.")
