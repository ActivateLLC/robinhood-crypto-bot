import os
import sys
import base64
from dotenv import load_dotenv

sys.path.append('.')
load_dotenv()

try:
    # Ensure the class is imported correctly
    try:
        from robinhood_api_trading import CryptoAPITrading
    except ImportError:
        print("Error: Could not import CryptoAPITrading. Make sure robinhood_api_trading.py is in the current directory.")
        sys.exit(1)

    # Load credentials from environment variables
    api_key = os.getenv('ROBINHOOD_API_KEY')
    private_key_b64 = os.getenv('ROBINHOOD_PRIVATE_KEY')

    if not api_key or not private_key_b64:
        raise ValueError('ROBINHOOD_API_KEY or ROBINHOOD_PRIVATE_KEY not found in .env file.')

    # Initialize the trading class
    trader = CryptoAPITrading()

    # Manually set the API key and decode/set the private key
    # Note: The __init__ in the provided class might need adjustment if it expects keys differently
    trader.api_key = api_key
    try:
        private_key_seed = base64.b64decode(private_key_b64)
        # Use the existing private key object's class to create a new instance with the seed
        trader.private_key = trader.private_key.__class__(private_key_seed)
    except Exception as decode_err:
        print(f"Error decoding private key: {decode_err}")
        sys.exit(1)

    # Check for POPCAT-USD trading pair
    print("Checking for POPCAT-USD trading pair...")
    trading_pairs = trader.get_trading_pairs('POPCAT-USD') # Check specific pair
    print("--- Trading Pairs Response ---")
    print(trading_pairs)
    print("----------------------------")

    # Check if the pair was found
    if trading_pairs and isinstance(trading_pairs, dict) and trading_pairs.get('results'):
        print("\nPOPCAT-USD trading pair FOUND. Selling might be possible.")
    else:
        print("\nPOPCAT-USD trading pair NOT FOUND. Selling via API is likely not possible.")

except ValueError as ve:
    print(f"Configuration Error: {ve}")
except ImportError as ie:
     print(f"Import Error: {ie}. Is the virtual environment activated?")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
