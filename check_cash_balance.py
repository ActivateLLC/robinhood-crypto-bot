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
    trader.api_key = api_key
    try:
        private_key_seed = base64.b64decode(private_key_b64)
        trader.private_key = trader.private_key.__class__(private_key_seed)
    except Exception as decode_err:
        print(f"Error decoding private key: {decode_err}")
        sys.exit(1)

    # Get Account Info
    print("Fetching account information...")
    account_info = trader.get_account()
    print("--- Account Info Response ---")
    print(account_info)
    print("---------------------------")

    # Extract and print buying power specifically if possible
    if account_info and isinstance(account_info, list) and len(account_info) > 0:
         # Assuming the structure is a list containing a dictionary
         acc_data = account_info[0]
         if isinstance(acc_data, dict):
             buying_power = acc_data.get('buying_power', {}).get('amount')
             currency = acc_data.get('buying_power', {}).get('currency')
             if buying_power and currency:
                 print(f"\nAvailable Buying Power: {buying_power} {currency}")
             else:
                 print("\nCould not extract buying power from response.")
         else:
             print("\nAccount info structure not as expected (expected dict in list).")
    else:
        print("\nCould not extract buying power. Response format might have changed or was empty/invalid.")

except ValueError as ve:
    print(f"Configuration Error: {ve}")
except ImportError as ie:
     print(f"Import Error: {ie}. Is the virtual environment activated?")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
