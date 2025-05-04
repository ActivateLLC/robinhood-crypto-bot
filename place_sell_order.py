import os
import sys
import base64
import uuid
import json
from dotenv import load_dotenv

sys.path.append('.')
load_dotenv()

# --- Configuration ---
SYMBOL_TO_SELL = "POPCAT-USD"
QUANTITY_TO_SELL = "96.670000000000000000" # Exact quantity from holdings check
ORDER_TYPE = "market"
SIDE = "sell"
# -------------------

try:
    # Ensure the class is imported correctly
    try:
        from robinhood_api_trading import CryptoAPITrading
    except ImportError:
        print("Error: Could not import CryptoAPITrading. Make sure robinhood_api_trading.py is in the current directory.")
        sys.exit(1)

    # Load credentials
    api_key = os.getenv('ROBINHOOD_API_KEY')
    private_key_b64 = os.getenv('ROBINHOOD_PRIVATE_KEY')
    if not api_key or not private_key_b64:
        raise ValueError('ROBINHOOD_API_KEY or ROBINHOOD_PRIVATE_KEY not found in .env file.')

    # Initialize trader
    trader = CryptoAPITrading()
    trader.api_key = api_key
    try:
        private_key_seed = base64.b64decode(private_key_b64)
        trader.private_key = trader.private_key.__class__(private_key_seed)
    except Exception as decode_err:
        print(f"Error decoding private key: {decode_err}")
        sys.exit(1)

    # Prepare order details
    client_order_id = str(uuid.uuid4())
    # Market sell order config requires asset_quantity according to API error
    order_config = {"asset_quantity": QUANTITY_TO_SELL} # Corrected key

    print(f"Placing {SIDE} {ORDER_TYPE} order for {QUANTITY_TO_SELL} {SYMBOL_TO_SELL}...")
    print(f"Client Order ID: {client_order_id}")

    # Place the order
    order_response = trader.place_order(
        client_order_id=client_order_id,
        side=SIDE,
        order_type=ORDER_TYPE,
        symbol=SYMBOL_TO_SELL,
        order_config=order_config
    )

    print("--- Order Placement Response ---")
    print(order_response)
    print("------------------------------")

    if order_response and isinstance(order_response, dict) and order_response.get('status') == 'filled':
        print("\nOrder appears to be filled successfully!")
    elif order_response and isinstance(order_response, dict):
         print(f"\nOrder status: {order_response.get('status', 'Unknown')}. Check details above.")
    else:
        print("\nOrder placement failed or response format was unexpected.")

except ValueError as ve:
    print(f"Configuration Error: {ve}")
except ImportError as ie:
     print(f"Import Error: {ie}. Is the virtual environment activated?")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
