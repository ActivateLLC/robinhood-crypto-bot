import os
import sys
from dotenv import load_dotenv
from robinhood_api_client import CryptoAPITrading

# Load environment variables from .env file
load_dotenv()

print("Attempting to fetch Robinhood account balance...")

# Get API credentials from environment
api_key = os.getenv("ROBINHOOD_API_KEY")
private_key = os.getenv("ROBINHOOD_PRIVATE_KEY")

if not api_key or not private_key:
    print("Error: ROBINHOOD_API_KEY or ROBINHOOD_PRIVATE_KEY not found in .env file.", file=sys.stderr)
    sys.exit(1)

try:
    # Initialize the API client
    print("Initializing API client...")
    # Instantiate the client without arguments; it reads keys from environment
    client = CryptoAPITrading()

    # Fetch account information
    print("Fetching account information from Robinhood...")
    account_info = client.get_account()

    if account_info:
        # --- Extract relevant balance information ---
        # Note: The exact keys might vary slightly based on the API response structure.
        # We'll focus on 'buying_power' as a primary indicator.
        buying_power_info = account_info.get('buying_power')
        cash_held_info = account_info.get('cash_held_for_orders')
        portfolio_cash_info = account_info.get('portfolio_cash') # Another potential field

        # Handle cases where info might be a direct string or a dict
        buying_power = buying_power_info if isinstance(buying_power_info, str) else (buying_power_info.get('amount', 'N/A') if buying_power_info else 'N/A')
        cash_held = cash_held_info if isinstance(cash_held_info, str) else (cash_held_info.get('amount', 'N/A') if cash_held_info else 'N/A')
        portfolio_cash = portfolio_cash_info if isinstance(portfolio_cash_info, str) else (portfolio_cash_info.get('amount', 'N/A') if portfolio_cash_info else 'N/A')


        print("\n--- Robinhood Account Balance ---")
        print(f"  Buying Power:          ${buying_power}")
        print(f"  Cash Held for Orders:  ${cash_held}")
        print(f"  Portfolio Cash Balance: ${portfolio_cash}")
        # You could print the full account_info dict here for more details if needed:
        # print("\nFull Account Details:")
        # import json
        # print(json.dumps(account_info, indent=2))
        print("-------------------------------")

    else:
        print("Error: Failed to retrieve account information. API keys might be invalid or network issues.", file=sys.stderr)
        sys.exit(1)

except Exception as e:
    print(f"An error occurred: {e}", file=sys.stderr)
    import traceback
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)
