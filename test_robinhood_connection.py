import os
import sys
import logging
from decimal import Decimal
from dotenv import load_dotenv

# Ensure the brokers directory is in the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from brokers.robinhood import RobinhoodBroker

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def test_connection():
    """Tests the connection and basic API calls to Robinhood."""
    logging.info("Attempting to load Robinhood credentials from .env file...")
    loaded = load_dotenv() 
    if not loaded:
        logging.warning("'.env' file not found or empty. Ensure ROBINHOOD_API_KEY and ROBINHOOD_PRIVATE_KEY are set.")

    api_key = os.getenv("ROBINHOOD_API_KEY")
    private_key = os.getenv("ROBINHOOD_PRIVATE_KEY")

    if not api_key or not private_key:
        logging.error("API Key or Private Key not found in environment variables. Cannot test connection.")
        return

    logging.info("Credentials found. Initializing RobinhoodBroker...")
    
    try:
        broker = RobinhoodBroker() 
        logging.info("Broker initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize RobinhoodBroker: {e}", exc_info=True)
        return

    # Test fetching capital
    logging.info("Attempting to fetch available capital...")
    try:
        capital = broker.get_available_capital()
        if capital is not None:
            logging.info(f"Successfully fetched available capital: {capital}")
        else:
            logging.warning("get_available_capital() returned None.")
    except Exception as e:
        logging.error(f"Error calling get_available_capital(): {e}", exc_info=True)

    # Test fetching current price
    symbol = 'BTC-USD'
    logging.info(f"Attempting to fetch current price for {symbol}...")
    try:
        price = broker.get_current_price(symbol)
        logging.info(f"Attempting to calculate mid-price for {symbol}...")
        if price is not None:
            logging.info(f"Successfully fetched current price for {symbol}: {price:.8f}")
            # Use market_data.bid and market_data.ask as per MarketData definition
            if broker.latest_market_data.get(symbol):
                 md = broker.latest_market_data[symbol]
                 if md and md.bid is not None and md.ask is not None:
                     mid_price = (Decimal(str(md.bid)) + Decimal(str(md.ask))) / 2
                     logging.info(f"Calculated mid-price for {symbol}: {mid_price:.8f} (Bid: {md.bid}, Ask: {md.ask})")
                 else:
                     logging.warning(f"Market data found for {symbol}, but bid/ask missing or None.")
            else:
                logging.warning(f"Could not retrieve latest market data object for {symbol} to calculate mid-price.")
        else:
            logging.warning(f"Could not fetch current price for {symbol}.")

    except Exception as e:
        logging.error(f"Error calling get_current_price('{symbol}'): {e}", exc_info=True)

    logging.info("Connection test finished.")

if __name__ == "__main__":
    test_connection()
