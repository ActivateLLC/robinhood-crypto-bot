import logging
import logging.handlers
import os
import json
from dotenv import load_dotenv

# Load environment variables to ensure LOG_LEVEL and LOG_FILE are available
load_dotenv()

PORTFOLIO_FILE = 'portfolio.json'
DEFAULT_PORTFOLIO = {'cash': 10000.0, 'holdings': {}} # Example starting cash

# Initialize logger globally for the module if needed elsewhere,
# but typically setup_logging() is called once in the main script.
# logger = setup_logging()


def load_portfolio() -> dict:
    """Loads the portfolio from a JSON file, returning defaults if not found or invalid."""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                portfolio = json.load(f)
                # Basic validation (ensure expected keys exist)
                if 'cash' in portfolio and 'holdings' in portfolio:
                    logging.info(f"Loaded portfolio from {PORTFOLIO_FILE}")
                    return portfolio
                else:
                    logging.warning(f"Invalid format in {PORTFOLIO_FILE}. Using default portfolio.")
                    return DEFAULT_PORTFOLIO.copy()
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {PORTFOLIO_FILE}. Using default portfolio.")
            return DEFAULT_PORTFOLIO.copy()
        except Exception as e:
            logging.error(f"Error loading portfolio from {PORTFOLIO_FILE}: {e}", exc_info=True)
            return DEFAULT_PORTFOLIO.copy()
    else:
        logging.info(f"{PORTFOLIO_FILE} not found. Starting with default portfolio.")
        # Save the default portfolio for the first run
        save_portfolio(DEFAULT_PORTFOLIO.copy())
        return DEFAULT_PORTFOLIO.copy()

def save_portfolio(portfolio: dict):
    """Saves the current portfolio state to a JSON file."""
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolio, f, indent=4)
        logging.debug(f"Portfolio saved to {PORTFOLIO_FILE}")
    except Exception as e:
        logging.error(f"Failed to save portfolio to {PORTFOLIO_FILE}: {e}", exc_info=True)

# Placeholder for check_api_limit - Implement actual logic if needed
def check_api_limit():
    """Placeholder function to check API rate limits. Implement as needed."""
    # Example: Add logic here to track requests or query API status if available
    # Since logger might not be initialized when this is called depending on import order,
    # use logging.debug directly or ensure logger is available.
    logging.debug("API limit check placeholder called.")
    return True # Assume okay for now
