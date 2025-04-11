import base64
import datetime
import json
import logging
import os
from typing import Any, Dict, Optional
import uuid
import requests
from nacl.signing import SigningKey
from dotenv import load_dotenv
from config import ROBINHOOD_API_KEY, ROBINHOOD_PRIVATE_KEY

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Set constants from imported config variables
API_KEY = ROBINHOOD_API_KEY
BASE64_PRIVATE_KEY = ROBINHOOD_PRIVATE_KEY

class CryptoAPITrading:
    def __init__(self):
        if not API_KEY or not BASE64_PRIVATE_KEY:
            logger.error("ROBINHOOD_API_KEY or ROBINHOOD_PRIVATE_KEY not found in environment variables.")
            raise ValueError("API Key and Private Key must be set in the environment.")

        self.api_key = API_KEY
        try:
            private_key_seed = base64.b64decode(BASE64_PRIVATE_KEY)
            self.private_key = SigningKey(private_key_seed)
            logger.info("Successfully initialized SigningKey from private key.")
        except (TypeError, base64.binascii.Error) as e:
            logger.error(f"Failed to decode or use private key: {e}")
            raise ValueError("Invalid Base64 Private Key format.") from e
        except Exception as e:
             logger.error(f"An unexpected error occurred initializing SigningKey: {e}")
             raise

        self.base_url = "https://trading.robinhood.com"
        logger.info(f"CryptoAPITrading client initialized for base URL: {self.base_url}")


    @staticmethod
    def _get_current_timestamp() -> int:
        return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())

    @staticmethod
    def get_query_params(key: str, *args: Optional[str]) -> str:
        if not args or all(arg is None for arg in args):
            return ""

        params = []
        for arg in args:
            if arg is not None: # Only include non-None args
                params.append(f"{key}={arg}")

        return "?" + "&".join(params) if params else ""


    def make_api_request(self, method: str, path: str, body: str = "") -> Any:
        timestamp = self._get_current_timestamp()
        url = self.base_url + path
        headers = {}
        json_payload = None

        try:
            # Create signature - handle empty body case for GET requests specifically
            message_to_sign = f"{self.api_key}{timestamp}{path}{method}"
            if body:
                 message_to_sign += body

            signed = self.private_key.sign(message_to_sign.encode("utf-8"))

            headers = {
                "x-api-key": self.api_key,
                "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
                "x-timestamp": str(timestamp),
                "Content-Type": "application/json; charset=utf-8" # Standard header
            }

            logger.debug(f"Making {method} request to {url}")
            logger.debug(f"Headers: {headers}") # Be careful logging headers in production if sensitive
            if body:
                 logger.debug(f"Body: {body}")
                 json_payload = json.loads(body) # Parse body for POST/PUT etc.

            response = requests.request(method, url, headers=headers, json=json_payload, timeout=15) # Use requests.request

            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # Handle potential empty response body for success codes like 204 No Content
            if response.status_code == 204:
                 logger.info(f"Request successful ({response.status_code}), no content returned.")
                 return None
            elif response.content:
                 # Try to decode JSON, but handle cases where it might not be JSON
                 try:
                     json_response = response.json()
                     logger.debug(f"Response ({response.status_code}): {json_response}")
                     return json_response
                 except json.JSONDecodeError:
                     logger.warning(f"Response ({response.status_code}) is not valid JSON: {response.text}")
                     return response.text # Return raw text if not JSON
            else:
                 logger.info(f"Response ({response.status_code}) received with empty body.")
                 return None # Return None for empty success responses

        except requests.exceptions.Timeout:
            logger.error(f"Request timed out: {method} {url}")
            return {"error": "Request timed out"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making API request to {url}: {e}")
            if e.response is not None:
                 logger.error(f"Response status: {e.response.status_code}")
                 try:
                    # Attempt to parse error details if response body exists and is JSON
                    error_details = e.response.json()
                    logger.error(f"Error details: {error_details}")
                    return {"error": "API request failed", "status_code": e.response.status_code, "details": error_details}
                 except json.JSONDecodeError:
                    logger.error(f"Error response body (non-JSON): {e.response.text}")
                    return {"error": "API request failed", "status_code": e.response.status_code, "details": e.response.text}
            return {"error": f"API request failed: {e}", "status_code": None, "details": None} # No response received
        except Exception as e:
             logger.error(f"An unexpected error occurred during API request to {url}: {e}", exc_info=True)
             return {"error": f"An unexpected error occurred: {e}"}


    # --- Wrapper Methods for API Endpoints ---

    def get_account(self) -> Any:
        """Fetch crypto trading account details."""
        path = "/api/v1/crypto/trading/accounts/"
        logger.info(f"Fetching account details from {path}")
        return self.make_api_request("GET", path)

    def get_trading_pairs(self, *symbols: Optional[str]) -> Any:
        """Fetch details for specific trading pairs or all if none specified."""
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/trading/trading_pairs/{query_params}"
        logger.info(f"Fetching trading pairs from {path}")
        return self.make_api_request("GET", path)

    def get_holdings(self, *asset_codes: Optional[str]) -> Any:
        """Fetch crypto holdings for specific assets or all if none specified."""
        query_params = self.get_query_params("asset_code", *asset_codes)
        path = f"/api/v1/crypto/trading/holdings/{query_params}"
        logger.info(f"Fetching holdings from {path}")
        return self.make_api_request("GET", path)

    def get_best_bid_ask(self, *symbols: Optional[str]) -> Any:
        """Fetch the best bid/ask price for specific symbols or all if none specified."""
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/marketdata/best_bid_ask/{query_params}"
        logger.info(f"Fetching best bid/ask from {path}")
        return self.make_api_request("GET", path)

    def get_estimated_price(self, symbol: str, side: str, quantity: str) -> Any:
        """Get estimated execution price for a given quantity."""
        # Ensure quantity is treated as a string list for the query param formatting
        quantities_str = quantity if isinstance(quantity, str) else ",".join(map(str, quantity))
        path = f"/api/v1/crypto/marketdata/estimated_price/?symbol={symbol}&side={side}&quantity={quantities_str}"
        logger.info(f"Fetching estimated price from {path}")
        return self.make_api_request("GET", path)

    def place_order(
            self,
            client_order_id: str,
            side: str, # 'buy' or 'sell'
            order_type: str, # 'market', 'limit', 'stop_loss', 'stop_limit'
            symbol: str, # e.g., 'BTC'
            order_config: Dict[str, Any], # Specific config for the order type
    ) -> Any:
        """Place a new crypto order."""
        # Validate inputs (basic)
        if side not in ['buy', 'sell']:
             logger.error(f"Invalid order side: {side}")
             return {"error": "Invalid order side"}
        if order_type not in ['market', 'limit', 'stop_loss', 'stop_limit']:
             logger.error(f"Invalid order type: {order_type}")
             return {"error": "Invalid order type"}

        body_dict = {
            "client_order_id": client_order_id,
            "side": side,
            "type": order_type,
            "symbol": symbol,
            f"{order_type}_order_config": order_config, # Dynamic key based on order type
        }
        path = "/api/v1/crypto/trading/orders/"
        body_json = json.dumps(body_dict, separators=(',', ':')) # Compact JSON
        logger.info(f"Placing {order_type} {side} order for {symbol} with client ID {client_order_id}")
        return self.make_api_request("POST", path, body_json)

    def cancel_order(self, order_id: str) -> Any:
        """Cancel an existing open order by its Robinhood Order ID."""
        path = f"/api/v1/crypto/trading/orders/{order_id}/cancel/"
        logger.info(f"Attempting to cancel order {order_id} at {path}")
        # Cancel requests typically don't have a request body
        return self.make_api_request("POST", path)

    def get_order(self, order_id: str) -> Any:
        """Fetch details of a specific order by its Robinhood Order ID."""
        path = f"/api/v1/crypto/trading/orders/{order_id}/"
        logger.info(f"Fetching order details for {order_id} from {path}")
        return self.make_api_request("GET", path)

    def get_orders(self, **kwargs) -> Any:
        """Fetch a list of orders, optionally filtering by query parameters."""
        # Example kwargs: state='filled', symbol='BTC', limit=50
        query_params_list = [f"{key}={value}" for key, value in kwargs.items() if value is not None]
        query_string = "?" + "&".join(query_params_list) if query_params_list else ""
        path = f"/api/v1/crypto/trading/orders/{query_string}"
        logger.info(f"Fetching orders from {path}")
        return self.make_api_request("GET", path)

    def convert_symbol_for_trading(self, symbol: str) -> str:
        """
        Convert symbol for Robinhood trading execution
        
        Args:
            symbol (str): Input symbol (e.g., 'BTC-USD')
        
        Returns:
            str: Symbol formatted for Robinhood trading (e.g., 'BTC')
        """
        # Strip any currency pairs and return base asset
        return symbol.split('-')[0]

    def convert_symbol_for_analysis(self, symbol: str) -> str:
        """
        Convert symbol for technical analysis
        
        Args:
            symbol (str): Input symbol (e.g., 'BTC')
        
        Returns:
            str: Symbol formatted for data analysis (e.g., 'BTC-USD')
        """
        return f"{symbol}-USD" if '-' not in symbol else symbol

# Example usage (optional, can be removed or kept for testing)
# def main():
#     try:
#         api_trading_client = CryptoAPITrading()
#         print("--- Account Info ---")
#         print(api_trading_client.get_account())
#         print("\n--- BTC Holdings ---")
#         print(api_trading_client.get_holdings("BTC"))
#         print("\n--- BTC Best Bid/Ask ---")
#         print(api_trading_client.get_best_bid_ask("BTC"))
#     except ValueError as e:
#          print(f"Initialization failed: {e}")
#     except Exception as e:
#          print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()
