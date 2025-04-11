import base64
import datetime
import json
from typing import Any, Dict, Optional
import uuid
import requests
from nacl.signing import SigningKey
from nacl.exceptions import BadSignatureError # For signature verification (optional)
import logging
from urllib.parse import urljoin

# --- Constants (Consider moving to config or loading from env) ---
# Assuming API_KEY and BASE64_PRIVATE_KEY will be passed during instantiation
ROBINHOOD_CRYPTO_API_BASE_URL = "https://trading.robinhood.com"
API_TIMEOUT_SECONDS = 15 # Increased timeout

class CryptoAPITrading:
    """Handles direct interaction with the Robinhood Crypto Trading API v1."""
    def __init__(self, api_key: str, base64_private_key: str):
        """
        Initializes the API client with credentials.

        Args:
            api_key: Your Robinhood API Key.
            base64_private_key: Your Base64 encoded private key string.
        """
        if not api_key or not base64_private_key:
            raise ValueError("API key and private key cannot be empty.")

        self.api_key = api_key
        self.base_url = ROBINHOOD_CRYPTO_API_BASE_URL
        self.private_key = None # Initialize

        try:
            private_key_seed = base64.b64decode(base64_private_key)
            self.private_key = SigningKey(private_key_seed)
            logging.info("CryptoAPITrading: Private key loaded successfully.")
        except (base64.binascii.Error, ValueError) as e:
            logging.error(f"CryptoAPITrading: Failed to decode Base64 private key: {e}", exc_info=True)
            # Raise a specific error or handle as needed, preventing initialization with invalid key
            raise ValueError("Invalid Base64 private key provided.") from e
        except ImportError:
            logging.error("CryptoAPITrading: PyNaCl library not found. Please install it (`pip install pynacl`).")
            raise ImportError("PyNaCl library is required for API signing.")

    @staticmethod
    def _get_current_timestamp() -> int:
        """Gets the current UTC timestamp as an integer."""
        return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())

    @staticmethod
    def get_query_params(key: str, *args: Optional[str]) -> str:
        """Formats query parameters for a URL."""
        valid_args = [str(arg) for arg in args if arg is not None and str(arg).strip()]
        if not valid_args:
            return ""

        params = [f"{key}={arg}" for arg in valid_args]
        return "?" + "&".join(params)

    def get_authorization_header(
            self, method: str, path: str, body: Any, timestamp: int
    ) -> Dict[str, str]:
        """Generates the required authorization headers for an API request."""
        # Ensure body is a string for signing, compact JSON if it's a dict
        body_str = ""
        if isinstance(body, dict):
             # Compact JSON encoding without extra whitespace, ensure keys are sorted for consistency
             body_str = json.dumps(body, separators=(',', ':'), sort_keys=True)
        elif isinstance(body, str):
             body_str = body # Assume string is already formatted if needed

        message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body_str}"
        logging.debug(f"Message to sign: {message_to_sign}")

        if not self.private_key:
             logging.error("Authorization header generation failed: Private key not initialized.")
             raise RuntimeError("Private key is not available for signing.") # Or handle appropriately

        try:
            signed = self.private_key.sign(message_to_sign.encode("utf-8"))
            signature_b64 = base64.b64encode(signed.signature).decode("utf-8")
            logging.debug(f"Generated Signature (Base64): {signature_b64}")

            return {
                "x-api-key": self.api_key,
                "x-signature": signature_b64,
                "x-timestamp": str(timestamp),
                "Content-Type": "application/json" # Required for POST/PUT with JSON body
            }
        except Exception as e:
            logging.error(f"Error generating signature: {e}", exc_info=True)
            raise # Re-raise the exception after logging

    def make_api_request(self, method: str, path: str, body: Any = None) -> Optional[Dict[str, Any]]:
        """
        Makes a generic API request to the specified path.

        Args:
            method: HTTP method (GET, POST, etc.). Case-insensitive.
            path: API endpoint path (e.g., /api/v1/crypto/trading/accounts/).
            body: Request body, typically a dict for POST/PUT (will be JSON serialized).
                  For GET/DELETE, this is usually None.

        Returns:
            The JSON response as a dictionary, or None if an error occurred.
        """
        method = method.upper() # Normalize method
        timestamp = self._get_current_timestamp()

        # Body processing: Convert dict to compact JSON string for POST/PUT if needed
        request_body_for_headers = body if method in ["POST", "PUT"] else "" # Use body for signing POST/PUT
        request_body_for_request = body # Keep as dict for requests library's json parameter

        try:
            headers = self.get_authorization_header(method, path, request_body_for_headers, timestamp)
        except Exception as e: # Catch potential errors from header generation
            logging.error(f"Failed to generate authorization header: {e}")
            return None

        url = urljoin(self.base_url, path)

        try:
            response = None
            logging.debug(f"Making API Request: {method} {url}")
            # logging.debug(f"Request Headers: {headers}") # Sensitive, maybe log keys only
            if method in ["POST", "PUT"] and request_body_for_request:
                logging.debug(f"Request Body: {json.dumps(request_body_for_request)}") # Log JSON body

            request_params = {
                "headers": headers,
                "timeout": API_TIMEOUT_SECONDS
            }
            if method in ["POST", "PUT"] and request_body_for_request:
                 request_params["json"] = request_body_for_request # Use json param for dicts
            elif method == "POST" and not request_body_for_request:
                 # Handle POST requests that might not have a body but still need Content-Type
                 request_params["data"] = None # Send empty body explicitly if needed

            response = requests.request(method, url, **request_params)

            logging.debug(f"API Response Status: {response.status_code}")
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            response_json = response.json()
            # logging.debug(f"API Response JSON: {response_json}") # Can be verbose
            return response_json

        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err} - Status: {response.status_code}")
            try:
                 error_content = response.json() # Try to parse JSON error details
                 logging.error(f"Error Response Body: {error_content}")
            except json.JSONDecodeError:
                 logging.error(f"Non-JSON Error Response Body: {response.text}")
            except Exception as e:
                 logging.error(f"Could not log detailed error response: {e}")
            return None
        except requests.exceptions.ConnectionError as conn_err:
            logging.error(f"Connection error occurred: {conn_err}")
            return None
        except requests.exceptions.Timeout as timeout_err:
            logging.error(f"Timeout error occurred: {timeout_err}")
            return None
        except requests.exceptions.RequestException as req_err:
            logging.error(f"An unexpected error occurred with the request: {req_err}")
            return None
        except json.JSONDecodeError as json_err:
             logging.error(f"Failed to decode API JSON response: {json_err}. Status: {response.status_code}. Response text: {response.text if response else 'N/A'}")
             return None
        except Exception as e:
            logging.error(f"An unexpected error occurred during API request: {e}", exc_info=True)
            return None

    # --- Specific API Endpoint Methods ---

    def get_account(self) -> Optional[Dict[str, Any]]:
        """Fetches crypto trading account details."""
        path = "/api/v1/crypto/trading/accounts/"
        return self.make_api_request("GET", path)

    def get_trading_pairs(self, *symbols: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Fetches details for specific trading pairs (e.g., BTC-USD).
        If no symbols are provided, fetches all supported pairs.
        """
        query_params = self.get_query_params("symbol", *symbols)
        # Path construction needs care: query params should start with '?' if present
        base_path = "/api/v1/crypto/trading/trading_pairs/"
        path = f"{base_path}{query_params}" if query_params else base_path
        logging.debug(f"Requesting trading pairs with path: {path}")
        return self.make_api_request("GET", path)

    def get_holdings(self, *asset_codes: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Fetches crypto holdings for specific asset codes (e.g., BTC, ETH).
        If no asset codes provided, fetches all holdings.
        """
        query_params = self.get_query_params("asset_code", *asset_codes)
        base_path = "/api/v1/crypto/trading/holdings/"
        path = f"{base_path}{query_params}" if query_params else base_path
        logging.debug(f"Requesting holdings with path: {path}")
        return self.make_api_request("GET", path)

    def get_best_bid_ask(self, *symbols: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Fetches the best bid and ask for specific trading pairs.
        If no symbols provided, fetches for all supported pairs.
        """
        query_params = self.get_query_params("symbol", *symbols)
        base_path = "/api/v1/crypto/marketdata/best_bid_ask/"
        path = f"{base_path}{query_params}" if query_params else base_path
        logging.debug(f"Requesting best bid/ask with path: {path}")
        return self.make_api_request("GET", path)

    def get_estimated_price(self, symbol: str, side: str, quantity: str) -> Optional[Dict[str, Any]]:
        """
        Gets an estimated execution price for a potential trade.

        Args:
            symbol: Trading pair (e.g., "BTC-USD").
            side: "bid" (for selling), "ask" (for buying).
            quantity: The amount of crypto asset (e.g., "0.01"). Can be comma-separated for multiple estimates.
        """
        # Basic validation
        if side.lower() not in ["bid", "ask"]:
             logging.error(f"Invalid side '{side}' for get_estimated_price. Must be 'bid' or 'ask'.")
             return None
        if not symbol or not quantity:
             logging.error("Symbol and quantity are required for get_estimated_price.")
             return None

        # Parameters must be passed as query params
        path = f"/api/v1/crypto/marketdata/estimated_price/?symbol={symbol}&side={side.lower()}&quantity={quantity}"
        logging.debug(f"Requesting estimated price with path: {path}")
        return self.make_api_request("GET", path)

    def place_order(
            self,
            client_order_id: str,
            side: str,
            order_type: str,
            symbol: str,
            order_config: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """
        Places a crypto order.

        Args:
            client_order_id: A unique ID for the order (UUID recommended).
            side: "buy" or "sell".
            order_type: "market" or "limit".
            symbol: Trading pair (e.g., "BTC-USD").
            order_config: Dictionary containing order specifics based on order_type.
                          For market: {"spend": "USD amount"} or {"unit_count": "crypto amount"}
                          For limit: {"limit_price": "price", "unit_count": "crypto amount"}
        """
        # Basic validation
        if side.lower() not in ["buy", "sell"]:
             logging.error(f"Invalid side '{side}' for place_order. Must be 'buy' or 'sell'.")
             return None
        if order_type.lower() not in ["market", "limit"]:
             logging.error(f"Invalid order_type '{order_type}' for place_order. Must be 'market' or 'limit'.")
             return None
        if not symbol:
             logging.error("Symbol is required for place_order.")
             return None

        # Ensure client_order_id is a string (UUID converted if necessary)
        if isinstance(client_order_id, uuid.UUID):
            client_order_id = str(client_order_id)

        body = {
            "client_order_id": client_order_id,
            "side": side.lower(),
            "type": order_type.lower(),
            "symbol": symbol,
            f"{order_type.lower()}_order_config": order_config,
        }
        path = "/api/v1/crypto/trading/orders/"
        logging.info(f"Placing {side} {order_type} order for {symbol}. Client ID: {client_order_id}")
        logging.debug(f"Order body: {json.dumps(body)}")
        return self.make_api_request("POST", path, body=body)

    def cancel_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Cancels a specific crypto order by its Robinhood-assigned ID."""
        if not order_id:
             logging.error("Order ID is required for cancel_order.")
             return None
        path = f"/api/v1/crypto/trading/orders/{order_id}/cancel/"
        logging.info(f"Attempting to cancel order: {order_id}")
        # Cancel usually requires an empty POST body
        return self.make_api_request("POST", path, body=None) # Explicitly None body for POST

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Fetches details for a specific crypto order by its Robinhood-assigned ID."""
        if not order_id:
             logging.error("Order ID is required for get_order.")
             return None
        path = f"/api/v1/crypto/trading/orders/{order_id}/"
        logging.debug(f"Fetching order details for: {order_id}")
        return self.make_api_request("GET", path)

    def get_orders(self, *order_ids: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Fetches details for multiple crypto orders by their Robinhood-assigned IDs.
        If no order IDs are provided, fetches recent orders (behavior might vary).
        """
        query_params = self.get_query_params("order_id", *order_ids)
        base_path = "/api/v1/crypto/trading/orders/"
        path = f"{base_path}{query_params}" if query_params else base_path
        logging.debug(f"Fetching orders with path: {path}")
        return self.make_api_request("GET", path)

    # --- Additional Helper Methods (Optional) ---

    def generate_client_order_id(self) -> str:
        """Generates a unique client order ID (UUID4)."""
        return str(uuid.uuid4())

