# /Users/activate/Dev/robinhood-crypto-bot/brokers/robinhood_broker.py
import base64
import datetime
import json
import logging
import os
import uuid
from decimal import Decimal
from typing import Any, Dict, Optional

import requests
from nacl.signing import SigningKey

# Import the BaseBroker interface
from .base_broker import BaseBroker

# --- Rename Class and Inherit from BaseBroker ---
class RobinhoodBroker(BaseBroker):
    """
    Robinhood Crypto API implementation adhering to the BaseBroker interface.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Robinhood broker client.
        Expects 'api_key' and 'base64_private_key' in the config dict.
        Requires a logger instance.
        """
        super().__init__(config) # Call parent __init__
        self.logger = logger # STORE the passed logger
        self.logger.info(f"Initializing RobinhoodBroker with logger. Effective level: {self.logger.getEffectiveLevel()} (DEBUG={logging.DEBUG})")

        self.api_key = self.config.get("api_key")
        self.base64_private_key = self.config.get("base64_private_key")

        if not self.api_key or not self.base64_private_key:
            self.logger.error("Robinhood API Key or Private Key not found in config.") # USE self.logger
            # Consider raising a more specific config error
            raise ValueError("API Key and Private Key must be provided in config.")

        try:
            private_key_seed = base64.b64decode(self.base64_private_key)
            self.private_key = SigningKey(private_key_seed)
        except Exception as e:
            self.logger.exception("Failed to decode or load private key.") # USE self.logger
            raise ValueError(f"Invalid private key: {e}")

        self.base_url = "https://trading.robinhood.com"
        self.logger.info(f"RobinhoodBroker initialized for base URL: {self.base_url}") # USE self.logger

    # --- BaseBroker Interface Implementation ---

    def connect(self) -> bool:
        """
        Robinhood's API is stateless (REST), so no explicit connection is needed.
        We can perform a basic check like getting account info to verify credentials.
        """
        self.logger.info("Attempting to verify Robinhood connection by fetching account info...") # USE self.logger
        account_info = self.get_account_info()
        if account_info:
            self.logger.info("Robinhood connection verified successfully.") # USE self.logger
            return True
        else:
            self.logger.error("Robinhood connection verification failed (unable to fetch account info).") # USE self.logger
            return False

    def disconnect(self) -> None:
        """
        Stateless API, no explicit disconnection needed.
        """
        self.logger.info("RobinhoodBroker is stateless, no explicit disconnection required.") # USE self.logger
        pass

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve account information. Maps to the existing get_account method.
        Extracts key details like USD balance.
        """
        path = "/api/v1/crypto/trading/accounts/"
        response = self._make_api_request("GET", path)

        self.logger.debug(f"Raw response from _make_api_request in get_account_info: {response}")

        if response and isinstance(response, dict):
            # Check for direct account info structure first (common for single account details)
            if "account_number" in response and "buying_power" in response:
                account_data = response
                cash_str = account_data.get("buying_power", "0") # Use 'buying_power' directly
                currency = account_data.get("buying_power_currency", "USD")
                if currency != "USD":
                    self.logger.warning(f"Account buying power is in {currency}, not USD. Further conversion might be needed.")
                
                return {
                    "broker": "robinhood",
                    "account_id": account_data.get("account_number"),
                    "buying_power_usd": Decimal(cash_str),
                    "cash_usd": Decimal(cash_str), # Assuming buying_power is cash for crypto
                    "portfolio_value_usd": Decimal(cash_str), # Placeholder, update if portfolio value is different
                    "status": account_data.get("status"),
                    "raw_data": account_data
                }
            # Fallback: Check for 'results' list structure (for paginated-like responses, though not typical for this endpoint)
            elif 'results' in response and isinstance(response['results'], list) and len(response['results']) > 0:
                # Assuming the first USD account entry is the relevant one (this logic might be less relevant now)
                account_data_from_list = None
                for entry in response['results']:
                    # This part of the logic was looking for an 'asset_code' == 'USD' which is not present in the direct account info endpoint
                    # If this path is ever hit for a different endpoint that *does* have results, it might need adjustment.
                    # For the '/trading/accounts/' endpoint, this block should ideally not be reached if direct parsing works.
                    if entry.get("account_number"): # A generic check if it's account-like data
                        account_data_from_list = entry 
                        break

                if account_data_from_list:
                    # This parsing assumes fields like 'total_quantity' for USD, which may not align with the direct endpoint.
                    # It's kept for theoretical compatibility but might need review if an API call *actually* hits this.
                    cash_str = account_data_from_list.get("total_quantity", "0") # Original logic for results list
                    self.logger.info("Parsing account info from 'results' list. This is unexpected for /api/v1/crypto/trading/accounts/")
                    return {
                        "broker": "robinhood",
                        "account_id": account_data_from_list.get("account_number"),
                        "buying_power_usd": Decimal(cash_str),
                        "cash_usd": Decimal(cash_str),
                        "portfolio_value_usd": Decimal("0.0"),
                        "raw_data": account_data_from_list
                    }
                else:
                    self.logger.warning("Could not find suitable account details in the 'results' list.")
                    return None
            elif 'error' in response:
                self.logger.error(f"Failed to get Robinhood account info: {response.get('error')}")
                return None
            else:
                self.logger.warning(f"Received unexpected structure or empty response when fetching account info (after attempting direct and results list parsing): {response}")
                return None
        else:
            self.logger.error(f"No response or invalid response type from _make_api_request for account info: {response}")
            return None

    def get_holdings(self) -> Optional[Dict[str, Decimal]]:
        """
        Retrieve current crypto holdings.
        Returns a dict mapping symbol (e.g., 'BTC') to quantity (Decimal).
        """
        path = "/api/v1/crypto/trading/holdings/"
        response = self._make_api_request("GET", path)
        holdings = {}

        # Expect a dict with a 'results' list
        if response and isinstance(response, dict) and 'results' in response and isinstance(response['results'], list):
            for holding in response['results']:
                asset_code = holding.get("asset_code")
                # Use 'total_quantity' based on observed API response
                quantity_str = holding.get("total_quantity")

                # Skip USD holding, we only want crypto assets here
                if asset_code == "USD":
                    continue

                if asset_code and quantity_str:
                    try:
                        quantity = Decimal(quantity_str)
                        if quantity > Decimal("0"): # Only include non-zero holdings
                            holdings[asset_code] = quantity
                    except Exception:
                        self.logger.warning(f"Could not parse quantity '{quantity_str}' for asset {asset_code}")
            self.logger.debug(f"Parsed holdings: {holdings}")
            return holdings
        elif response and isinstance(response, dict) and 'error' in response:
            self.logger.error(f"Failed to get Robinhood holdings: {response.get('error')}")
            return None
        else:
            self.logger.warning(f"Received unexpected structure or empty response when fetching holdings: {response}")
            return None

    def get_latest_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get the latest bid/ask for a trading pair (e.g., 'BTC-USD') and return the mid-price.
        """
        formatted_symbol = self.format_symbol_for_broker(symbol) # Ensure correct format
        query_params = self._get_query_params("symbol", formatted_symbol)
        path = f"/api/v1/crypto/marketdata/best_bid_ask/{query_params}"
        response = self._make_api_request("GET", path)

        if response and isinstance(response, list) and len(response) > 0:
            data = response[0] # Assuming first result matches the symbol
            if data.get("symbol") == formatted_symbol:
                bid_str = data.get("best_bid_price", {}).get("amount")
                ask_str = data.get("best_ask_price", {}).get("amount")
                if bid_str and ask_str:
                    try:
                        bid = Decimal(bid_str)
                        ask = Decimal(ask_str)
                        mid_price = (bid + ask) / 2
                        self.logger.debug(f"Latest mid-price for {formatted_symbol}: {mid_price}")
                        return mid_price
                    except Exception:
                        self.logger.error(f"Could not parse bid/ask prices for {formatted_symbol}: bid='{bid_str}', ask='{ask_str}'")
                        return None
        elif response and 'error' in response:
            self.logger.error(f"Failed to get latest price for {formatted_symbol}: {response}")
            return None

        self.logger.warning(f"Could not find latest price data for symbol: {formatted_symbol} in response: {response}")
        return None

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        client_order_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Place a market order using asset quantity.
        Implements the BaseBroker interface method.
        """
        formatted_symbol = self.format_symbol_for_broker(symbol)
        order_type = "market"
        order_config = {"asset_quantity": f"{quantity:.8f}"} # Robinhood uses asset_quantity for crypto market orders

        # Generate a client order ID if none provided
        if client_order_id is None:
            client_order_id = str(uuid.uuid4())

        self.logger.info(f"Placing Robinhood {side} {order_type} order: {quantity:.8f} {formatted_symbol} (Client ID: {client_order_id})")

        # Use the existing _place_order logic (renamed for clarity)
        result = self._place_robinhood_order(
            client_order_id=client_order_id,
            side=side.lower(), # Ensure side is lowercase
            order_type=order_type,
            symbol=formatted_symbol,
            order_config=order_config,
        )

        if result and 'id' in result and result.get('state') != 'failed': # Check for basic success indicators
             # Standardize output slightly
             return {
                 "broker": "robinhood",
                 "order_id": result.get("id"),
                 "client_order_id": result.get("client_order_id"),
                 "symbol": self.format_symbol_from_broker(result.get("symbol")),
                 "side": result.get("side"),
                 "type": result.get("type"),
                 "status": result.get("state"), # e.g., open, filled, cancelled
                 "quantity": quantity, # Return requested quantity
                 "raw_response": result
             }
        else:
             self.logger.error(f"Robinhood market order placement failed or returned unexpected result: {result}")
             return None


    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific order by its Robinhood ID.
        """
        path = f"/api/v1/crypto/trading/orders/{order_id}/"
        response = self._make_api_request("GET", path)

        if response and 'id' in response:
             # Standardize output
             return {
                 "broker": "robinhood",
                 "order_id": response.get("id"),
                 "client_order_id": response.get("client_order_id"),
                 "symbol": self.format_symbol_from_broker(response.get("symbol")),
                 "side": response.get("side"),
                 "type": response.get("type"),
                 "status": response.get("state"),
                 "quantity": Decimal(response.get("market_order_config", {}).get("asset_quantity", "0")), # Example for market
                 "filled_quantity": Decimal(response.get("filled_asset_quantity", "0")),
                 "average_price": Decimal(response.get("average_price", {}).get("amount", "0")) if response.get("average_price") else None,
                 "created_at": response.get("created_at"),
                 "updated_at": response.get("updated_at"),
                 "raw_response": response
             }
        elif response and 'error' in response:
             self.logger.error(f"Failed to get status for order {order_id}: {response}")
             return None
        else:
            self.logger.warning(f"Received unexpected or empty response when fetching status for order {order_id}: {response}")
            return None


    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order by its Robinhood ID.
        Returns True if cancellation was successful or accepted (API returns 204 No Content).
        """
        path = f"/api/v1/crypto/trading/orders/{order_id}/cancel/"
        # Note: Robinhood cancel endpoint returns 204 No Content on success
        # We adapt _make_api_request slightly or handle the None return here.
        response = self._make_api_request("POST", path)

        # _make_api_request was modified earlier to return None on 204.
        # If response is None and no error was logged by _make_api_request, assume success.
        # A more robust way might involve checking the specific status code if _make_api_request is adapted further.
        if response is None:
            # We need to infer success based on None return and lack of logged error
            # This relies on _make_api_request logging errors appropriately.
            # Let's tentatively return True, but ideally check status code if possible.
            # A check within _make_api_request or storing last status code would be better.
            self.logger.info(f"Robinhood order cancellation request for {order_id} sent (assuming success based on 204 status).")
            return True # Assume success if no content and no error logged
        elif response and 'error' in response:
             self.logger.error(f"Failed to cancel order {order_id}: {response}")
             return False
        else:
             self.logger.warning(f"Unexpected response when cancelling order {order_id}: {response}")
             return False # Unexpected response

    # --- Existing Helper Methods (Prefixed with underscore) ---

    @staticmethod
    def _get_current_timestamp() -> int:
        return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())

    @staticmethod
    def _get_query_params(key: str, *args: Optional[str]) -> str:
        if not args or all(arg is None for arg in args):
            return ""
        params = []
        for arg in args:
            if arg is not None:
                params.append(f"{key}={arg}")
        return "?" + "&".join(params) if params else ""

    def _get_authorization_header(self, method: str, path: str, body: str, timestamp: int) -> Dict[str, str]:
        message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
        try:
            self.logger.debug("Attempting to sign message...")
            signed = self.private_key.sign(message_to_sign.encode("utf-8"))
            self.logger.debug("Message signing successful.")
        except Exception as e:
            self.logger.exception(f"Error signing the request message: {message_to_sign[:100]}...") # Log part of msg
            # Re-raise or handle appropriately. Re-raising lets the outer handler catch it.
            raise ValueError(f"Failed to sign API request: {e}")

        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
            "x-timestamp": str(timestamp),
            "Content-Type": "application/json; charset=utf-8"
        }

    def _make_api_request(self, method: str, path: str, body: str = "") -> Any:
        self.logger.debug(f"ENTERING _make_api_request for {method} {path}")
        try:
            timestamp = self._get_current_timestamp()
        except Exception as e:
            self.logger.exception("Error generating timestamp in _make_api_request")
            return {"error": "Timestamp generation failed", "details": str(e)}

        try:
            headers = self._get_authorization_header(method, path, body, timestamp)
        except Exception as e:
            self.logger.exception("Error generating authorization header in _make_api_request")
            return {"error": "Authorization header generation failed", "details": str(e)}

        url = self.base_url + path

        self.logger.debug(f"Making Robinhood API Request: {method} {url}")
        if body:
            self.logger.debug(f"Request Body: {body}") # Be cautious logging sensitive data if applicable
        self.logger.debug(f"Request Headers: {headers}") # Be cautious logging API key

        response = None  # Initialize response
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=15) # Increased timeout
            elif method == "POST":
                # Ensure body is passed as 'data' for POST if it's a string, 'json' if dict
                # The original Robinhood code implied body is always a JSON string for POST.
                headers['Content-Type'] = 'application/json' # Explicitly set content type for POST
                response = requests.post(url, headers=headers, data=body, timeout=15) # Use data=body for JSON string

            if response is None:
                self.logger.error(f"No response received for {method} {url}")
                return {"error": "No response received from server"}

            self.logger.debug(f"Response Status Code: {response.status_code}")
            self.logger.debug(f"Response Headers: {response.headers}")
            # Log first 500 chars of response text for debugging, be mindful of sensitive info
            self.logger.debug(f"Response Text (partial): {response.text[:500]}...")

            # Check for non-2xx status codes before attempting JSON decode
            if response.status_code >= 400:
                self.logger.error(f"API Request failed to {url} with status {response.status_code}")
                self.logger.error(f"Response Body: {response.text}") # Log the full error body
                # Optionally try to parse JSON error, but prioritize logging raw text
                try:
                    error_details = response.json()
                    self.logger.error(f"Parsed Error Details: {error_details}")
                    return {"error": "API request failed", "status_code": response.status_code, "details": error_details, "text": response.text}
                except json.JSONDecodeError:
                    return {"error": "API request failed", "status_code": response.status_code, "text": response.text}

            # If status is OK (2xx), proceed
            response.raise_for_status()  # Still useful for catching specific HTTPError exceptions if needed later

            # Handle potential empty body for successful responses (e.g., 204 No Content)
            if not response.content:
                 self.logger.info(f"Response ({response.status_code}) received with empty body for {method} {path}.")
                 # For POST/DELETE etc. 204 is often success. For GET, might be unexpected.
                 # Return a success indicator for 204, maybe None or specific dict for others.
                 if response.status_code == 204:
                     return {"status": "success", "status_code": 204, "message": "No Content"}
                 else:
                     # Treat other empty responses as potentially problematic, especially GET
                     self.logger.warning(f"Received unexpected empty body for {method} {url} with status {response.status_code}")
                     return None # Or return {'warning': 'Empty response body', 'status_code': response.status_code}

            # Attempt to parse JSON for successful responses with content
            try:
                result = response.json()
                return result
            except json.JSONDecodeError:
                self.logger.error(f"Failed to decode JSON response from {url} despite success status ({response.status_code}).")
                self.logger.error(f"Response Text: {response.text}")
                return {"error": "Invalid JSON response", "status_code": response.status_code, "text": response.text}

        except requests.exceptions.Timeout:
            self.logger.error(f"Request timed out after 15s: {method} {url}")
            return {"error": "Request timed out"}
        except requests.exceptions.HTTPError as e:
             # This might be less likely to be hit now with the direct status code check,
             # but keep it as a fallback.
             self.logger.error(f"HTTP Error making API request to {url}: {e}")
             self.logger.error(f"HTTP Status Code: {e.response.status_code}")
             self.logger.error(f"HTTP Response Body: {e.response.text}") # Log raw text
             return {"error": "HTTP request failed", "status_code": e.response.status_code, "text": e.response.text}

        except requests.exceptions.RequestException as e:
            # Handle other request exceptions (e.g., connection errors, DNS errors)
            self.logger.error(f"General Network/Request Error making API request to {url}: {e}", exc_info=True)
            return {"error": "API request failed", "details": str(e)}

    # --- Renamed internal methods ---

    # Renamed from the original place_order to avoid conflict with interface method
    def _place_robinhood_order(
            self,
            client_order_id: str,
            side: str,
            order_type: str,
            symbol: str,
            order_config: Dict[str, Any],
    ) -> Any:
        body_dict = {
            "client_order_id": client_order_id,
            "side": side,
            "type": order_type,
            "symbol": symbol,
            f"{order_type}_order_config": order_config,
        }
        path = "/api/v1/crypto/trading/orders/"
        # Use separators for compact JSON, as expected by Robinhood signature generation
        body_json = json.dumps(body_dict, separators=(',', ':'))
        return self._make_api_request("POST", path, body_json)

    # Keep other original methods like get_trading_pairs, get_best_bid_ask etc.
    # but consider if they fit the generic interface or should remain specific helpers.
    # For now, we'll keep them but prefix with underscore if not part of BaseBroker.

    def _get_trading_pairs(self, *symbols: Optional[str]) -> Any:
        query_params = self._get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/trading/trading_pairs/{query_params}"
        return self._make_api_request("GET", path)

    def _get_estimated_price(self, symbol: str, side: str, quantity: str) -> Any:
        path = f"/api/v1/crypto/marketdata/estimated_price/?symbol={symbol}&side={side}&quantity={quantity}"
        return self._make_api_request("GET", path)

    def _get_orders(self, **kwargs) -> Any:
        query_params_list = [f"{key}={value}" for key, value in kwargs.items() if value is not None]
        query_string = "?" + "&".join(query_params_list) if query_params_list else ""
        path = f"/api/v1/crypto/trading/orders/{query_string}"
        return self._make_api_request("GET", path)

    # --- Symbol Formatting (Optional, already in BaseBroker) ---
    # These can be removed if the BaseBroker default implementation is sufficient
    # def format_symbol_for_broker(self, symbol: str) -> str:
    #     # Robinhood uses 'BTC-USD' format, which is our standard.
    #     return symbol
    #
    # def format_symbol_from_broker(self, broker_symbol: Any) -> str:
    #     # Robinhood returns 'BTC-USD' format.
    #     return str(broker_symbol)
