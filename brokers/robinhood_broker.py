# /Users/activate/Dev/robinhood-crypto-bot/brokers/robinhood_broker.py
import base64
import datetime
import json
import logging
import os
import uuid
from decimal import Decimal
from typing import Any, Dict, Optional, List

import requests
from nacl.signing import SigningKey

# Import the BaseBroker interface
from .base_broker import BaseBroker
# Import CryptoAPITrading from project root (assuming project root in sys.path)
from robinhood_api_trading import CryptoAPITrading

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
        self.logger = logger 
        self.logger.info(f"Initializing RobinhoodBroker with logger. Effective level: {self.logger.getEffectiveLevel()} (DEBUG={logging.DEBUG})")

        api_key_val = self.config.get("api_key")
        base64_private_key_val = self.config.get("base64_private_key")

        if not api_key_val or not base64_private_key_val:
            self.logger.error("Robinhood API Key or Private Key not found in config.")
            raise ValueError("API Key and Private Key must be provided in config.")

        try:
            # Instantiate CryptoAPITrading for API interactions
            self.trader = CryptoAPITrading(
                api_key=api_key_val,
                base64_private_key=base64_private_key_val
            )
        except Exception as e:
            self.logger.exception("Failed to initialize CryptoAPITrading instance.")
            raise ValueError(f"Error initializing CryptoAPITrading: {e}")

        self.logger.info(f"RobinhoodBroker initialized using CryptoAPITrading.")

    # --- BaseBroker Interface Implementation ---

    def connect(self) -> bool:
        """
        Robinhood's API is stateless (REST), so no explicit connection is needed.
        We can perform a basic check like getting account info to verify credentials.
        """
        self.logger.info("Attempting to verify Robinhood connection by fetching account info...")
        account_details = self.get_account_info() 
        if account_details:
            # Minimal check: if we get any response (even error dict), trader is somewhat working
            if isinstance(account_details, dict) and account_details.get("error"):
                 self.logger.error(f"Robinhood connection verification failed (API error): {account_details.get('error')}")
                 return False            
            self.logger.info("Robinhood connection verified successfully.")
            return True
        else:
            self.logger.error("Robinhood connection verification failed (unable to fetch account info).")
            return False

    def disconnect(self) -> None:
        """
        Stateless API, no explicit disconnection needed.
        """
        self.logger.info("RobinhoodBroker is stateless, no explicit disconnection required.")
        pass

    def get_account_info(self) -> Optional[List[Dict[str, Any]]]: 
        """
        Retrieve account information using CryptoAPITrading.
        This method implements the get_account_info from BaseBroker.
        """
        self.logger.debug("Fetching account information via self.trader.get_account() for get_account_info")
        try:
            response = self.trader.get_account()

            if response is None:
                self.logger.error("Failed to get account info: self.trader.get_account() returned None.")
                return None
            
            # Handle potential error dict from CryptoAPITrading's make_api_request exception handling
            if isinstance(response, dict) and 'error' in response:
                self.logger.error(f"Error fetching account info from API: {response.get('details', response.get('error'))}")
                return [response] # Propagate error dict as a list item for potential downstream parsing
            
            # If the response is a dictionary and contains a 'results' key, that's the list we want.
            if isinstance(response, dict) and 'results' in response:
                account_list = response.get('results')
                if isinstance(account_list, list):
                    self.logger.debug(f"Successfully fetched account info results: {account_list}")
                    return account_list
                else:
                    self.logger.error(f"Account info 'results' field is not a list: {type(account_list)}. Response: {response}")
                    return None # Or handle as an error, e.g., return [{'error': 'invalid_results_format'}]
            
            # Fallback for unexpected structures or if response was already a list (e.g., if API changes)
            if isinstance(response, list):
                self.logger.debug(f"Successfully fetched account info (already a list): {response}")
                return response
            
            self.logger.error(f"Unexpected response format for account info: {type(response)}. Response: {response}")
            return None # Or handle as an error
        except Exception as e:
            self.logger.exception(f"Exception in get_account_info: {e}")
            return None

    def get_holdings(self, asset_code: Optional[str] = None) -> Optional[List[Dict[str, Any]]]: 
        """
        Retrieve current crypto holdings for a specific asset_code or all if None.
        Uses CryptoAPITrading.
        """
        self.logger.debug(f"Fetching holdings for asset_code '{asset_code}' via self.trader.get_holdings()")
        try:
            if asset_code:
                response = self.trader.get_holdings(asset_code)
            else:
                response = self.trader.get_holdings()

            if response is None:
                self.logger.error(f"Failed to get holdings for '{asset_code}': self.trader.get_holdings() returned None.")
                return None

            # Handle potential error dict from CryptoAPITrading's make_api_request exception handling
            if isinstance(response, dict) and 'error' in response:
                self.logger.error(f"Error fetching holdings from API for '{asset_code}': {response.get('details', response.get('error'))}")
                return [response] # Propagate error dict as a list item

            # If the response is a dictionary and contains a 'results' key, that's the list we want.
            if isinstance(response, dict) and 'results' in response:
                holdings_list = response.get('results')
                if isinstance(holdings_list, list):
                    self.logger.debug(f"Successfully fetched holdings results for '{asset_code}': {holdings_list}")
                    return holdings_list
                else:
                    self.logger.error(f"Holdings 'results' field for '{asset_code}' is not a list: {type(holdings_list)}. Response: {response}")
                    return None 
                    
            # Fallback for unexpected structures or if response was already a list
            if isinstance(response, list):
                self.logger.debug(f"Successfully fetched holdings for '{asset_code}' (already a list): {response}")
                return response

            self.logger.error(f"Unexpected response format for holdings '{asset_code}': {type(response)}. Response: {response}")
            return None
        except Exception as e:
            self.logger.exception(f"Exception in get_holdings for asset_code '{asset_code}': {e}")
            return None

    def get_latest_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get the latest bid/ask for a trading pair (e.g., 'BTC-USD') and return the mid-price.
        Uses CryptoAPITrading.
        """
        formatted_symbol = self.format_symbol_for_broker(symbol) 
        self.logger.debug(f"Fetching latest price for {formatted_symbol} via self.trader.get_best_bid_ask()")
        try:
            response = self.trader.get_best_bid_ask(formatted_symbol)

            if response is None:
                self.logger.error(f"Failed to get latest price for {formatted_symbol}: self.trader.get_best_bid_ask() returned None.")
                return None
            
            if isinstance(response, dict) and 'error' in response:
                self.logger.error(f"Error fetching latest price for {formatted_symbol} from API: {response.get('details', response.get('error'))}")
                return None

            if isinstance(response, list) and len(response) > 0:
                data = response[0] # Assuming first result matches the symbol
                if data.get("symbol") == formatted_symbol:
                    bid_str = data.get("best_bid_price", {}).get("amount")
                    ask_str = data.get("best_ask_price", {}).get("amount")
                    if bid_str and ask_str:
                        try:
                            mid_price = (Decimal(bid_str) + Decimal(ask_str)) / 2
                            self.logger.info(f"Mid price for {formatted_symbol}: {mid_price}")
                            return mid_price
                        except Exception as e:
                            self.logger.error(f"Could not parse bid/ask prices for {formatted_symbol}: {bid_str}, {ask_str}. Error: {e}")
                            return None
                self.logger.warning(f"Could not find matching symbol {formatted_symbol} in best_bid_ask response: {response}")
            else:
                self.logger.warning(f"Unexpected response structure for best_bid_ask {formatted_symbol}: {response}")
            return None
        except Exception as e:
            self.logger.exception(f"Exception in get_latest_price for {symbol}: {e}")
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
        Implements the BaseBroker interface method using CryptoAPITrading.
        """
        formatted_symbol = self.format_symbol_for_broker(symbol)
        order_type = "market"
        # Robinhood's place_order via CryptoAPITrading expects order_config values as strings.
        order_config = {"asset_quantity": f"{quantity:.8f}"} 

        if client_order_id is None:
            client_order_id = str(uuid.uuid4())

        self.logger.info(f"Placing {side} {order_type} order for {quantity} {formatted_symbol} (Client ID: {client_order_id})")
        try:
            result = self.trader.place_order(
                client_order_id=client_order_id,
                side=side.lower(), 
                order_type=order_type,
                symbol=formatted_symbol,
                order_config=order_config,
            )

            if result is None:
                self.logger.error(f"Failed to place order for {formatted_symbol}: self.trader.place_order() returned None.")
                return None

            if isinstance(result, dict) and 'error' in result:
                self.logger.error(f"Error placing order for {formatted_symbol} from API: {result.get('details', result.get('error'))}")
                # Propagate raw error response if needed by caller
                return {"broker": "robinhood", "error": result.get('details', result.get('error')), "raw_response": result}

            if isinstance(result, dict) and 'id' in result and result.get('state') != 'failed': 
                self.logger.info(f"Order placed successfully for {formatted_symbol}. Order ID: {result.get('id')}, State: {result.get('state')}")
                return {
                    "broker": "robinhood",
                    "order_id": result.get("id"),
                    "client_order_id": result.get("client_order_id", client_order_id),
                    "symbol": self.format_symbol_from_broker(result.get("symbol")),
                    "side": result.get("side"),
                    "type": result.get("type"),
                    "status": result.get("state"), 
                    "quantity": quantity, 
                    "raw_response": result
                }
            else:
                self.logger.error(f"Order placement for {formatted_symbol} failed or response structure unrecognized: {result}")
                return {"broker": "robinhood", "error": "Order placement failed or unrecognized response", "raw_response": result}
        except Exception as e:
            self.logger.exception(f"Exception in place_market_order for {symbol}: {e}")
            return {"broker": "robinhood", "error": str(e)}

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        limit_price: Decimal,
        client_order_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Place a limit order using asset quantity and limit price.
        Implements the BaseBroker interface method using CryptoAPITrading.
        """
        formatted_symbol = self.format_symbol_for_broker(symbol)
        order_type = "limit"
        # Robinhood's place_order via CryptoAPITrading expects order_config values as strings.
        order_config = {
            "asset_quantity": f"{quantity:.8f}",
            "limit_price": f"{limit_price:.2f}" # Assuming 2 decimal places for price, adjust if needed
        }

        if client_order_id is None:
            client_order_id = str(uuid.uuid4())

        self.logger.info(f"Placing {side} {order_type} order for {quantity} {formatted_symbol} @ {limit_price} (Client ID: {client_order_id})")
        try:
            result = self.trader.place_order(
                client_order_id=client_order_id,
                side=side.lower(),
                order_type=order_type,
                symbol=formatted_symbol,
                order_config=order_config,
            )

            if result is None:
                self.logger.error(f"Failed to place limit order for {formatted_symbol}: self.trader.place_order() returned None.")
                return None

            if isinstance(result, dict) and 'error' in result:
                self.logger.error(f"Error placing limit order for {formatted_symbol} from API: {result.get('details', result.get('error'))}")
                return {"broker": "robinhood", "error": result.get('details', result.get('error')), "raw_response": result}

            if isinstance(result, dict) and 'id' in result and result.get('state') != 'failed':
                self.logger.info(f"Limit order placed successfully for {formatted_symbol}. Order ID: {result.get('id')}, State: {result.get('state')}")
                return {
                    "broker": "robinhood",
                    "order_id": result.get("id"),
                    "client_order_id": result.get("client_order_id", client_order_id),
                    "symbol": self.format_symbol_from_broker(result.get("symbol")),
                    "side": result.get("side"),
                    "type": result.get("type"),
                    "status": result.get("state"),
                    "quantity": quantity,
                    "limit_price": limit_price,
                    "raw_response": result
                }
            else:
                self.logger.error(f"Limit order placement for {formatted_symbol} failed or response structure unrecognized: {result}")
                return {"broker": "robinhood", "error": "Limit order placement failed or unrecognized response", "raw_response": result}
        except Exception as e:
            self.logger.exception(f"Exception in place_limit_order for {symbol}: {e}")
            return {"broker": "robinhood", "error": str(e)}

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific order by its Robinhood ID using CryptoAPITrading.
        """
        self.logger.info(f"Fetching status for order ID: {order_id}")
        try:
            response = self.trader.get_order(order_id)

            if response is None:
                self.logger.error(f"Failed to get order status for {order_id}: self.trader.get_order() returned None.")
                return None
            
            if isinstance(response, dict) and 'error' in response:
                self.logger.error(f"Error fetching order status for {order_id} from API: {response.get('details', response.get('error'))}")
                return {"broker": "robinhood", "order_id": order_id, "status": "error", "error_details": response, "raw_response": response}
            
            if isinstance(response, dict) and 'id' in response:
                self.logger.info(f"Order status for {order_id}: {response.get('state')}")
                return {
                    "broker": "robinhood",
                    "order_id": response.get("id"),
                    "client_order_id": response.get("client_order_id"),
                    "symbol": self.format_symbol_from_broker(response.get("symbol")),
                    "side": response.get("side"),
                    "type": response.get("type"),
                    "status": response.get("state"),
                    "quantity": Decimal(response.get("market_order_config", {}).get("asset_quantity", "0")) if response.get("market_order_config") else Decimal(response.get("limit_order_config", {}).get("asset_quantity", "0")), 
                    "filled_quantity": Decimal(response.get("filled_asset_quantity", "0")),
                    "average_price": Decimal(response.get("average_price", {}).get("amount", "0")) if response.get("average_price") else None,
                    "created_at": response.get("created_at"),
                    "updated_at": response.get("updated_at"),
                    "raw_response": response
                }
            else:
                self.logger.warning(f"Unexpected response structure for order status {order_id}: {response}")
                return {"broker": "robinhood", "order_id": order_id, "status": "unknown", "raw_response": response}
        except Exception as e:
            self.logger.exception(f"Exception in get_order_status for order ID {order_id}: {e}")
            return {"broker": "robinhood", "order_id": order_id, "status": "exception", "error_details": str(e)}

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order by its Robinhood ID using CryptoAPITrading.
        Returns True if cancellation was successful or accepted.
        """
        self.logger.info(f"Attempting to cancel order ID: {order_id}")
        try:
            response = self.trader.cancel_order(order_id)

            if response is None:
                self.logger.error(f"Failed to cancel order {order_id}: self.trader.cancel_order() returned None.")
                return False
            
            # CryptoAPITrading.cancel_order returns response.json(), which might be an empty dict on 204,
            # or an error dict.
            if isinstance(response, dict):
                if 'error' in response:
                    self.logger.error(f"Error cancelling order {order_id} from API: {response.get('details', response.get('error'))}")
                    return False
                # Robinhood API for cancel often returns 204 No Content, which requests.json() might parse as empty dict or error.
                # If no error key, and it's a dict, assume success (e.g. if API returns {} on success).
                # CryptoAPITrading's make_api_request for 204 returns {'status': 'success', 'status_code': 204, 'message': 'No Content'}
                if response.get('status') == 'success' and response.get('status_code') == 204:
                     self.logger.info(f"Order {order_id} cancelled successfully (API returned 204 No Content).")
                     return True
                # If it's just an empty dict {} without error, also consider it a success for robustness
                if not response: # Empty dict
                    self.logger.info(f"Order {order_id} cancellation request acknowledged (empty JSON response, assuming success).")
                    return True
                self.logger.warning(f"Order {order_id} cancellation - API returned a dict, but not a clear success/error: {response}")
                # Depending on strictness, this could be False. Let's be optimistic if no 'error' key.
                return True 
            
            self.logger.warning(f"Unexpected response type when cancelling order {order_id}: {type(response)}, {response}")
            return False
        except Exception as e:
            self.logger.exception(f"Exception in cancel_order for order ID {order_id}: {e}")
            return False

    # --- Renamed internal methods (now using self.trader) ---

    def _get_trading_pairs(self, *symbols: Optional[str]) -> Any:
        self.logger.debug(f"Fetching trading pairs for symbols: {symbols} via self.trader.get_trading_pairs()")
        try:
            return self.trader.get_trading_pairs(*symbols)
        except Exception as e:
            self.logger.exception(f"Exception in _get_trading_pairs: {e}")
            return {"error": "Failed to get trading pairs", "details": str(e)}

    def _get_estimated_price(self, symbol: str, side: str, quantity: str) -> Any:
        self.logger.debug(f"Fetching estimated price for {symbol} {side} {quantity} via self.trader.get_estimated_price()")
        try:
            return self.trader.get_estimated_price(symbol, side, quantity)
        except Exception as e:
            self.logger.exception(f"Exception in _get_estimated_price: {e}")
            return {"error": "Failed to get estimated price", "details": str(e)}

    def _get_orders(self, **kwargs) -> Any:
        # This method in CryptoAPITrading is get_orders_history.
        # It targets /api/v1/crypto/trading/orders/history/
        # The original _get_orders targeted /api/v1/crypto/trading/orders/
        # For now, mapping to get_orders_history. If specific non-history endpoint is needed, CryptoAPITrading might need an update.
        self.logger.debug(f"Fetching orders with params: {kwargs} via self.trader.get_orders_history()")
        try:
            return self.trader.get_orders_history(**kwargs)
        except Exception as e:
            self.logger.exception(f"Exception in _get_orders (mapped to get_orders_history): {e}")
            return {"error": "Failed to get orders history", "details": str(e)}

    # --- Symbol Formatting (Optional, already in BaseBroker) ---
    # These can be removed if the BaseBroker default implementation is sufficient
    # def format_symbol_for_broker(self, symbol: str) -> str:
    #     # Robinhood uses 'BTC-USD' format, which is our standard.
    #     return symbol
    #
    # def format_symbol_from_broker(self, broker_symbol: Any) -> str:
    #     # Robinhood returns 'BTC-USD' format.
    #     return str(broker_symbol)
