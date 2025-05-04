import base64
import datetime
import json
import logging
import os
import uuid
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, List, Tuple, Union

import pandas as pd
import requests
import yfinance as yf
from nacl.signing import SigningKey
from dotenv import load_dotenv

from config import ROBINHOOD_API_KEY, ROBINHOOD_PRIVATE_KEY # Assuming config.py exists at project root
from .base_broker import BrokerInterface, Order, Holding, MarketData # Remove Position

# Load environment variables from .env file if present
load_dotenv()

logger = logging.getLogger(__name__)

# Use variables from config or fallback to environment variables
API_KEY = ROBINHOOD_API_KEY or os.getenv("ROBINHOOD_API_KEY")
BASE64_PRIVATE_KEY = ROBINHOOD_PRIVATE_KEY or os.getenv("ROBINHOOD_PRIVATE_KEY")

class RobinhoodBroker(BrokerInterface):
    """Robinhood Crypto API implementation for the BrokerInterface."""

    def __init__(self):
        if not API_KEY or not BASE64_PRIVATE_KEY:
            logger.error("ROBINHOOD_API_KEY or ROBINHOOD_PRIVATE_KEY not found in config or environment variables.")
            raise ValueError("API Key and Private Key must be set.")

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
        self._session = requests.Session() # Use a session for potential performance benefits
        self._session.headers.update({"Content-Type": "application/json; charset=utf-8"})
        self.latest_market_data = {} # Initialize the attribute
        logger.info(f"RobinhoodBroker initialized for base URL: {self.base_url}")

    def connect(self) -> bool:
        """Check if API keys are loaded. Connection is implicit."""
        # In this REST API client, 'connection' is just the readiness to make requests.
        # The real test happens with the first API call.
        if self.api_key and self.private_key:
            logger.info("RobinhoodBroker connection ready (API keys loaded).")
            return True
        logger.error("RobinhoodBroker connection failed: API keys missing.")
        return False

    def disconnect(self) -> None:
        """Close the underlying session."""
        self._session.close()
        logger.info("RobinhoodBroker disconnected (session closed).")

    def get_account_summary(self) -> Dict[str, Any]:
        """Fetch crypto trading account details."""
        path = "/api/v1/crypto/trading/accounts/"
        logger.info(f"Fetching account summary from {path}")
        # This method returns the raw dictionary for now.
        # Parsing specific fields like capital will be done in a dedicated method.
        return self._make_api_request("GET", path)

    def get_available_capital(self) -> Optional[Decimal]:
        """Retrieve the available trading capital."""
        account_data = self.get_account_summary()
        logger.debug(f"Raw account summary response received: {account_data}") 

        # --- Corrected Parsing Logic --- 
        if account_data and isinstance(account_data, dict):
            # The 'buying_power' key is directly in the main dictionary, not nested in 'results'
            potential_keys = ['buying_power'] # Primary key based on observed response
            for key in potential_keys:
                 if key in account_data:
                     try:
                         capital_str = account_data[key]
                         capital = Decimal(capital_str)
                         logger.info(f"Successfully parsed available capital '{capital_str}' using key '{key}'")
                         return capital
                     except (InvalidOperation, TypeError, ValueError) as e:
                         logger.error(f"Error converting value '{account_data[key]}' for key '{key}' to Decimal: {e}")
            
            # If primary key not found or failed, log and fall through
            logger.error(f"Could not find a valid capital key {potential_keys} in account summary: {account_data}")
            return None # Return None if no valid key found
        else:
            logger.error(f"Failed to fetch or parse account summary for capital. Invalid structure or empty results received: {account_data}")
            return None

    def get_holdings(self, asset: Optional[str] = None) -> List[Holding]:
        """Fetch crypto holdings, optionally filtered by asset."""
        asset_code = self._convert_symbol_for_trading(asset) if asset else None
        query_params = self._get_query_params("asset_code", asset_code)
        path = f"/api/v1/crypto/trading/holdings/{query_params}"
        logger.info(f"Fetching holdings from {path}")
        response_data = self._make_api_request("GET", path)

        holdings_list = []
        if response_data and 'results' in response_data:
            for item in response_data['results']:
                try:
                    holding = Holding(
                        asset=item.get('asset_code'),
                        quantity=Decimal(item.get('quantity', '0')),
                        average_cost=Decimal(item.get('cost_basis', '0')) / Decimal(item.get('quantity', '1')) if Decimal(item.get('quantity', '0')) > 0 else Decimal('0'), # Approximate avg cost
                        market_value=Decimal(item.get('market_value', '0')) # RH provides market value
                    )
                    holdings_list.append(holding)
                except Exception as e:
                    logger.error(f"Error parsing holding item: {item}. Error: {e}", exc_info=True)
                    # Skip this item if parsing fails
        return holdings_list

    def get_holding_quantity(self, asset: str) -> Optional[Decimal]:
        """Fetch the quantity of a specific crypto asset held."""
        try:
            holdings = self.get_holdings(asset=asset)
            if holdings: # get_holdings returns a list
                # Assuming the first element is the one we asked for
                return holdings[0].quantity
            else:
                 logger.info(f"No holdings found for asset: {asset}")
                 return Decimal('0.0') # Return 0 if no holding exists
        except Exception as e:
            logger.error(f"Error fetching holding quantity for {asset}: {e}", exc_info=True)
            return None

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch the best bid/ask price for a specific symbol."""
        # trading_symbol = self._convert_symbol_for_trading(symbol) # REMOVE conversion for this endpoint
        # The best_bid_ask endpoint expects the full pair like 'BTC-USD'
        if not symbol or '-' not in symbol:
             logger.error(f"Invalid symbol format for get_market_data: '{symbol}'. Expected 'BASE-QUOTE' (e.g., 'BTC-USD').")
             return None
        
        # Use the symbol directly
        query_params = self._get_query_params("symbol", symbol) 
        path = f"/api/v1/crypto/marketdata/best_bid_ask/{query_params}"
        logger.info(f"Fetching market data for {symbol} from {path}") # Log the correct symbol
        response_data = self._make_api_request("GET", path)

        if response_data and 'results' in response_data and response_data['results']:
            # RH returns a list even for a single symbol query
            market_info = response_data['results'][0] 
            logger.debug(f"Raw market data received for {symbol}: {market_info}")
            try:
                # Extract the correct bid/ask fields from the response
                bid_price_str = market_info.get('bid_inclusive_of_sell_spread')
                ask_price_str = market_info.get('ask_inclusive_of_buy_spread')
                timestamp_str = market_info.get('timestamp') # Get timestamp
                
                if bid_price_str is None or ask_price_str is None:
                    logger.warning(f"Missing bid or ask price in market data for {symbol}: {market_info}")
                    return None

                # Parse to Decimal
                bid_price = Decimal(bid_price_str)
                ask_price = Decimal(ask_price_str)
                
                # Convert timestamp string to datetime object, truncating to microseconds
                try:
                    if '.' in timestamp_str:
                        ts_parts = timestamp_str.split('.')
                        # Truncate fractional seconds to 6 digits (microseconds)
                        fractional_seconds = ts_parts[1].replace('Z', '')[:6]
                        timestamp_str_truncated = f"{ts_parts[0]}.{fractional_seconds}+00:00"
                    else:
                        timestamp_str_truncated = timestamp_str.replace('Z', '+00:00')
                        
                    timestamp_dt = datetime.datetime.fromisoformat(timestamp_str_truncated)
                except (ValueError, TypeError, IndexError) as ts_err:
                    logger.warning(f"Could not parse timestamp '{timestamp_str}': {ts_err}")
                    timestamp_dt = datetime.datetime.now(datetime.timezone.utc) # Fallback

                # Instantiate MarketData with the correct fields
                market_data_obj = MarketData(
                    symbol=self._convert_symbol_for_analysis(market_info.get('symbol')),
                    bid=float(bid_price), 
                    ask=float(ask_price),
                    timestamp=timestamp_dt
                )
                # Store the fetched market data
                self.latest_market_data[symbol] = market_data_obj 
                return market_data_obj # Return the created object
            except (InvalidOperation, KeyError, TypeError) as e:
                logger.warning(f"Could not parse market data: {market_info}. Error: {e}")
                return None
        else:
            logger.warning(f"No market data found for symbol: {symbol}")
            return None

    def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Fetch the current market price (mid-price) for a symbol."""
        market_data = self.get_market_data(symbol)
        if market_data and market_data.bid is not None and market_data.ask is not None:
            try:
                # Use Decimal for precision, convert bid/ask (floats) back to Decimal
                mid_price = (Decimal(str(market_data.bid)) + Decimal(str(market_data.ask))) / Decimal(2)
                logger.debug(f"Calculated mid-price for {symbol}: {mid_price:.8f} (Bid: {market_data.bid}, Ask: {market_data.ask})")
                return mid_price
            except (TypeError, InvalidOperation) as e:
                 logger.warning(f"Error calculating mid-price for {symbol} from Bid={market_data.bid}, Ask={market_data.ask}: {e}")
                 return None
        else:
            logger.warning(f"Could not determine mid-price for {symbol} from market data: {market_data}")
            return None

    def place_order(self, symbol: str, side: str, order_type: str, quantity: Optional[Decimal] = None, amount: Optional[Decimal] = None, time_in_force: str = 'gtc', limit_price: Optional[Decimal] = None, stop_price: Optional[Decimal] = None) -> Optional[Order]:
        """Place a trading order."""
        trading_symbol = self._convert_symbol_for_trading(symbol)
        client_order_id = str(uuid.uuid4())
        path = "/api/v1/crypto/trading/orders/"

        # --- Construct Order Body --- 
        order_config = {}
        if order_type == 'limit':
            if limit_price is None:
                logger.error("Limit price is required for limit orders.")
                return None
            order_config['limit_price'] = str(limit_price.quantize(Decimal('0.00000001')))
            # Robinhood uses quantity for limit orders
            if quantity is None:
                logger.error("Quantity is required for limit orders.")
                return None
            order_config['quantity'] = str(quantity.quantize(Decimal('0.00000001'))) # Adjust precision as needed
        elif order_type == 'market':
            # Robinhood market orders can be based on quantity (crypto asset) or amount (quote currency)
            if quantity is not None:
                 order_config['quantity'] = str(quantity.quantize(Decimal('0.00000001'))) # Adjust precision
            elif amount is not None:
                 order_config['amount'] = str(amount.quantize(Decimal('0.01'))) # Quote currency precision
            else:
                 logger.error("Either quantity or amount is required for market orders.")
                 return None
        else:
            logger.error(f"Unsupported order type: {order_type}")
            return None

        body = {
            "client_order_id": client_order_id,
            "side": side,
            "type": order_type,
            "symbol": trading_symbol,
            "time_in_force": time_in_force,
            f"{order_type}_order_config": order_config,
        }

        # --- Make API Request --- 
        logger.info(f"Placing order: {body}")
        try:
            response_data = self._make_api_request("POST", path, json.dumps(body))
            if response_data and isinstance(response_data, dict):
                logger.info(f"Order placement response: {response_data}")
                # Parse response into Order object
                return self._parse_order_response(response_data)
            else:
                logger.error(f"Failed to place order. Response: {response_data}")
                return None
        except Exception as e:
            logger.error(f"Exception during order placement: {e}", exc_info=True)
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        path = f"/api/v1/crypto/trading/orders/{order_id}/cancel/"
        logger.info(f"Attempting to cancel order {order_id} at {path}")
        # Cancel requests typically don't have a request body.
        # A successful cancel usually returns 200 OK with details or 204 No Content.
        response_data = self._make_api_request("POST", path)
        # Check if response is None (for 204) or has a success-like structure
        # This is heuristic; RH API docs should be the definitive source
        if response_data is None or response_data.get('order_id') == order_id:
            logger.info(f"Successfully requested cancellation for order {order_id}")
            return True
        else:
            logger.warning(f"Cancellation request for order {order_id} might have failed or status unclear. Response: {response_data}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Fetch details of a specific order by its Robinhood Order ID."""
        path = f"/api/v1/crypto/trading/orders/{order_id}/"
        logger.info(f"Fetching order details for {order_id} from {path}")
        response_data = self._make_api_request("GET", path)

        if response_data and 'order_id' in response_data:
            try:
                avg_price = Decimal(response_data.get('average_price') if response_data.get('average_price') else '0')
                order = Order(
                    id=response_data.get('id'),
                    client_order_id=response_data.get('client_order_id'),
                    symbol=self._convert_symbol_for_analysis(response_data.get('symbol')),
                    side=response_data.get('side'),
                    order_type=response_data.get('type'),
                    quantity=Decimal(response_data.get('quantity', '0')),
                    avg_fill_price=avg_price,
                    status=response_data.get('state'),
                    created_at=response_data.get('created_at')
                )
                logger.info(f"Order placed successfully: {order}")
                return order
            except Exception as e:
                logger.error(f"Could not parse order data for {order_id}: {response_data}. Error: {e}")
                return None
        else:
            logger.warning(f"Could not retrieve status for order {order_id}. Response: {response_data}")
            return None

    def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Order]:
        """Get historical order data."""
        query_params = []
        if symbol:
            query_params.append(f"symbol={self._convert_symbol_for_trading(symbol)}")
        query_params.append(f"page_size={limit}")
        
        path = f"/api/v1/crypto/trading/orders/?{'&'.join(query_params)}"
        logger.info(f"Fetching order history from {path}")
        try:
            response_data = self._make_api_request("GET", path)
            orders = []
            if response_data and 'results' in response_data and isinstance(response_data['results'], list):
                for order_data in response_data['results']:
                    parsed_order = self._parse_order_response(order_data)
                    if parsed_order:
                        orders.append(parsed_order)
                logger.info(f"Fetched {len(orders)} historical orders.")
                return orders
            else:
                logger.warning("No order history found or invalid response.")
                return []
        except Exception as e:
            logger.error(f"Error fetching order history: {e}", exc_info=True)
            return []

    def is_connected(self) -> bool:
        """Check if the broker connection is active by making a simple API call."""
        try:
            # Attempt a simple API call like getting account summary
            account_info = self.get_account_summary()
            # Check if the response is valid (e.g., not None and maybe contains expected keys)
            if account_info and isinstance(account_info, dict):
                logger.info("Robinhood connection test successful.")
                return True
            else:
                logger.warning("Robinhood connection test failed: Invalid response received.")
                return False
        except Exception as e:
            logger.error(f"Robinhood connection test failed: {e}", exc_info=True)
            return False

    def get_positions(self) -> List[Holding]: 
        """Retrieve all current crypto positions/holdings."""
        logger.info("Fetching all positions (holdings)...")
        try:
            # Reuse the get_holdings method without arguments to get all holdings
            holdings = self.get_holdings() 
            # Position interface might differ from Holding, adjust if necessary.
            # For now, assume Holding provides sufficient position info.
            logger.info(f"Found {len(holdings)} positions.")
            return holdings
        except Exception as e:
            logger.error(f"Error fetching positions: {e}", exc_info=True)
            return [] # Return empty list on error

    # --- Historical Data Method (using yfinance) --- 
    def get_historical_data(
        self,
        symbol: str,
        periods: int,
        interval: str = '1h' # Default to hourly, maps reasonably to yfinance intervals
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for a symbol.
        Uses yfinance as Robinhood's trading API doesn't provide this directly.

        Args:
            symbol (str): Trading symbol (e.g., 'BTC-USD').
            periods (int): Number of data points (candles) to fetch.
            interval (str): Data interval (e.g., '1m', '5m', '15m', '30m', '1h', '1d', '1wk').

        Returns:
            pd.DataFrame: DataFrame with columns [Timestamp, Open, High, Low, Close, Volume]
                          indexed by Timestamp, or None if fetching fails.
        """
        logger.info(f"Fetching historical data for {symbol} using yfinance: {periods} periods, interval {interval}")

        # Map environment interval to yfinance interval/period
        # yfinance intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        # yfinance max periods vary by interval (e.g., 1m is last 7 days)
        valid_intervals = {'1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'}
        yf_interval = interval if interval in valid_intervals else '1h' # Default to 1h if invalid
        if interval != yf_interval:
            logger.warning(f"Interval '{interval}' not directly supported by yfinance, using '{yf_interval}'.")

        # Determine period string for yfinance based on interval and periods
        # This is tricky because yfinance uses period strings ('1d', '5d', '1mo', 'max')
        # or start/end dates. Fetching exactly N periods requires calculation.
        # Let's try fetching slightly more data and tailing it.
        # Estimate duration needed based on interval
        try:
            if 'm' in yf_interval or 'h' in yf_interval:
                 # For intraday, fetch max allowed (e.g., 60 days for hourly, 7 days for minutely) and tail
                 if 'm' in yf_interval:
                     fetch_period = '7d'
                 else: # hourly
                     fetch_period = '60d' 
                 data = yf.download(symbol, period=fetch_period, interval=yf_interval, progress=False)
            elif 'd' in yf_interval or 'wk' in yf_interval or 'mo' in yf_interval:
                 # For daily/weekly/monthly, calculate start date roughly
                 # This is approximate!
                 days_per_period = {'d': 1, 'wk': 7, 'mo': 30}[yf_interval[-2:]]
                 total_days_needed = periods * days_per_period * 1.2 # Fetch 20% extra
                 end_date = datetime.datetime.now()
                 start_date = end_date - datetime.timedelta(days=total_days_needed)
                 data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=yf_interval, progress=False)
            else:
                 logger.error(f"Could not determine fetch parameters for interval {yf_interval}")
                 return None

            if data is None or data.empty:
                logger.error(f"yfinance returned no data for {symbol} with interval {yf_interval}.")
                return None

            # Ensure correct columns and index
            data.index.name = 'Timestamp'
            data.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                 logger.error(f"yfinance data missing required columns for {symbol}. Columns: {data.columns}")
                 return None

            # Take the last N periods
            historical_data = data[required_cols].tail(periods)

            if len(historical_data) < periods:
                 logger.warning(f"yfinance returned fewer periods ({len(historical_data)}) than requested ({periods}) for {symbol}.")

            logger.info(f"Successfully fetched {len(historical_data)} historical data points for {symbol} via yfinance.")
            return historical_data

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol} using yfinance: {e}", exc_info=True)
            return None

    # --- Helper Methods --- 

    @staticmethod
    def _get_current_timestamp() -> int:
        return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())

    @staticmethod
    def _get_query_params(key: str, *args: Optional[str]) -> str:
        """Helper to construct URL query parameters, handling symbol conversion except for the 'symbol' key itself."""
        if not args or all(arg is None for arg in args):
            return ""

        params = []
        for arg in args:
            if arg is not None:
                # Special handling for the 'symbol' key: use the value directly
                # as some endpoints (like best_bid_ask) expect the full 'XXX-USD' pair.
                value_to_use = arg if key == 'symbol' else self._convert_symbol_for_trading(arg)
                if value_to_use: # Ensure value is not None after potential conversion
                     params.append(f"{key}={value_to_use}")
            
        return "?" + "&".join(params)

    def _make_api_request(self, method: str, path: str, body: str = "") -> Any:
        timestamp = self._get_current_timestamp()
        url = self.base_url + path
        headers = {}
        json_payload = None

        try:
            message_to_sign = f"{self.api_key}{timestamp}{path}{method}"
            if body:
                 message_to_sign += body

            signed = self.private_key.sign(message_to_sign.encode("utf-8"))

            headers = {
                "x-api-key": self.api_key,
                "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
                "x-timestamp": str(timestamp),
            }

            logger.debug(f"Attempting {method} request to URL: {url}")
            logger.debug(f"Request Headers: {headers}")
            if method == "POST" and body:
                logger.debug(f"Request Body: {body}")

            response = self._session.request(method.upper(), url, headers=headers, json=json_payload, timeout=20) # Increased timeout

            logger.debug(f"Response status code: {response.status_code}")
            response_content_for_log = None
            if response.content:
                try:
                    # Attempt to log JSON response if possible, otherwise log text
                    response_json = response.json()
                    logger.debug(f"Response JSON: {response_json}")
                    response_content_for_log = response_json # Store for return on error
                except json.JSONDecodeError:
                    response_text = response.text
                    logger.debug(f"Response Text: {response_text[:500]}...") # Log truncated text
                    response_content_for_log = response_text # Store for return on error
            
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            # Handle potential empty response body for successful requests (e.g., 204 No Content)
            if response.status_code == 204 or not response.content:
                logger.debug("Request successful with empty response body.")
                return None 
            
            # Get JSON if possible
            json_response = None
            try:
                 json_response = response.json() if isinstance(response_content_for_log, dict) else response_content_for_log
                 # Log successful account summary response
                 if path == "/api/v1/crypto/trading/accounts/" and response.ok:
                    logger.info(f"ACCOUNT SUMMARY SUCCESS RESPONSE: {json_response}")
                 return json_response
            except Exception as json_err: # Catch potential errors if response_content_for_log wasn't dict
                 logger.error(f"Error processing response content: {json_err}. Content was: {response_content_for_log}")
                 return None

        except requests.exceptions.HTTPError as http_err:
            # Log detailed Robinhood error if available
            logger.error(f"HTTP error occurred during {method} request to {url}: {http_err}") # Log URL here
            # Add specific logging for account summary failures
            if path == "/api/v1/crypto/trading/accounts/":
                logger.error(f"ACCOUNT SUMMARY HTTP ERROR: Status={http_err.response.status_code}, Response Text={http_err.response.text[:500]}...")
            
            response_content_for_log = None
            try:
                 error_details = http_err.response.json()
                 logger.error(f"Robinhood error details: {error_details}")
                 response_content_for_log = error_details # Store parsed error
            except json.JSONDecodeError:
                 response_text = http_err.response.text
                 logger.error(f"Could not parse Robinhood error details from response: {response_text[:500]}...")
                 response_content_for_log = response_text # Store raw error text
            # Return the response content if available, otherwise None
            return response_content_for_log
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making API request to {path}: {e}", exc_info=True)
            # Add specific logging for account summary failures
            if path == "/api/v1/crypto/trading/accounts/":
                logger.error(f"ACCOUNT SUMMARY REQUEST EXCEPTION: Error={e}")
            return None

    def _convert_symbol_for_trading(self, symbol: Optional[str]) -> Optional[str]:
        """Converts a symbol like 'BTC-USD' to the format Robinhood API expects (e.g., 'BTC')."""
        if symbol is None:
            return None
        # Check if the symbol already contains a hyphen (likely a trading pair)
        if '-' in symbol:
            # Assume it's already in the correct format (e.g., 'BTC-USD') for endpoints expecting pairs
            # The API documentation is inconsistent; some endpoints want 'BTC', some want 'BTC-USD'.
            # Let's *assume* market data endpoints want the pair. If other endpoints fail, adjust.
            # UPDATE: Based on the error, market data endpoints *do not* want the pair.
            # Reverting to splitting logic, but logging a warning.
            parts = symbol.split('-')
            base_asset = parts[0].upper()
            logger.debug(f"Converting symbol '{symbol}' to base asset '{base_asset}' for API call. Ensure this is correct for the specific endpoint.")
            return base_asset
        else:
            # If no hyphen, assume it's already the base asset code
            return symbol.upper()

    def _convert_symbol_for_analysis(self, symbol: Optional[str]) -> Optional[str]:
        """Convert symbol like 'BTC' to analysis format 'BTC-USD'. Handles None."""
        if symbol is None:
            return None
        # Assumes USD pair if not specified, adjust if other pairs needed
        return f"{symbol.upper()}-USD" if '-' not in symbol else symbol.upper()

    def _parse_order_response(self, order_data: Dict[str, Any]) -> Optional[Order]:
        """Parses the JSON response for an order into an Order object."""
        if not isinstance(order_data, dict):
            logger.warning(f"Invalid order data format received: {order_data}")
            return None
        try:
            # Extract necessary fields, handling potential missing keys
            order_id = order_data.get('order_id')
            client_order_id = order_data.get('client_order_id')
            symbol = order_data.get('symbol') # RH uses trading pair like 'BTC-USD'
            side = order_data.get('side')
            order_type = order_data.get('type')
            status = order_data.get('state') # RH uses 'state' for status
            created_at = order_data.get('created_at')
            updated_at = order_data.get('last_transaction_at') or order_data.get('updated_at')

            # Quantity and price details might be nested or named differently
            quantity = order_data.get('quantity')
            avg_fill_price = order_data.get('average_price') # Or 'average_crypto_price'? Check API docs
            filled_quantity = order_data.get('cumulative_quantity')
            limit_price = order_data.get('limit_price')
            time_in_force = order_data.get('time_in_force')

            # Convert numeric fields to Decimal, handling None
            quantity_dec = Decimal(quantity) if quantity is not None else None
            avg_fill_price_dec = Decimal(avg_fill_price) if avg_fill_price is not None else None
            filled_quantity_dec = Decimal(filled_quantity) if filled_quantity is not None else None
            limit_price_dec = Decimal(limit_price) if limit_price is not None else None

            # Basic validation
            if not all([order_id, symbol, side, order_type, status]):
                logger.warning(f"Missing essential fields in order data: {order_data}")
                return None

            return Order(
                order_id=order_id,
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                status=status,
                quantity=quantity_dec,
                filled_quantity=filled_quantity_dec,
                avg_fill_price=avg_fill_price_dec,
                limit_price=limit_price_dec,
                created_at=created_at,
                updated_at=updated_at,
                time_in_force=time_in_force
            )
        except (TypeError, KeyError, InvalidOperation) as e:
            logger.error(f"Error parsing order response: {e}. Data: {order_data}", exc_info=True)
            return None

    def _get_signature(self, method: str, path: str, body: str, timestamp: int) -> str:
        """Generates the request signature."""
