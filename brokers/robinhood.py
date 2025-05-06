import os
import base64
import datetime
import json
from typing import Any, Dict, Optional, List
import uuid
from nacl.signing import SigningKey
from dotenv import load_dotenv
import requests
import pandas as pd
import logging
import time
from decimal import Decimal, InvalidOperation
from .base_broker import BaseBroker, Order, Holding, MarketData

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class RobinhoodBroker(BaseBroker):
    """Broker implementation for Robinhood Crypto API."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Robinhood broker, loading keys and setting up."""
        super().__init__(config if config is not None else {})
        logger.info("Initializing RobinhoodBroker...")
        self.api_key = os.getenv("ROBINHOOD_API_KEY")
        base64_private_key_env = os.getenv("BASE64_PRIVATE_KEY")

        if not self.api_key:
            logger.error("ROBINHOOD_API_KEY not found in environment variables.")
            raise ValueError("ROBINHOOD_API_KEY must be set.")
        if not base64_private_key_env:
            logger.error("BASE64_PRIVATE_KEY not found in environment variables.")
            raise ValueError("BASE64_PRIVATE_KEY must be set.")

        try:
            private_key_seed = base64.b64decode(base64_private_key_env)
            self.private_key = SigningKey(private_key_seed)
            logger.info("Signing key generated successfully from BASE64_PRIVATE_KEY.")
        except Exception as e:
            logger.error(f"Failed to decode BASE64_PRIVATE_KEY or create SigningKey: {e}")
            raise ValueError("Invalid BASE64_PRIVATE_KEY provided.") from e

        self.base_url = "https://trading.robinhood.com"
        self.latest_market_data = {}
        self._session = requests.Session() # Session for connection pooling
        # Headers will be generated per request
        logger.info(f"RobinhoodBroker initialized for base URL: {self.base_url}")

    @staticmethod
    def _get_current_timestamp() -> int:
        return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())

    @staticmethod
    def _get_query_params(key: str, *args: Optional[str]) -> str:
        if not args:
            return ""

        params = []
        for arg in args:
            if arg: # Ensure arg is not None or empty before adding
                params.append(f"{key}={arg}")

        return "?" + "&".join(params) if params else ""

    def get_authorization_header(
            self, method: str, path: str, body: str, timestamp: int
    ) -> Dict[str, str]:
        """Generate the authorization header based on the API documentation."""
        message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
        signed = self.private_key.sign(message_to_sign.encode("utf-8"))

        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
            "x-timestamp": str(timestamp),
            "Content-Type": "application/json; charset=utf-8" # Add Content-Type here
        }

    def _make_api_request(self, method: str, path: str, body: str = "") -> Optional[Any]:
        """Make an authenticated API request to Robinhood."""
        timestamp = self._get_current_timestamp()
        headers = self.get_authorization_header(method, path, body, timestamp)
        url = self.base_url + path
        response_json = None

        try:
            if method == "GET":
                response = self._session.get(url, headers=headers, timeout=10)
            elif method == "POST":
                # Ensure body is valid JSON string if provided
                request_body_data = json.loads(body) if body else None
                response = self._session.post(url, headers=headers, json=request_body_data, timeout=10)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None

            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            response_json = response.json()
            logger.debug(f"{method} {path} successful. Response: {response_json}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error making {method} request to {url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    logger.error(f"API Error Response: {error_details}")
                except json.JSONDecodeError:
                    logger.error(f"API Error Response (non-JSON): {e.response.text}")
            return None # Indicate failure
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode JSON response from {url}: {e}")
             return None # Indicate failure
        except Exception as e:
            logger.exception(f"An unexpected error occurred during API request to {url}: {e}")
            return None # Indicate failure

        return response_json

    def connect(self) -> bool:
        """Check if API keys are loaded. Connection is implicit."""
        if self.api_key:
            logger.info("RobinhoodBroker connection ready (API keys loaded).")
            return True
        logger.error("RobinhoodBroker connection failed: API keys missing.")
        return False

    def disconnect(self) -> None:
        """Close the underlying requests session."""
        if self._session:
            self._session.close()
            logger.info("RobinhoodBroker disconnected (session closed).")

    # Renamed from get_account_summary to match BaseBroker
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Fetch crypto trading account details. Implements BaseBroker.get_account_info."""
        path = "/api/v1/crypto/trading/accounts/"
        return self._make_api_request("GET", path)

    def get_available_capital(self) -> Optional[Decimal]:
        """Retrieve the available trading capital."""
        account_data = self.get_account_info()
        logger.debug(f"Raw account summary response received: {account_data}") 

        if account_data and isinstance(account_data, dict):
            potential_keys = ['buying_power'] 
            for key in potential_keys:
                 if key in account_data:
                     try:
                         capital_str = account_data[key]
                         capital = Decimal(capital_str)
                         logger.info(f"Successfully parsed available capital '{capital_str}' using key '{key}'")
                         return capital
                     except (InvalidOperation, TypeError, ValueError) as e:
                         logger.error(f"Error converting value '{account_data[key]}' for key '{key}' to Decimal: {e}")
            
            logger.error(f"Could not find a valid capital key {potential_keys} in account summary: {account_data}")
            return None 
        else:
            logger.error(f"Failed to fetch or parse account summary for capital. Invalid structure or empty results received: {account_data}")
            return None

    # Renamed from get_holdings, and modified to return Dict[str, Decimal]
    def _get_holdings_internal(self, asset_code: Optional[str] = None) -> Optional[Dict[str, Decimal]]:
        """Fetch crypto holdings, optionally filtered by asset_code (e.g., 'BTC'), returns a dict of asset_code to Decimal quantity."""
        query_params = self._get_query_params("asset_code", asset_code) if asset_code else ""
        path = f"/api/v1/crypto/trading/holdings/{query_params}"
        response = self._make_api_request("GET", path)

        if response and isinstance(response, dict) and 'results' in response:
            holdings_dict = {}
            for item in response['results']:
                try:
                    code = item.get('asset_code')
                    quantity_str = item.get('quantity')
                    if code and quantity_str is not None:
                        holdings_dict[code] = Decimal(quantity_str)
                    elif code:
                        holdings_dict[code] = Decimal('0') # Assume 0 if quantity is None but code exists
                except InvalidOperation:
                    logger.error(f"Invalid quantity format for asset {code}: {item.get('quantity')}")
                except Exception as e:
                    logger.error(f"Error processing holding item {item}: {e}")
            
            if asset_code and asset_code in holdings_dict: # If a specific asset was requested
                return {asset_code: holdings_dict[asset_code]}
            elif not asset_code: # If all assets were requested
                return holdings_dict
            else: # Specific asset requested but not found
                return {}
        elif response is not None: # Non-error response but not the expected format or empty
            logger.info(f"No holdings found or unexpected response format for asset_code '{asset_code}'. Response: {response}")
            return {}
        return None # API request failed

    # Implements BaseBroker.get_holdings
    def get_holdings(self) -> Optional[Dict[str, Decimal]]:
        """Retrieve current portfolio holdings (all assets)."""
        return self._get_holdings_internal(asset_code=None)

    def get_holding_quantity(self, asset: str) -> Optional[Decimal]:
        """Fetch the quantity of a specific crypto asset held."""
        # This method now relies on _get_holdings_internal returning the correct format
        holdings = self._get_holdings_internal(asset_code=asset)
        if holdings and asset in holdings:
            return holdings[asset]
        elif holdings is not None: # holdings is an empty dict if asset not found
            return Decimal('0')
        return None # Error fetching holdings

    def get_market_data(self, symbol: str) -> Optional[Dict[str, Decimal]]:
        """Fetch the best bid/ask price for a specific symbol (e.g., 'BTC-USD')."""
        if not symbol or '-' not in symbol:
             logger.error(f"Invalid symbol format for get_market_data: '{symbol}'. Expected 'BASE-QUOTE' (e.g., 'BTC-USD').")
             return None
        
        query_params = self._get_query_params("symbol", symbol) 
        path = f"/api/v1/crypto/marketdata/best_bid_ask/{query_params}"
        logger.info(f"Fetching market data for {symbol} from {path}") 
        response_data = self._make_api_request("GET", path)

        if response_data and 'results' in response_data and response_data['results']:
            market_info = response_data['results'][0] 
            logger.debug(f"Raw market data received for {symbol}: {market_info}")
            try:
                bid_price_str = market_info.get('bid_inclusive_of_sell_spread')
                ask_price_str = market_info.get('ask_inclusive_of_buy_spread')
                timestamp_str = market_info.get('timestamp') 
                
                if bid_price_str is None or ask_price_str is None:
                    logger.warning(f"Missing bid or ask price in market data for {symbol}: {market_info}")
                    return None

                bid_price = Decimal(bid_price_str)
                ask_price = Decimal(ask_price_str)
                
                try:
                    if '.' in timestamp_str:
                        ts_parts = timestamp_str.split('.')
                        fractional_seconds = ts_parts[1].replace('Z', '')[:6]
                        timestamp_str_truncated = f"{ts_parts[0]}.{fractional_seconds}+00:00"
                    else:
                        timestamp_str_truncated = timestamp_str.replace('Z', '+00:00')
                        
                    timestamp_dt = datetime.datetime.fromisoformat(timestamp_str_truncated)
                except (ValueError, TypeError, IndexError) as ts_err:
                    logger.warning(f"Could not parse timestamp '{timestamp_str}': {ts_err}")
                    timestamp_dt = datetime.datetime.now(datetime.timezone.utc) 

                market_data_obj = {
                    "symbol": self._convert_symbol_for_analysis(market_info.get('symbol')),
                    "bid_price": float(bid_price), 
                    "ask_price": float(ask_price),
                    "timestamp": timestamp_dt
                }
                self.latest_market_data[symbol] = market_data_obj 
                return market_data_obj 
            except (InvalidOperation, KeyError, TypeError) as e:
                logger.warning(f"Could not parse market data: {market_info}. Error: {e}")
                return None
        else:
            logger.warning(f"No market data found for symbol: {symbol}")
            return None

    # Renamed from get_current_price to match BaseBroker
    def get_latest_price(self, symbol: str) -> Optional[Decimal]:
        """Fetch the current market price (mid-price) for a symbol. Implements BaseBroker.get_latest_price."""
        market_data = self.get_market_data(symbol)
        if market_data and 'bid_price' in market_data and 'ask_price' in market_data:
            try:
                bid_price = Decimal(market_data['bid_price'])
                ask_price = Decimal(market_data['ask_price'])
                return (bid_price + ask_price) / 2
            except InvalidOperation:
                logger.error(f"Could not convert bid/ask to Decimal for {symbol}: {market_data}")
                return None
        return None

    # Modified to include client_order_id
    def place_order(self, symbol: str, side: str, order_type: str, quantity: Optional[Decimal] = None, amount: Optional[Decimal] = None, time_in_force: str = 'gtc', limit_price: Optional[Decimal] = None, stop_price: Optional[Decimal] = None, client_order_id: Optional[str] = None) -> Optional[Order]:
        """Place a trading order."""
        trading_symbol = self._convert_symbol_for_trading(symbol)
        if not trading_symbol:
            logger.error(f"Invalid symbol for trading: {symbol}. Expected format 'BASE-QUOTE' (e.g., BTC-USD)")
            return None

        client_order_id_to_use = client_order_id or str(uuid.uuid4())
        logger.info(f"Placing {order_type} order: Side={side}, Symbol={trading_symbol}, Quantity={quantity}, Amount={amount}, ClientOrderID={client_order_id_to_use}")

        order_config = {}
        if order_type.lower() == "market":
            if quantity:
                # Robinhood API expects quantity as string for market orders based on quantity
                order_config['quantity'] = str(quantity.quantize(Decimal('1e-8'))) 
            elif amount:
                # Robinhood API expects amount as string for market orders based on amount (e.g. $100 of BTC)
                order_config['amount'] = str(amount.quantize(Decimal('1e-2'))) # Typically 2 decimal places for currency
            else:
                logger.error("Market order requires either quantity or amount.")
                return None
        elif order_type.lower() == "limit":
            if not quantity or not limit_price:
                logger.error("Limit order requires quantity and limit_price.")
                return None
            order_config['quantity'] = str(quantity.quantize(Decimal('1e-8')))
            order_config['limit_price'] = str(limit_price.quantize(Decimal('1e-2'))) # Price usually 2 decimal places
            order_config['time_in_force'] = time_in_force
        # Add other order types like stop_loss, stop_limit, take_profit if supported and needed
        else:
            logger.error(f"Unsupported order type: {order_type}")
            return None

        body = {
            "client_order_id": client_order_id_to_use,
            "side": side.upper(), # Ensure side is uppercase (BUY/SELL)
            "type": order_type.upper(), # Ensure type is uppercase (MARKET/LIMIT)
            "symbol": trading_symbol,
            f"{order_type.lower()}_order_config": order_config,
        }

        path = "/api/v1/crypto/trading/orders/"
        logger.debug(f"Placing order with body: {json.dumps(body)}")
        response_data = self._make_api_request("POST", path, json.dumps(body))

        if response_data:
            logger.info(f"Order placement attempt response: {response_data}")
            # Check for immediate error in response, e.g. if 'id' is not present or 'state' indicates failure
            if 'id' not in response_data or response_data.get('state') == 'failed':
                 logger.error(f"Order placement failed or 'id' missing in response: {response_data}")
                 # Consider if an Order object should still be created with error status
                 # For now, returning None if critical info like 'id' is missing or explicit failure.
                 return None 
            return self._parse_order_response(response_data)
        else:
            logger.error("Failed to place order: No response from API.")
            return None

    # Implements BaseBroker.place_market_order
    def place_market_order(self, symbol: str, side: str, quantity: Decimal, client_order_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Place a market order."""
        order_object = self.place_order(
            symbol=symbol, 
            side=side, 
            order_type="market", 
            quantity=quantity, 
            client_order_id=client_order_id
        )
        if order_object:
            return order_object.__dict__
        return None

    # Modified to return Optional[Dict[str, Any]] to match BaseBroker
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the status of a specific order by its ID."""
        path = f"/api/v1/crypto/trading/orders/{order_id}/"
        logger.info(f"Fetching order details for {order_id} from {path}")
        response_data = self._make_api_request("GET", path)

        if response_data and isinstance(response_data, dict):
            logger.info(f"Order status response: {response_data}")
            order_obj = self._parse_order_response(response_data)
            if order_obj:
                return order_obj.__dict__
            else:
                logger.error(f"Failed to parse order status response for order {order_id}: {response_data}")
                return None # Parsing failed
        else:
            logger.warning(f"No data or non-dict response received for order status {order_id}. Response: {response_data}")
            return None # API request failed or bad response

    # Implements BaseBroker.cancel_order
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order. Returns True on success, False otherwise."""
        path = f"/api/v1/crypto/trading/orders/{order_id}/cancel/"
        logger.info(f"Attempting to cancel order {order_id} at {path}")
        # The cancel API returns an empty body (200 OK) on success.
        # _make_api_request returns {} for successful empty JSON responses, or None on error.
        response_data = self._make_api_request("POST", path) 
        
        if response_data is not None: # Could be an empty dict {} for success
            logger.info(f"Successfully requested cancellation for order {order_id}. API Response: {response_data}")
            return True
        else:
            logger.warning(f"Cancellation request for order {order_id} failed or status unclear. API did not return a successful response.")
            return False

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
            account_info = self.get_account_info()
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
            holdings = self.get_holdings() 
            logger.info(f"Found {len(holdings)} positions.")
            return holdings
        except Exception as e:
            logger.error(f"Error fetching positions: {e}", exc_info=True)
            return [] 

    def get_historical_data(
        self,
        symbol: str,
        periods: int,
        interval: str = '1h' 
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

        valid_intervals = {'1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'}
        yf_interval = interval if interval in valid_intervals else '1h' 
        if interval != yf_interval:
            logger.warning(f"Interval '{interval}' not directly supported by yfinance, using '{yf_interval}'.")

        try:
            if 'm' in yf_interval or 'h' in yf_interval:
                 fetch_period = '7d' if 'm' in yf_interval else '60d' 
                 data = pd.read_json(f"https://query1.finance.yahoo.com/v7/finance/chart/{symbol}?range=1d&interval={yf_interval}&corsDomain=finance.yahoo.com")
            elif 'd' in yf_interval or 'wk' in yf_interval or 'mo' in yf_interval:
                 days_per_period = {'d': 1, 'wk': 7, 'mo': 30}[yf_interval[-2:]]
                 total_days_needed = periods * days_per_period * 1.2 
                 end_date = datetime.datetime.now()
                 start_date = end_date - datetime.timedelta(days=total_days_needed)
                 data = pd.read_json(f"https://query1.finance.yahoo.com/v7/finance/chart/{symbol}?period1={int(start_date.timestamp())}&period2={int(end_date.timestamp())}&interval={yf_interval}&corsDomain=finance.yahoo.com")
            else:
                 logger.error(f"Could not determine fetch parameters for interval {yf_interval}")
                 return None

            if data is None or data.empty:
                logger.error(f"yfinance returned no data for {symbol} with interval {yf_interval}.")
                return None

            data.index.name = 'Timestamp'
            data.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                 logger.error(f"yfinance data missing required columns for {symbol}. Columns: {data.columns}")
                 return None

            historical_data = data[required_cols].tail(periods)

            if len(historical_data) < periods:
                 logger.warning(f"yfinance returned fewer periods ({len(historical_data)}) than requested ({periods}) for {symbol}.")

            logger.info(f"Successfully fetched {len(historical_data)} historical data points for {symbol} via yfinance.")
            return historical_data

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol} using yfinance: {e}", exc_info=True)
            return None

    def _convert_symbol_for_trading(self, symbol: Optional[str]) -> Optional[str]:
        """Converts a symbol like 'BTC-USD' to the format Robinhood API expects (e.g., 'BTC')."""
        if symbol is None:
            return None
        if '-' in symbol:
            parts = symbol.split('-')
            base_asset = parts[0].upper()
            logger.debug(f"Converting symbol '{symbol}' to base asset '{base_asset}' for API call. Ensure this is correct for the specific endpoint.")
            return base_asset
        else:
            return symbol.upper()

    def _convert_symbol_for_analysis(self, symbol: Optional[str]) -> Optional[str]:
        """Convert symbol like 'BTC' to analysis format 'BTC-USD'. Handles None."""
        if symbol is None:
            return None
        return f"{symbol.upper()}-USD" if '-' not in symbol else symbol.upper()

    def _parse_order_response(self, order_data: Dict[str, Any]) -> Optional[Order]:
        """Parses the JSON response for an order into an Order object."""
        if not isinstance(order_data, dict):
            logger.warning(f"Invalid order data format received: {order_data}")
            return None
        try:
            order_id = order_data.get('order_id')
            client_order_id = order_data.get('client_order_id')
            symbol = order_data.get('symbol') 
            side = order_data.get('side')
            order_type = order_data.get('type')
            status = order_data.get('state') 
            created_at = order_data.get('created_at')
            updated_at = order_data.get('last_transaction_at') or order_data.get('updated_at')

            quantity = order_data.get('quantity')
            avg_fill_price = order_data.get('average_price') 
            filled_quantity = order_data.get('cumulative_quantity')
            limit_price = order_data.get('limit_price')
            time_in_force = order_data.get('time_in_force')

            quantity_dec = Decimal(quantity) if quantity is not None else None
            avg_fill_price_dec = Decimal(avg_fill_price) if avg_fill_price is not None else None
            filled_quantity_dec = Decimal(filled_quantity) if filled_quantity is not None else None
            limit_price_dec = Decimal(limit_price) if limit_price is not None else None

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
