import base64
import datetime
import json
from typing import Any, Dict, Optional
import uuid
import requests
from nacl.signing import SigningKey
import os # Added import os
from dotenv import load_dotenv # Added dotenv import

# Load environment variables
load_dotenv()

# Use environment variables for keys
API_KEY = os.getenv("ROBINHOOD_API_KEY", "ADD YOUR API KEY HERE")
BASE64_PRIVATE_KEY = os.getenv("ROBINHOOD_PRIVATE_KEY", "ADD YOUR PRIVATE KEY HERE")

class CryptoAPITrading:
    def __init__(self):
        self.api_key = API_KEY
        # Ensure private key is loaded correctly, handle potential None
        if BASE64_PRIVATE_KEY == "ADD YOUR PRIVATE KEY HERE" or BASE64_PRIVATE_KEY is None:
             raise ValueError("ROBINHOOD_PRIVATE_KEY not set in environment or .env file")
        try:
            private_key_seed = base64.b64decode(BASE64_PRIVATE_KEY)
            self.private_key = SigningKey(private_key_seed)
        except Exception as e:
            raise ValueError(f"Error decoding or initializing private key: {e}")

        self.base_url = "https://trading.robinhood.com"

    @staticmethod
    def _get_current_timestamp() -> int:
        return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())

    @staticmethod
    def get_query_params(key: str, *args: Optional[str]) -> str:
        if not args:
            return ""

        params = []
        for arg in args:
            if arg: # Ensure arg is not None or empty
                params.append(f"{key}={arg}")

        return "?" + "&".join(params)

    def make_api_request(self, method: str, path: str, body: str = "") -> Any:
        timestamp = self._get_current_timestamp()
        headers = self.get_authorization_header(method, path, body, timestamp)
        url = self.base_url + path

        try:
            response = None # Initialize response
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method == "POST":
                # Ensure body is valid JSON before loading if it's a string
                json_body = {}
                if body:
                    try:
                        json_body = json.loads(body)
                    except json.JSONDecodeError:
                        print(f"Warning: Body is not valid JSON: {body}")
                        # Decide how to handle non-JSON body for POST
                        # For now, let's pass it as data if not JSON
                        response = requests.post(url, headers=headers, data=body, timeout=10)
                    else:
                        response = requests.post(url, headers=headers, json=json_body, timeout=10)
                else:
                     # Handle POST request with empty body if necessary
                     response = requests.post(url, headers=headers, timeout=10)

            if response is None:
                 print(f"Error: No response received for {method} {url}")
                 return None

            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            # Try to parse JSON, return text if not JSON
            try:
                return response.json()
            except json.JSONDecodeError:
                return response.text

        except requests.RequestException as e:
            print(f"Error making API request to {url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during API request: {e}")
            return None

    def get_authorization_header(
            self, method: str, path: str, body: str, timestamp: int
    ) -> Dict[str, str]:
        message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
        signed = self.private_key.sign(message_to_sign.encode("utf-8"))

        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
            "x-timestamp": str(timestamp),
            "Content-Type": "application/json" # Added Content-Type for POSTs
        }

    def get_account(self) -> Any:
        path = "/api/v1/crypto/trading/accounts/"
        return self.make_api_request("GET", path)

    # The symbols argument must be formatted in trading pairs, e.g "BTC-USD", "ETH-USD". If no symbols are provided,
    # all supported symbols will be returned
    def get_trading_pairs(self, *symbols: Optional[str]) -> Any:
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/trading/trading_pairs/{query_params}"
        return self.make_api_request("GET", path)

    # The asset_codes argument must be formatted as the short form name for a crypto, e.g "BTC", "ETH". If no asset
    # codes are provided, all crypto holdings will be returned
    def get_holdings(self, *asset_codes: Optional[str]) -> Any:
        # Filter out None or empty strings from asset_codes before creating query params
        valid_asset_codes = [code for code in asset_codes if code]
        query_params = self.get_query_params("asset_code", *valid_asset_codes)
        path = f"/api/v1/crypto/trading/holdings/{query_params}"
        return self.make_api_request("GET", path)

    # The symbols argument must be formatted in trading pairs, e.g "BTC-USD", "ETH-USD". If no symbols are provided,
    # the best bid and ask for all supported symbols will be returned
    def get_best_bid_ask(self, *symbols: Optional[str]) -> Any:
        valid_symbols = [s for s in symbols if s]
        query_params = self.get_query_params("symbol", *valid_symbols)
        path = f"/api/v1/crypto/marketdata/best_bid_ask/{query_params}"
        return self.make_api_request("GET", path)

    # The symbol argument must be formatted in a trading pair, e.g "BTC-USD", "ETH-USD"
    # The side argument must be "bid", "ask", or "both".
    # Multiple quantities can be specified in the quantity argument, e.g. "0.1,1,1.999".
    def get_estimated_price(self, symbol: str, side: str, quantity: str) -> Any:
        # Basic validation
        if not symbol or side not in ["bid", "ask", "both"] or not quantity:
            print("Error: Invalid arguments for get_estimated_price")
            return None
        path = f"/api/v1/crypto/marketdata/estimated_price/?symbol={symbol}&side={side}&quantity={quantity}"
        return self.make_api_request("GET", path)

    def place_order(
            self,
            client_order_id: str,
            side: str,
            order_type: str,
            symbol: str,
            order_config: Dict[str, str],
    ) -> Any:
        # Basic validation
        if not all([client_order_id, side in ["buy", "sell"], order_type in ["market", "limit"], symbol, order_config]):
            print("Error: Invalid arguments for place_order")
            return None

        body = {
            "client_order_id": client_order_id,
            "side": side,
            "type": order_type,
            "symbol": symbol,
            f"{order_type}_order_config": order_config,
        }
        path = "/api/v1/crypto/trading/orders/"
        # Ensure body is dumped to a JSON string for the POST request
        return self.make_api_request("POST", path, json.dumps(body))

    def cancel_order(self, order_id: str) -> Any:
        if not order_id:
            print("Error: Invalid order_id for cancel_order")
            return None
        path = f"/api/v1/crypto/trading/orders/{order_id}/cancel/"
        # Cancel request likely needs an empty POST body
        return self.make_api_request("POST", path, "")

    def get_order(self, order_id: str) -> Any:
        if not order_id:
            print("Error: Invalid order_id for get_order")
            return None
        path = f"/api/v1/crypto/trading/orders/{order_id}/"
        return self.make_api_request("GET", path)

# Example usage (optional, can be commented out or removed)
# if __name__ == '__main__':
#     try:
#         trader = CryptoAPITrading()
#         print("Fetching account info...")
#         account_info = trader.get_account()
#         print(account_info)

#         print("\nFetching BTC holdings...")
#         btc_holdings = trader.get_holdings('BTC')
#         print(btc_holdings)

#         print("\nFetching BTC-USD best bid/ask...")
#         best_bid_ask = trader.get_best_bid_ask('BTC-USD')
#         print(best_bid_ask)

#         # Example: Place a small market buy order (USE WITH EXTREME CAUTION)
#         # print("\nPlacing example market buy order...")
#         # market_order_config = {"spend": "5.00"} # Spend $5 USD
#         # client_id = str(uuid.uuid4())
#         # order_response = trader.place_order(client_id, "buy", "market", "BTC-USD", market_order_config)
#         # print(order_response)

#     except ValueError as ve:
#         print(f"Initialization Error: {ve}")
#     except Exception as e:
#         print(f"An error occurred during example usage: {e}")
