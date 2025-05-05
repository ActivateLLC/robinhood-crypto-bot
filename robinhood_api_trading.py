import base64
import datetime
import json
from typing import Any, Dict, Optional
import uuid
import requests
from nacl.signing import SigningKey
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use environment variables for keys
API_KEY = os.getenv("ROBINHOOD_API_KEY")
BASE64_PRIVATE_KEY = os.getenv("ROBINHOOD_PRIVATE_KEY")

class CryptoAPITrading:
    def __init__(self):
        if not API_KEY or not BASE64_PRIVATE_KEY:
            raise ValueError("API Key and Private Key must be set in the environment.")
        self.api_key = API_KEY
        private_key_seed = base64.b64decode(BASE64_PRIVATE_KEY)
        self.private_key = SigningKey(private_key_seed)
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
            params.append(f"{key}={arg}")
        return "?" + "&".join(params)

    def make_api_request(self, method: str, path: str, body: str = "") -> Any:
        timestamp = self._get_current_timestamp()
        headers = self.get_authorization_header(method, path, body, timestamp)
        url = self.base_url + path
        try:
            response = {}
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=json.loads(body), timeout=10)
            return response.json()
        except requests.RequestException as e:
            print(f"Error making API request: {e}")
            return None

    def get_authorization_header(self, method: str, path: str, body: str, timestamp: int) -> Dict[str, str]:
        message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
        signed = self.private_key.sign(message_to_sign.encode("utf-8"))
        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
            "x-timestamp": str(timestamp),
        }

    def get_account(self) -> Any:
        path = "/api/v1/crypto/trading/accounts/"
        return self.make_api_request("GET", path)

    def get_trading_pairs(self, *symbols: Optional[str]) -> Any:
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/trading/trading_pairs/{query_params}"
        return self.make_api_request("GET", path)

    def get_holdings(self, *asset_codes: Optional[str]) -> Any:
        query_params = self.get_query_params("asset_code", *asset_codes)
        path = f"/api/v1/crypto/trading/holdings/{query_params}"
        return self.make_api_request("GET", path)

    def get_best_bid_ask(self, *symbols: Optional[str]) -> Any:
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/marketdata/best_bid_ask/{query_params}"
        return self.make_api_request("GET", path)

    def get_estimated_price(self, symbol: str, side: str, quantity: str) -> Any:
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
        body = {
            "client_order_id": client_order_id,
            "side": side,
            "type": order_type,
            "symbol": symbol,
            f"{order_type}_order_config": order_config,
        }
        path = "/api/v1/crypto/trading/orders/"
        return self.make_api_request("POST", path, json.dumps(body))

    def cancel_order(self, order_id: str) -> Any:
        path = f"/api/v1/crypto/trading/orders/{order_id}/cancel/"
        return self.make_api_request("POST", path)

    def get_order(self, order_id: str) -> Any:
        path = f"/api/v1/crypto/trading/orders/{order_id}/"
        return self.make_api_request("GET", path)

    def get_orders(self) -> Any:
        path = "/api/v1/crypto/trading/orders/"
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
