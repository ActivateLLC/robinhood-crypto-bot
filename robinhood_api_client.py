import base64
import datetime
import json
import logging
import os
from typing import Any, Dict, Optional
import uuid
import requests
from dotenv import load_dotenv
from nacl.signing import SigningKey

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("ROBINHOOD_API_KEY")
BASE64_PRIVATE_KEY = os.getenv("ROBINHOOD_PRIVATE_KEY")

class CryptoAPITrading:
    def __init__(self):
        if not API_KEY or not BASE64_PRIVATE_KEY:
            logger.error("ROBINHOOD_API_KEY or ROBINHOOD_PRIVATE_KEY not found in environment variables.")
            raise ValueError("API Key and Private Key must be set in the environment.")
        self.api_key = API_KEY
        private_key_seed = base64.b64decode(BASE64_PRIVATE_KEY)
        self.private_key = SigningKey(private_key_seed)
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
            if arg is not None:
                params.append(f"{key}={arg}")
        return "?" + "&".join(params) if params else ""

    def get_authorization_header(self, method: str, path: str, body: str, timestamp: int) -> Dict[str, str]:
        message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
        signed = self.private_key.sign(message_to_sign.encode("utf-8"))
        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
            "x-timestamp": str(timestamp),
            "Content-Type": "application/json; charset=utf-8"
        }

    def make_api_request(self, method: str, path: str, body: str = "") -> Any:
        timestamp = self._get_current_timestamp()
        headers = self.get_authorization_header(method, path, body, timestamp)
        url = self.base_url + path
        json_payload = None
        try:
            if body:
                json_payload = json.loads(body)
            response = requests.request(method, url, headers=headers, json=json_payload, timeout=15)
            response.raise_for_status()
            if response.status_code == 204:
                logger.info(f"Request successful ({response.status_code}), no content returned.")
                return None
            elif response.content:
                try:
                    json_response = response.json()
                    logger.debug(f"Response ({response.status_code}): {json_response}")
                    return json_response
                except json.JSONDecodeError:
                    logger.warning(f"Response ({response.status_code}) is not valid JSON: {response.text}")
                    return response.text
            else:
                logger.info(f"Response ({response.status_code}) received with empty body.")
                return None
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out: {method} {url}")
            return {"error": "Request timed out"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making API request to {url}: {e}")
            if e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {error_details}")
                    return {"error": "API request failed", "status_code": e.response.status_code, "details": error_details}
                except json.JSONDecodeError:
                    logger.error(f"Error response body (non-JSON): {e.response.text}")
            return None

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
        body_json = json.dumps(body_dict, separators=(',', ':')) 
        return self.make_api_request("POST", path, body_json)

    def cancel_order(self, order_id: str) -> Any:
        path = f"/api/v1/crypto/trading/orders/{order_id}/cancel/"
        return self.make_api_request("POST", path)

    def get_order(self, order_id: str) -> Any:
        path = f"/api/v1/crypto/trading/orders/{order_id}/"
        return self.make_api_request("GET", path)

    def get_orders(self, **kwargs) -> Any:
        query_params_list = [f"{key}={value}" for key, value in kwargs.items() if value is not None]
        query_string = "?" + "&".join(query_params_list) if query_params_list else ""
        path = f"/api/v1/crypto/trading/orders/{query_string}"
        return self.make_api_request("GET", path)

    def convert_symbol_for_trading(self, symbol: str) -> str:
        return symbol.split('-')[0]

    def convert_symbol_for_analysis(self, symbol: str) -> str:
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
