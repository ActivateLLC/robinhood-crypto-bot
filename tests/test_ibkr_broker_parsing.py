import pytest
from ib_insync import Contract
import datetime
from typing import List, Optional, Dict, Any
from decimal import Decimal
import pandas as pd

# Assuming IBKRBroker is accessible for testing
# If not, adjust the import path as necessary
from brokers.ibkr import IBKRBroker
from brokers.base_broker import Holding, MarketData, Order

# Mock logger or configure basic logging for tests if needed
import logging
logging.basicConfig(level=logging.DEBUG)

# --- Create a concrete class for testing parsing --- 
class ConcreteTestIBKRBroker(IBKRBroker):
    # --- Stubs for BaseBroker abstract methods ---
    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        pass

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        return None

    def get_holdings(self) -> Optional[Dict[str, Decimal]]:
        return None

    def get_latest_price(self, symbol: str) -> Optional[Decimal]:
        return None

    def place_market_order(self, symbol: str, side: str, quantity: Decimal, client_order_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return {"order_id": "dummy_market_order", "status": "Submitted"}

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        return None

    def get_order_history(self, limit: int = 100) -> List[Order]:
        return []

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        return None

    def get_historical_candles(self, symbol: str, interval: str, lookback_period: str) -> List[pd.DataFrame]:
        return []

    def get_account_summary(self) -> dict: 
        return {}

    def get_positions(self) -> list: 
        return []

    def get_open_orders(self) -> list: 
        return []

    def get_current_price(self, symbol: str) -> float: 
        return None

    def get_historical_data(self, symbol: str, interval: str, lookback: str) -> list: 
        return []

@pytest.fixture
def broker():
    """Provides a concrete IBKRBroker instance for testing parsing methods."""
    # We don't need a real connection for parsing tests
    # Use the concrete subclass defined above
    return ConcreteTestIBKRBroker(host='dummy', port=0, client_id=999)

# --- Test Stock Parsing --- 

def test_parse_stock_valid(broker):
    contract = broker._parse_stock_symbol("AAPL")
    assert contract is not None
    assert contract.symbol == "AAPL"
    assert contract.secType == "STK"
    assert contract.currency == "USD"
    assert contract.exchange == "SMART"

def test_parse_stock_invalid_lowercase(broker):
    # Expect None because the helper expects uppercase
    assert broker._parse_stock_symbol("aapl") is None 

def test_parse_stock_invalid_long(broker):
    assert broker._parse_stock_symbol("TOOLONG") is None

def test_parse_stock_invalid_with_num(broker):
    assert broker._parse_stock_symbol("AAPL1") is None

# --- Test Option Parsing --- 

def test_parse_option_valid_call(broker):
    contract = broker._parse_option_symbol("SPY 241220C00500000")
    assert contract is not None
    assert contract.symbol == "SPY"
    assert contract.secType == "OPT"
    assert contract.lastTradeDateOrContractMonth == "20241220"
    assert contract.strike == 500.0
    assert contract.right == "C"
    assert contract.currency == "USD"
    assert contract.exchange == "SMART"

def test_parse_option_valid_put(broker):
    contract = broker._parse_option_symbol("QQQ 250321P00400000")
    assert contract is not None
    assert contract.symbol == "QQQ"
    assert contract.secType == "OPT"
    assert contract.lastTradeDateOrContractMonth == "20250321"
    assert contract.strike == 400.0
    assert contract.right == "P"

def test_parse_option_invalid_format_no_space(broker):
    assert broker._parse_option_symbol("AAPL230915C00180000") is None

def test_parse_option_invalid_format_wrong_date(broker):
    assert broker._parse_option_symbol("AAPL 23091C00180000") is None

def test_parse_option_invalid_format_wrong_right(broker):
    assert broker._parse_option_symbol("AAPL 230915X00180000") is None

def test_parse_option_invalid_format_wrong_strike(broker):
    assert broker._parse_option_symbol("AAPL 230915C0180000") is None # Needs 8 digits

# --- Test Future Parsing --- 

@pytest.mark.parametrize("symbol, expected_root, expected_month, expected_exchange", [
    ("ESZ3", "ES", "202312", "CME"),
    ("NQ H4", "NQ", "202403", "CME"), # Test with space (should be stripped)
    ("CLG25", "CL", "202502", "NYMEX"),
    ("ZCN24", "ZC", "202407", "CBOT"),
    ("RTYM4", "RTY", "202406", "CME"),
])
def test_parse_future_valid(broker, symbol, expected_root, expected_month, expected_exchange):
    contract = broker._parse_future_symbol(symbol.upper().replace(" ", ""))
    assert contract is not None
    assert contract.symbol == expected_root
    assert contract.secType == "FUT"
    assert contract.lastTradeDateOrContractMonth == expected_month
    assert contract.exchange == expected_exchange
    assert contract.currency == "USD"

def test_parse_future_invalid_month(broker):
    assert broker._parse_future_symbol("ESY3") is None # Y is not a valid month code

def test_parse_future_invalid_format(broker):
    assert broker._parse_future_symbol("ESDEC23") is None

# --- Test Crypto Parsing --- 

def test_parse_crypto_valid_usd(broker):
    contract = broker._parse_crypto_symbol("BTC-USD")
    assert contract is not None
    assert contract.symbol == "BTC"
    assert contract.secType == "CRYPTO"
    assert contract.currency == "USD"
    assert contract.exchange == "PAXOS"

def test_parse_crypto_valid_eur(broker):
    # Note: The helper warns about non-USD but still parses
    contract = broker._parse_crypto_symbol("ETH-EUR")
    assert contract is not None
    assert contract.symbol == "ETH"
    assert contract.secType == "CRYPTO"
    assert contract.currency == "EUR"
    assert contract.exchange == "PAXOS" # Default, might need adjustment

def test_parse_crypto_invalid_format_no_dash(broker):
    assert broker._parse_crypto_symbol("BTCUSD") is None

def test_parse_crypto_invalid_format_short_quote(broker):
    assert broker._parse_crypto_symbol("BTC-US") is None

# --- Test Main _symbol_to_contract Delegator --- 

@pytest.mark.parametrize("symbol, expected_secType", [
    ("AAPL", "STK"),
    ("GOOG", "STK"),
    ("SPY 241220C00500000", "OPT"),
    ("QQQ 250321P00400000", "OPT"),
    ("ESZ3", "FUT"),
    ("NQ H4", "FUT"),
    ("CLG25", "FUT"),
    ("BTC-USD", "CRYPTO"),
    ("ETH-EUR", "CRYPTO"),
])
def test_symbol_to_contract_delegation(broker, symbol, expected_secType):
    contract = broker._symbol_to_contract(symbol)
    assert contract is not None
    assert contract.secType == expected_secType

def test_symbol_to_contract_unknown(broker):
    assert broker._symbol_to_contract("UNKNOWN_SYMBOL_FORMAT") is None

def test_symbol_to_contract_empty(broker):
    assert broker._symbol_to_contract("") is None
