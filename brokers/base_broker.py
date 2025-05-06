# /Users/activate/Dev/robinhood-crypto-bot/brokers/base_broker.py
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal

logger = logging.getLogger(__name__)

class BaseBroker(ABC):
    """
    Abstract Base Class for all broker implementations.
    Defines the standard interface for interacting with different brokers.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the broker with necessary configuration.
        """
        self.config = config
        logger.info(f"Initializing {self.__class__.__name__}...")

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish a connection to the broker, if necessary.
        Returns True if connection is successful or not needed, False otherwise.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Disconnect from the broker, if necessary.
        """
        pass

    @abstractmethod
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve general account information (e.g., balance, buying power).
        Returns a dictionary with account details or None if an error occurs.
        """
        pass

    @abstractmethod
    def get_holdings(self) -> Optional[Dict[str, Decimal]]:
        """
        Retrieve current portfolio holdings.
        Returns a dictionary mapping asset symbols (e.g., 'BTC', 'ETH')
        to their quantities (as Decimal), or None if an error occurs.
        """
        pass

    @abstractmethod
    def get_latest_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get the latest market price for a given trading pair symbol (e.g., 'BTC-USD').
        Returns the price as a Decimal or None if an error occurs.
        """
        pass

    @abstractmethod
    def place_market_order(
        self,
        symbol: str,          # Trading pair (e.g., 'BTC-USD')
        side: str,            # 'buy' or 'sell'
        quantity: Decimal,    # Amount of the base asset (e.g., BTC quantity)
        client_order_id: Optional[str] = None # Optional client-provided ID
    ) -> Optional[Dict[str, Any]]:
        """
        Place a market order.
        Returns a dictionary containing order details (id, status, etc.) or None if failed.
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific order by its ID.
        Returns a dictionary with order status details or None if an error occurs.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order by its ID.
        Returns True if cancellation was successful or accepted, False otherwise.
        """
        pass

    # --- Helper/Utility methods (Optional but useful) ---

    def format_symbol_for_broker(self, symbol: str) -> str:
        """
        Convert a standard symbol (e.g., 'BTC-USD') to the format required by the specific broker API.
        Default implementation assumes the broker uses the standard format.
        """
        return symbol

    def format_symbol_from_broker(self, broker_symbol: Any) -> str:
        """
        Convert a symbol received from the broker API back to the standard format (e.g., 'BTC-USD').
        Default implementation assumes the broker returns the standard format.
        """
        return str(broker_symbol)
