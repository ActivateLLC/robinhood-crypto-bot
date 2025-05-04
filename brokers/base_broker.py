import abc
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import datetime

# --- Data Structures ---
@dataclass
class Order:
    # Non-default fields first
    id: str  # Broker's unique order ID (e.g., permId for IBKR)
    symbol: str
    order_type: str  # e.g., 'market', 'limit'
    side: str  # e.g., 'buy', 'sell'
    quantity: float
    status: str # e.g., 'submitted', 'filled', 'cancelled', 'pending_submit', 'api_pending'
    
    # Fields with defaults follow
    client_order_id: Optional[str] = None # User-defined ID, if provided
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0 # Initial quantity - filled_quantity
    average_fill_price: Optional[float] = None
    commission: Optional[float] = None
    raw_status_details: Optional[Any] = None # Store the raw broker status object if needed

@dataclass
class Holding:
    symbol: str
    quantity: float
    average_cost: float
    current_value: Optional[float] = None
    market_price: Optional[float] = None # Add market price if available
    raw_details: Optional[Any] = None # Store raw broker position object

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime.datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[float] = None
    # Add other fields as needed (e.g., open, high, low, close for historical)
    raw_ticker: Optional[Any] = None # Store raw broker ticker object

# --- Broker Interface ---

class BrokerInterface(abc.ABC):
    """Abstract base class defining the common interface for broker interactions."""

    @abc.abstractmethod
    def connect(self, credentials: Dict[str, Any]) -> bool:
        """Connect to the broker's API using provided credentials."""
        pass

    @abc.abstractmethod
    def disconnect(self):
        """Disconnect from the broker's API."""
        pass

    @abc.abstractmethod
    def place_order(self, order_details: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order with the broker.

        Args:
            order_details: Dictionary containing order specifics like symbol, 
                           quantity, side (buy/sell), type (market/limit), etc.

        Returns:
            Dictionary containing the order confirmation or status.
        """
        pass

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order.

        Args:
            order_id: The unique identifier of the order to cancel.

        Returns:
            Dictionary containing the cancellation confirmation or status.
        """
        pass

    @abc.abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get the status of a specific order.

        Args:
            order_id: The unique identifier of the order.

        Returns:
            Dictionary containing the order status details.
        """
        pass

    @abc.abstractmethod
    def get_account_summary(self) -> Dict[str, Any]:
        """Retrieve account summary details (e.g., balances, equity)."""
        pass

    @abc.abstractmethod
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve current open positions, optionally filtered by symbol."""
        pass

    @abc.abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Retrieve historical market data for a given symbol and timeframe."""
        pass

    @abc.abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get the current market price for a given symbol."""
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Return True if currently connected to the broker, False otherwise."""
        pass

    # Consider changing return types to use the dataclasses above
    # e.g., get_order_status(self, order_id: str) -> Optional[Order]:
    # e.g., get_positions(...) -> List[Holding]:
    # e.g., get_current_price(...) -> Optional[MarketData]:
