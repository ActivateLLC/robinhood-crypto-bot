from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Dict, Any

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initializes the strategy with optional parameters."""
        self.params = params if params is not None else {}
        self._validate_params()

    def _validate_params(self):
        """Validates parameters specific to the strategy. Override in subclasses."""
        pass # Default implementation does nothing

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Optional[str]:
        """
        Generates a trading signal ('buy', 'sell', or 'hold') based on input data.

        Args:
            data: DataFrame containing historical price data and indicators.

        Returns:
            A string signal ('buy', 'sell', 'hold') or None if no signal.
        """
        pass
