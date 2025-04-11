import pandas as pd
import logging
from typing import Optional, Dict, Any
from .base_strategy import BaseStrategy

class RSIStrategy(BaseStrategy):
    """Implements a basic RSI overbought/oversold strategy."""

    def _validate_params(self):
        """Validate parameters specific to RSI."""
        self.oversold_threshold = self.params.get('rsi_oversold', 30)
        self.overbought_threshold = self.params.get('rsi_overbought', 70)
        self.window = self.params.get('rsi_window', 14) # Window used for RSI calculation
        logging.info(f"RSIStrategy initialized with params: oversold={self.oversold_threshold}, overbought={self.overbought_threshold}, window={self.window}")

    def generate_signal(self, data: pd.DataFrame) -> Optional[str]:
        """Generates buy/sell signals based on RSI levels."""
        if data.empty or 'rsi' not in data.columns:
            logging.warning("RSIStrategy: Data empty or missing 'rsi' column.")
            return 'hold' # Default to hold if required data is missing

        try:
            last_rsi = data['rsi'].iloc[-1]

            if last_rsi < self.oversold_threshold:
                logging.debug(f"RSI Oversold ({last_rsi:.2f} < {self.oversold_threshold}) detected on {data.index[-1]}")
                return 'buy'
            elif last_rsi > self.overbought_threshold:
                logging.debug(f"RSI Overbought ({last_rsi:.2f} > {self.overbought_threshold}) detected on {data.index[-1]}")
                return 'sell'
            else:
                return 'hold' # RSI is in the neutral zone

        except IndexError:
             logging.warning("RSIStrategy: Not enough data points (< 1) for RSI signal.")
             return 'hold'
        except Exception as e:
             logging.error(f"RSIStrategy: Error generating signal: {e}", exc_info=True)
             return 'hold' # Default to hold on error
