import pandas as pd
import logging
from typing import Optional, Dict, Any
from .base_strategy import BaseStrategy

class MACDStrategy(BaseStrategy):
    """Implements a basic MACD crossover strategy."""

    def _validate_params(self):
        """Validate parameters specific to MACD."""
        # Example validation (can be expanded)
        self.fast = self.params.get('macd_fast', 12)
        self.slow = self.params.get('macd_slow', 26)
        self.sign = self.params.get('macd_sign', 9)
        logging.info(f"MACDStrategy initialized with params: fast={self.fast}, slow={self.slow}, sign={self.sign}")

    def generate_signal(self, data: pd.DataFrame) -> Optional[str]:
        """Generates buy/sell signals based on MACD crossover."""
        if data.empty or 'macd' not in data.columns or 'macd_signal' not in data.columns:
            logging.warning("MACDStrategy: Data empty or missing 'macd'/'macd_signal' columns.")
            return 'hold' # Default to hold if required data is missing

        try:
            # Get the last two rows to check for crossover
            last_row = data.iloc[-1]
            prev_row = data.iloc[-2]

            # Check for bullish crossover (MACD crosses above signal line)
            if prev_row['macd'] < prev_row['macd_signal'] and last_row['macd'] > last_row['macd_signal']:
                logging.debug(f"MACD Bullish Crossover detected on {last_row.name}")
                return 'buy'

            # Check for bearish crossover (MACD crosses below signal line)
            elif prev_row['macd'] > prev_row['macd_signal'] and last_row['macd'] < last_row['macd_signal']:
                logging.debug(f"MACD Bearish Crossover detected on {last_row.name}")
                return 'sell'

            else:
                return 'hold' # No crossover

        except IndexError:
             logging.warning("MACDStrategy: Not enough data points (< 2) to detect crossover.")
             return 'hold'
        except Exception as e:
             logging.error(f"MACDStrategy: Error generating signal: {e}", exc_info=True)
             return 'hold' # Default to hold on error
