#!/usr/bin/env python3
"""
GreenGekko Trading Strategies for Robinhood Crypto Bot

This module implements advanced trading strategies from GreenGekko
for use with the Robinhood Cryptocurrency Trading Bot.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List
import pandas_ta as ta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gekko_strategies.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gekko_strategies")

class GekkoStrategies:
    """
    Implementation of GreenGekko trading strategies for the Robinhood Crypto Bot.
    """
    
    def __init__(self):
        """Initialize the GekkoStrategies class."""
        # Store trend information for each strategy and symbol
        self.trends = {}
        
    def initialize_trend(self, strategy: str, symbol: str) -> None:
        """
        Initialize trend tracking for a strategy and symbol.
        
        Args:
            strategy: The strategy name
            symbol: The cryptocurrency symbol
        """
        if strategy not in self.trends:
            self.trends[strategy] = {}
            
        self.trends[strategy][symbol] = {
            "direction": "none",
            "duration": 0,
            "persisted": False,
            "adviced": False
        }
        
    def get_trend(self, strategy: str, symbol: str) -> Dict:
        """
        Get current trend for a strategy and symbol.
        
        Args:
            strategy: The strategy name
            symbol: The cryptocurrency symbol
            
        Returns:
            Dict containing trend information
        """
        if strategy not in self.trends or symbol not in self.trends[strategy]:
            self.initialize_trend(strategy, symbol)
            
        return self.trends[strategy][symbol]
    
    def update_trend(self, strategy: str, symbol: str, direction: str, persistence: int = 1) -> Dict:
        """
        Update trend information for a strategy and symbol.
        
        Args:
            strategy: The strategy name
            symbol: The cryptocurrency symbol
            direction: The trend direction ('up', 'down', 'high', 'low', or 'none')
            persistence: Number of candles required for trend to persist
            
        Returns:
            Updated trend information and trading advice
        """
        trend = self.get_trend(strategy, symbol)
        advice = "hold"
        
        # If direction changed, reset the trend
        if trend["direction"] != direction:
            trend = {
                "direction": direction,
                "duration": 0,
                "persisted": False,
                "adviced": False
            }
            self.trends[strategy][symbol] = trend
        
        # Increment duration
        trend["duration"] += 1
        
        # Check if trend has persisted long enough
        if trend["duration"] >= persistence:
            trend["persisted"] = True
            
        # Generate advice if trend has persisted and we haven't advised yet
        if trend["persisted"] and not trend["adviced"]:
            trend["adviced"] = True
            
            if direction in ["up", "low"]:
                advice = "buy"
            elif direction in ["down", "high"]:
                advice = "sell"
                
        # Update the stored trend
        self.trends[strategy][symbol] = trend
        
        return {
            "trend": trend,
            "advice": advice
        }
    
    def reset_advice(self, strategy: str, symbol: str) -> None:
        """
        Reset the advised flag for a trend.
        
        Args:
            strategy: The strategy name
            symbol: The cryptocurrency symbol
        """
        if strategy in self.trends and symbol in self.trends[strategy]:
            self.trends[strategy][symbol]["adviced"] = False
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data: Price data series
            period: EMA period
            
        Returns:
            Series containing EMA values
        """
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_macd(self, data: pd.DataFrame, short_period: int = 12, 
                      long_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price data DataFrame
            short_period: Short EMA period
            long_period: Long EMA period
            signal_period: Signal EMA period
            
        Returns:
            Tuple of (MACD line, Signal line, MACD histogram)
        """
        # Calculate EMAs
        short_ema = self.calculate_ema(data['close'], short_period)
        long_ema = self.calculate_ema(data['close'], long_period)
        
        # Calculate MACD line
        macd_line = short_ema - long_ema
        
        # Calculate Signal line
        signal_line = self.calculate_ema(macd_line, signal_period)
        
        # Calculate MACD histogram
        macd_histogram = macd_line - signal_line
        
        return macd_line, signal_line, macd_histogram
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index).
        
        Args:
            data: Price data DataFrame
            period: RSI period
            
        Returns:
            Series containing RSI values
        """
        delta = data['close'].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = self.calculate_ema(gains, period)
        avg_losses = self.calculate_ema(losses, period)
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses.replace(0, 0.00001)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_stochastic_rsi(self, data: pd.DataFrame, rsi_period: int = 14, 
                               stoch_period: int = 14, k_period: int = 3, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic RSI.
        
        Args:
            data: Price data DataFrame
            rsi_period: Period for RSI calculation
            stoch_period: Period for Stochastic calculation
            k_period: %K smoothing period
            d_period: %D smoothing period
            
        Returns:
            Tuple of (%K line, %D line)
        """
        rsi = self.calculate_rsi(data, rsi_period)
        
        # Calculate Stoch RSI
        min_rsi = rsi.rolling(window=stoch_period).min()
        max_rsi = rsi.rolling(window=stoch_period).max()
        
        stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, 0.00001) # Avoid division by zero
        
        # Calculate %K and %D
        k_line = stoch_rsi.rolling(window=k_period).mean() * 100
        d_line = k_line.rolling(window=d_period).mean()
        
        return k_line, d_line

    def run_macd_strategy(self, df: pd.DataFrame, symbol: str, 
                          short_period: int = 12, long_period: int = 26, 
                          signal_period: int = 9, persistence: int = 1) -> str:
        """
        Runs the MACD strategy and returns advice.

        Args:
            df: Price data DataFrame
            symbol: Cryptocurrency symbol
            short_period: Short EMA period
            long_period: Long EMA period
            signal_period: Signal EMA period
            persistence: Candles needed for trend persistence

        Returns:
            Trading advice ("buy", "sell", "hold")
        """
        try:
            macd_line, signal_line, _ = self.calculate_macd(df, short_period, long_period, signal_period)
            
            # Ensure we have enough data
            if macd_line.isna().all() or signal_line.isna().all() or len(macd_line) < 2:
                logger.debug(f"MACD ({symbol}): Not enough data for signal.")
                return "hold"

            # Get latest values
            last_macd = macd_line.iloc[-1]
            last_signal = signal_line.iloc[-1]
            prev_macd = macd_line.iloc[-2]
            prev_signal = signal_line.iloc[-2]
            
            current_trend = "none"
            # Bullish crossover
            if last_macd > last_signal and prev_macd <= prev_signal:
                current_trend = "up"
            # Bearish crossover
            elif last_macd < last_signal and prev_macd >= prev_signal:
                current_trend = "down"

            # Update trend and get advice
            trend_info = self.update_trend(strategy="MACD", symbol=symbol, 
                                           direction=current_trend, persistence=persistence)
            
            logger.debug(f"MACD ({symbol}): Trend={trend_info['trend']}, Advice={trend_info['advice']}")
            return trend_info["advice"]
            
        except Exception as e:
            logger.exception(f"Error running MACD strategy for {symbol}: {e}")
            return "hold"
        
    def gekko_macd_strategy(self, data: pd.DataFrame, 
                          short_period: int = 12, 
                          long_period: int = 26, 
                          signal_period: int = 9,
                          threshold_up: float = 0.025,
                          threshold_down: float = -0.025,
                          persistence: int = 1,
                          symbol: str = "BTC") -> str:
        """
        GreenGekko MACD strategy implementation.
        
        Args:
            data: Price data DataFrame
            short_period: Short EMA period
            long_period: Long EMA period
            signal_period: Signal EMA period
            threshold_up: Threshold for uptrend
            threshold_down: Threshold for downtrend
            persistence: Number of candles required for trend to persist
            symbol: Cryptocurrency symbol
            
        Returns:
            Trading advice: 'buy', 'sell', or 'hold'
        """
        # Calculate MACD
        _, _, macd_histogram = self.calculate_macd(
            data, short_period, long_period, signal_period
        )
        
        # Get the latest MACD histogram value
        latest_macd = macd_histogram.iloc[-1]
        
        # Determine trend direction
        if latest_macd > threshold_up:
            direction = "up"
        elif latest_macd < threshold_down:
            direction = "down"
        else:
            direction = "none"
            
        # Update trend and get advice
        result = self.update_trend("macd", symbol, direction, persistence)
        
        logger.info(f"MACD Strategy - Symbol: {symbol}, MACD: {latest_macd:.6f}, " +
                   f"Direction: {direction}, Duration: {result['trend']['duration']}, " +
                   f"Advice: {result['advice']}")
        
        return result['advice']
    
    def gekko_rsi_strategy(self, data: pd.DataFrame,
                         period: int = 14,
                         threshold_low: float = 30,
                         threshold_high: float = 70,
                         persistence: int = 2,
                         symbol: str = "BTC") -> str:
        """
        GreenGekko RSI strategy implementation.
        
        Args:
            data: Price data DataFrame
            period: RSI period
            threshold_low: Oversold threshold
            threshold_high: Overbought threshold
            persistence: Number of candles required for trend to persist
            symbol: Cryptocurrency symbol
            
        Returns:
            Trading advice: 'buy', 'sell', or 'hold'
        """
        # Calculate RSI
        rsi = self.calculate_rsi(data, period)
        
        # Get the latest RSI value
        latest_rsi = rsi.iloc[-1]
        
        # Determine trend direction
        if latest_rsi > threshold_high:
            direction = "high"
        elif latest_rsi < threshold_low:
            direction = "low"
        else:
            direction = "none"
            
        # Update trend and get advice
        result = self.update_trend("rsi", symbol, direction, persistence)
        
        logger.info(f"RSI Strategy - Symbol: {symbol}, RSI: {latest_rsi:.2f}, " +
                   f"Direction: {direction}, Duration: {result['trend']['duration']}, " +
                   f"Advice: {result['advice']}")
        
        return result['advice']
    
    def gekko_stoch_rsi_strategy(self, data: pd.DataFrame,
                               rsi_period: int = 14,
                               stoch_period: int = 14,
                               k_period: int = 3,
                               d_period: int = 3,
                               threshold_low: float = 20,
                               threshold_high: float = 80,
                               persistence: int = 1,
                               symbol: str = "BTC") -> str:
        """
        GreenGekko Stochastic RSI strategy implementation.
        
        Args:
            data: Price data DataFrame
            rsi_period: Period for RSI calculation
            stoch_period: Period for Stochastic calculation
            k_period: %K smoothing period
            d_period: %D smoothing period
            threshold_low: Oversold threshold
            threshold_high: Overbought threshold
            persistence: Number of candles required for trend to persist
            symbol: Cryptocurrency symbol
            
        Returns:
            Trading advice: 'buy', 'sell', or 'hold'
        """
        # Calculate Stochastic RSI
        k, d = self.calculate_stochastic_rsi(
            data, rsi_period, stoch_period, k_period, d_period
        )
        
        # Get the latest values
        latest_k = k.iloc[-1]
        latest_d = d.iloc[-1]
        
        # Determine trend direction
        if latest_k > threshold_high and latest_d > threshold_high:
            direction = "high"
        elif latest_k < threshold_low and latest_d < threshold_low:
            direction = "low"
        else:
            direction = "none"
            
        # Update trend and get advice
        result = self.update_trend("stoch_rsi", symbol, direction, persistence)
        
        logger.info(f"Stochastic RSI Strategy - Symbol: {symbol}, %K: {latest_k:.2f}, %D: {latest_d:.2f}, " +
                   f"Direction: {direction}, Duration: {result['trend']['duration']}, " +
                   f"Advice: {result['advice']}")
        
        return result['advice']
    
    def gekko_bollinger_bands_strategy(self, data: pd.DataFrame,
                                     period: int = 20,
                                     std_dev: float = 2.0,
                                     persistence: int = 1,
                                     symbol: str = "BTC") -> str:
        """
        GreenGekko Bollinger Bands strategy implementation.
        
        Args:
            data: Price data DataFrame
            period: Moving average period
            std_dev: Number of standard deviations
            persistence: Number of candles required for trend to persist
            symbol: Cryptocurrency symbol
            
        Returns:
            Trading advice: 'buy', 'sell', or 'hold'
        """
        # Calculate Bollinger Bands
        middle_band, upper_band, lower_band = self.calculate_bollinger_bands(
            data, period, std_dev
        )
        
        # Get the latest values
        latest_close = data['close'].iloc[-1]
        latest_upper = upper_band.iloc[-1]
        latest_lower = lower_band.iloc[-1]
        latest_middle = middle_band.iloc[-1]
        
        # Determine trend direction
        if latest_close < latest_lower:
            direction = "low"  # Price below lower band - potential buy
        elif latest_close > latest_upper:
            direction = "high"  # Price above upper band - potential sell
        else:
            direction = "none"
            
        # Update trend and get advice
        result = self.update_trend("bollinger", symbol, direction, persistence)
        
        logger.info(f"Bollinger Bands Strategy - Symbol: {symbol}, Close: {latest_close:.2f}, " +
                   f"Upper: {latest_upper:.2f}, Lower: {latest_lower:.2f}, " +
                   f"Direction: {direction}, Duration: {result['trend']['duration']}, " +
                   f"Advice: {result['advice']}")
        
        return result['advice']
    
    def gekko_multi_strategy(self, data: pd.DataFrame, symbol: str = "BTC") -> str:
        """
        GreenGekko Multi-Strategy implementation that combines multiple strategies.
        
        Args:
            data: Price data DataFrame
            symbol: Cryptocurrency symbol
            
        Returns:
            Trading advice: 'buy', 'sell', or 'hold'
        """
        # Get advice from each strategy
        macd_advice = self.gekko_macd_strategy(data, symbol=symbol)
        rsi_advice = self.gekko_rsi_strategy(data, symbol=symbol)
        bb_advice = self.gekko_bollinger_bands_strategy(data, symbol=symbol)
        
        # Count buy and sell signals
        buy_count = sum(1 for advice in [macd_advice, rsi_advice, bb_advice] if advice == "buy")
        sell_count = sum(1 for advice in [macd_advice, rsi_advice, bb_advice] if advice == "sell")
        
        # Make decision based on majority vote
        if buy_count >= 2:
            final_advice = "buy"
        elif sell_count >= 2:
            final_advice = "sell"
        else:
            final_advice = "hold"
            
        logger.info(f"Multi-Strategy - Symbol: {symbol}, " +
                   f"MACD: {macd_advice}, RSI: {rsi_advice}, BB: {bb_advice}, " +
                   f"Final Advice: {final_advice}")
        
        return final_advice
    
    def analyze_with_strategy(self, data: pd.DataFrame, strategy: str, symbol: str) -> str:
        """
        Analyze price data with the specified GreenGekko strategy.
        
        Args:
            data: Price data DataFrame
            strategy: Strategy name ('macd', 'rsi', 'stoch_rsi', 'bollinger', 'multi')
            symbol: Cryptocurrency symbol
            
        Returns:
            Trading advice: 'buy', 'sell', or 'hold'
        """
        if strategy == "macd":
            return self.gekko_macd_strategy(data, symbol=symbol)
        elif strategy == "rsi":
            return self.gekko_rsi_strategy(data, symbol=symbol)
        elif strategy == "stoch_rsi":
            return self.gekko_stoch_rsi_strategy(data, symbol=symbol)
        elif strategy == "bollinger":
            return self.gekko_bollinger_bands_strategy(data, symbol=symbol)
        elif strategy == "multi":
            return self.gekko_multi_strategy(data, symbol=symbol)
        else:
            logger.warning(f"Unknown strategy: {strategy}. Defaulting to RSI.")
            return self.gekko_rsi_strategy(data, symbol=symbol)


class BaseStrategy:
    def __init__(self, params: dict = None):
        """Initialize the strategy with specific parameters."""
        self.params = params or {}
        self.name = "BaseStrategy"

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate the required technical indicators."""
        raise NotImplementedError("Subclasses should implement this method.")

    def check_signal(self, data: pd.DataFrame) -> str:
        """
        Checks for buy/sell signals based on the calculated indicators.
        Returns 'buy', 'sell', or 'hold'.
        """
        if data is None or data.empty or len(data) < 2:
            return "hold" # Not enough data
            
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # --- Define required columns ---
        required_squeeze_cols = ['TTM_SQUEEZE_ON', 'TTM_SQUEEZE_MOM']
        required_stoch_cols = ['STOCHk_14_3_3', 'STOCHd_14_3_3']
        required_wave_cols = ['TTM_WAVE1', 'TTM_WAVE2']
        
        # --- Check if required columns exist ---
        if not all(col in latest.index for col in required_squeeze_cols) or \
           not all(col in latest.index for col in required_stoch_cols) or \
           not all(col in latest.index for col in required_wave_cols):
            # logger.warning(f"Strategy {self.name}: Missing required indicator columns in latest data. Holding.") # Optional logging
            return "hold"
            
        # --- TTM Squeeze Logic ---
        try:
            squeeze_off_prev = prev['TTM_SQUEEZE_ON'] == 1
            squeeze_off_latest = latest['TTM_SQUEEZE_ON'] == 0
            positive_mom = latest['TTM_SQUEEZE_MOM'] > 0
            negative_mom = latest['TTM_SQUEEZE_MOM'] < 0
            
            squeeze_fires_bullish = squeeze_off_prev and squeeze_off_latest and positive_mom
            squeeze_fires_bearish = squeeze_off_prev and squeeze_off_latest and negative_mom
        except KeyError as e:
            # logger.warning(f"Strategy {self.name}: KeyError accessing TTM Squeeze columns: {e}. Holding.") # Optional logging
            return "hold"
            
        # --- Stochastic Logic ---
        try:
            stoch_k = latest['STOCHk_14_3_3']
            stoch_d = latest['STOCHd_14_3_3']
            stoch_k_prev = prev['STOCHk_14_3_3']
            stoch_d_prev = prev['STOCHd_14_3_3']
            
            stoch_bullish_cross = stoch_k_prev < stoch_d_prev and stoch_k > stoch_d and stoch_k < self.stoch_overbought
            stoch_bearish_cross = stoch_k_prev > stoch_d_prev and stoch_k < stoch_d and stoch_k > self.stoch_oversold
        except KeyError as e:
            # logger.warning(f"Strategy {self.name}: KeyError accessing Stochastic columns: {e}. Holding.") # Optional logging
            return "hold"

        # --- TTM Wave Logic ---
        try:
            wave1 = latest['TTM_WAVE1']
            wave2 = latest['TTM_WAVE2']
            wave_positive = wave1 > 0 and wave2 > 0
            wave_negative = wave1 < 0 and wave2 < 0
        except KeyError as e:
            # logger.warning(f"Strategy {self.name}: KeyError accessing TTM Wave columns: {e}. Holding.") # Optional logging
            return "hold"
        
        # --- Combine Signals ---
        buy_signal = squeeze_fires_bullish and stoch_bullish_cross and wave_positive
        sell_signal = squeeze_fires_bearish and stoch_bearish_cross and wave_negative

        if buy_signal:
            return "buy"
        elif sell_signal:
            return "sell"
        else:
            return "hold"


class TTMScalperStochWaveStrategy(BaseStrategy):
    """
    Combines TTM Scalper concepts (via TTM Squeeze), Stochastic, and TTM Wave (EMA difference).
    
    Strategy Logic (Example):
    - Buy: TTM Squeeze fires bullish (squeeze off, momentum positive) AND Stochastic K crosses above 20 AND TTM Wave is positive.
    - Sell: TTM Squeeze fires bearish (squeeze off, momentum negative) AND Stochastic K crosses below 80 AND TTM Wave is negative.
    """
    def __init__(self, params: dict = None):
        """Initialize the strategy with specific parameters."""
        default_params = {
            'squeeze_length': 20,  # Length for BBands/KC for Squeeze
            'squeeze_mult': 2.0,   # Multiplier for BBands
            'kc_mult': 1.5,        # Multiplier for Keltner Channels
            'stoch_k': 14,         # Stochastic %K period
            'stoch_d': 3,          # Stochastic %D period (SMA of %K)
            'stoch_smooth_k': 3,   # Smoothing for %K
            'wave_fast_ema': 8,    # Fast EMA for TTM Wave
            'wave_slow_ema': 21    # Slow EMA for TTM Wave
        }
        self.params = {**default_params, **(params or {})}
        self.name = "TTM Scalper/Stoch/Wave"

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate the required technical indicators."""
        if data.empty:
            return data
        
        # Ensure columns are named correctly for pandas_ta (lowercase)
        data.columns = [col.lower() for col in data.columns]
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain {required_cols}")
            
        # Calculate TTM Squeeze
        # Uses Bollinger Bands (BB) and Keltner Channels (KC)
        # Columns added: SQZ_20_2.0_1.5, SQZ_ON, SQZ_OFF, SQZ_NO
        # Also adds momentum histogram: SQZ_20_2.0_1.5_HIST
        data.ta.squeeze(bb_length=self.params['squeeze_length'], 
                         bb_std=self.params['squeeze_mult'], 
                         kc_length=self.params['squeeze_length'], 
                         kc_scalar=self.params['kc_mult'], 
                         append=True)
        
        # Calculate Stochastic
        # Columns added: STOCHk_14_3_3, STOCHd_14_3_3
        data.ta.stoch(k=self.params['stoch_k'], 
                      d=self.params['stoch_d'], 
                      smooth_k=self.params['stoch_smooth_k'], 
                      append=True)
        
        # Calculate TTM Wave (Fast EMA - Slow EMA)
        fast_ema = data.ta.ema(length=self.params['wave_fast_ema'], append=False)
        slow_ema = data.ta.ema(length=self.params['wave_slow_ema'], append=False)
        data['TTM_WAVE'] = fast_ema - slow_ema
        
        # Rename columns for clarity if needed (optional, but good practice)
        squeeze_col = f'SQZ_{self.params["squeeze_length"]}_{self.params["squeeze_mult"]}_{self.params["kc_mult"]}'
        squeeze_hist_col = f'{squeeze_col}_HIST'
        stoch_k_col = f'STOCHk_{self.params["stoch_k"]}_{self.params["stoch_d"]}_{self.params["stoch_smooth_k"]}'
        stoch_d_col = f'STOCHd_{self.params["stoch_k"]}_{self.params["stoch_d"]}_{self.params["stoch_smooth_k"]}'
        
        data.rename(columns={
            'SQZ_ON': 'TTM_SQUEEZE_ON',
            squeeze_hist_col: 'TTM_SQUEEZE_MOM', # Momentum Histogram
            stoch_k_col: 'STOCH_K',
            stoch_d_col: 'STOCH_D'
        }, inplace=True)
        
        return data

    def check_signal(self, data: pd.DataFrame) -> str:
        """
        Checks for buy/sell signals based on the calculated indicators.
        Returns 'buy', 'sell', or 'hold'.
        """
        if data is None or data.empty or len(data) < 2:
            return "hold" # Not enough data
            
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # --- Define required columns ---
        required_squeeze_cols = ['TTM_SQUEEZE_ON', 'TTM_SQUEEZE_MOM']
        required_stoch_cols = ['STOCHk_14_3_3', 'STOCHd_14_3_3']
        required_wave_cols = ['TTM_WAVE']
        
        # --- Check if required columns exist ---
        if not all(col in latest.index for col in required_squeeze_cols) or \
           not all(col in latest.index for col in required_stoch_cols) or \
           not all(col in latest.index for col in required_wave_cols):
            # logger.warning(f"Strategy {self.name}: Missing required indicator columns in latest data. Holding.") # Optional logging
            return "hold"
            
        # --- TTM Squeeze Logic ---
        try:
            squeeze_off_prev = prev['TTM_SQUEEZE_ON'] == 1
            squeeze_off_latest = latest['TTM_SQUEEZE_ON'] == 0
            positive_mom = latest['TTM_SQUEEZE_MOM'] > 0
            negative_mom = latest['TTM_SQUEEZE_MOM'] < 0
            
            squeeze_fires_bullish = squeeze_off_prev and squeeze_off_latest and positive_mom
            squeeze_fires_bearish = squeeze_off_prev and squeeze_off_latest and negative_mom
        except KeyError as e:
            # logger.warning(f"Strategy {self.name}: KeyError accessing TTM Squeeze columns: {e}. Holding.") # Optional logging
            return "hold"
            
        # --- Stochastic Logic ---
        try:
            stoch_k = latest['STOCHk_14_3_3']
            stoch_d = latest['STOCHd_14_3_3']
            stoch_k_prev = prev['STOCHk_14_3_3']
            stoch_d_prev = prev['STOCHd_14_3_3']
            
            stoch_bullish_cross = stoch_k_prev < stoch_d_prev and stoch_k > stoch_d and stoch_k < 80
            stoch_bearish_cross = stoch_k_prev > stoch_d_prev and stoch_k < stoch_d and stoch_k > 20
        except KeyError as e:
            # logger.warning(f"Strategy {self.name}: KeyError accessing Stochastic columns: {e}. Holding.") # Optional logging
            return "hold"

        # --- TTM Wave Logic ---
        try:
            wave = latest['TTM_WAVE']
            wave_positive = wave > 0
            wave_negative = wave < 0
        except KeyError as e:
            # logger.warning(f"Strategy {self.name}: KeyError accessing TTM Wave columns: {e}. Holding.") # Optional logging
            return "hold"
        
        # --- Combine Signals ---
        buy_signal = squeeze_fires_bullish and stoch_bullish_cross and wave_positive
        sell_signal = squeeze_fires_bearish and stoch_bearish_cross and wave_negative

        if buy_signal:
            return "buy"
        elif sell_signal:
            return "sell"
        else:
            return "hold"


class RSIStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        """Initialize the strategy with specific parameters."""
        default_params = {
            'rsi_period': 14,
            'threshold_low': 30,
            'threshold_high': 70
        }
        self.params = {**default_params, **(params or {})}
        self.name = "RSI"

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate the required technical indicators."""
        if data.empty:
            return data
        
        # Calculate RSI
        data['RSI'] = ta.rsi(data['close'], length=self.params['rsi_period'])
        
        return data

    def check_signal(self, data: pd.DataFrame) -> str:
        """
        Checks for buy/sell signals based on the calculated indicators.
        Returns 'buy', 'sell', or 'hold'.
        """
        if data is None or data.empty or len(data) < 2:
            return "hold" # Not enough data
            
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # --- Define required columns ---
        required_rsi_cols = ['RSI']
        
        # --- Check if required columns exist ---
        if not all(col in latest.index for col in required_rsi_cols):
            # logger.warning(f"Strategy {self.name}: Missing required indicator columns in latest data. Holding.") # Optional logging
            return "hold"
            
        # --- RSI Logic ---
        try:
            rsi = latest['RSI']
            rsi_prev = prev['RSI']
            
            rsi_bullish_cross = rsi_prev < self.params['threshold_low'] and rsi > self.params['threshold_low']
            rsi_bearish_cross = rsi_prev > self.params['threshold_high'] and rsi < self.params['threshold_high']
        except KeyError as e:
            # logger.warning(f"Strategy {self.name}: KeyError accessing RSI columns: {e}. Holding.") # Optional logging
            return "hold"
        
        # --- Combine Signals ---
        buy_signal = rsi_bullish_cross
        sell_signal = rsi_bearish_cross

        if buy_signal:
            return "buy"
        elif sell_signal:
            return "sell"
        else:
            return "hold"


# Example usage:
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=50, freq='D')
    data = {
        'timestamp': dates,
        'open': np.random.normal(100, 10, 50),
        'high': np.random.normal(105, 10, 50),
        'low': np.random.normal(95, 10, 50),
        'close': np.random.normal(100, 10, 50),
        'volume': np.random.normal(1000000, 100000, 50)
    }
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # Initialize strategies
    strategies = GekkoStrategies()
    
    # Test each strategy
    print("MACD Strategy:", strategies.gekko_macd_strategy(df))
    print("RSI Strategy:", strategies.gekko_rsi_strategy(df))
    print("Stochastic RSI Strategy:", strategies.gekko_stoch_rsi_strategy(df))
    print("Bollinger Bands Strategy:", strategies.gekko_bollinger_bands_strategy(df))
    print("Multi-Strategy:", strategies.gekko_multi_strategy(df))

    ttm_strategy = TTMScalperStochWaveStrategy()
    ttm_data = ttm_strategy.calculate_indicators(df)
    print("TTM Scalper/Stoch/Wave Strategy:", ttm_strategy.check_signal(ttm_data))

    rsi_strategy = RSIStrategy()
    rsi_data = rsi_strategy.calculate_indicators(df)
    print("RSI Strategy:", rsi_strategy.check_signal(rsi_data))
