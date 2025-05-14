from typing import (
    Any, 
    Dict, 
    Optional, 
    Tuple, 
    Union
)

import logging
import numpy as np
import pandas as pd
import gymnasium as gym
import ta # Add import for TA-Lib
from alt_crypto_data import AltCryptoDataProvider
from ta import *
try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn'])
    from sklearn.preprocessing import StandardScaler
from crypto_sentiment import CryptoSentimentAnalyzer

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class CryptoTradingEnv(gym.Env):  
    """Cryptocurrency Trading Environment that follows Gymnasium interface"""
    """
    A custom Gymnasium environment for Reinforcement Learning-based cryptocurrency trading.

    This environment uses historical OHLCV data, incorporates technical indicators,
    handles transaction fees, and provides rewards based on profit/loss with
    penalties for holding losers and buying into losing positions.

    Action Space (Discrete, 3 actions):
        - 0 = Hold
        - 1 = Buy
        - 2 = Sell

    Observation Space (Box):
        Flattened array containing a lookback window of:
        - Normalized OHLCV data
        - Normalized technical indicators (RSI, MACD signal, Bollinger Band width)
        - Normalized cash balance
        - Normalized crypto holdings
    """
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(
        self, 
        market_data=None,  
        symbol: str = 'BTC-USD', 
        initial_balance: float = 10000, 
        transaction_fee_percent: float = 0.001,
        data_period: str = '6mo', 
        data_interval: str = '1h',
        use_sentiment: bool = True,
        lookback_window: int = 30,
        **kwargs  
    ):
        """
        Initialize the cryptocurrency trading environment.
        
        Args:
            market_data (pd.DataFrame): Historical market data (default: None)
            symbol (str): Trading symbol (default: 'BTC-USD')
            initial_balance (float): Starting portfolio balance (default: 10000)
            transaction_fee_percent (float): Trading transaction fee (default: 0.1%)
            data_period (str): Historical data period (default: '6mo')
            data_interval (str): Data interval (default: '1h')
            use_sentiment (bool): Enable sentiment-based reward modification
            lookback_window (int): Number of historical steps to consider in state
        """
        super().__init__()
        
        # Validate inputs
        if initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        
        # Core trading parameters
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        
        # Data configuration
        self.data_period = data_period
        self.data_interval = data_interval
        
        # Sentiment Analysis Integration
        self.use_sentiment = use_sentiment
        self.sentiment_analyzer = CryptoSentimentAnalyzer([symbol.split('-')[0]])
        
        # Initialize data provider
        self.data_provider = AltCryptoDataProvider()
        
        # Logging setup
        self.logger = logging.getLogger('rl_environment')
        
        # Initialize technical indicators list. This must be done *before* _preprocess_data is called.
        self.technical_indicators = [
            'adi', 'obv', 'cmf', 'fi', 'em', 'nvi', 'atr', 
            'bb_bbm', 'bb_bbw', 'bb_bbp', 
            'kc_kcc', 
            'macd', 'macd_signal', 'macd_diff', 
            'adx', 
            'rsi', 
            'tsi', 
            'uo', 
            'stoch', 'stoch_signal', 
            'williams_r', 
            'ao', 
            'ema_10', 
            'sma_50', 
            'log_returns', 
            'squeeze_on', 
            'ttm_momentum'
        ]
        if len(self.technical_indicators) != 27:
            # This is a critical error, indicates a mismatch in design.
            self.logger.error(f"FATAL: Technical indicators list MUST contain 27 items for a 31-feature observation space. Found {len(self.technical_indicators)}. Check definition in __init__.")
            raise ValueError(f"Technical indicators list must contain 27 items. Found {len(self.technical_indicators)}.")

        # Load and prepare historical data
        if market_data is None:
            # Fetch default market data if not provided
            self.df_processed = self._load_and_prepare_data(
                symbol=self.symbol,
                period=self.data_period,
                interval=self.data_interval
            )
        else:
            self.df_processed = market_data
        
        # Validate loaded data
        self.df_processed = self._validate_data(self.df_processed)
        
        # Add technical indicators
        self.df_processed = self._preprocess_data(self.df_processed)
        
        # Trading state variables
        self.current_step = 0
        self.max_steps = len(self.df_processed) - 1
        
        # Portfolio tracking
        self.portfolio_balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.crypto_held = 0.0
        self.entry_price = 0.0
        
        # Portfolio history tracking
        self.portfolio_history = [float(self.initial_balance)]
        self.trade_history = []
        
        # Technical indicator tracking
        self.lookback_window = lookback_window  # Default lookback window
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        # Observation space: 27 technical indicators + 4 portfolio metrics
        # Technical indicators generated in _preprocess_data:
        self.portfolio_feature_count = 4 # balance, value, crypto_held, current_price (or normalized price)
        observation_feature_count = len(self.technical_indicators) + self.portfolio_feature_count # 27 + 4 = 31
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(observation_feature_count,), # Should be (31,)
            dtype=np.float32
        )
        
        # Reset environment to initial state
        self.reset()

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and preprocess market data for trading environment
        
        Args:
            df (pd.DataFrame): Input market data
        
        Returns:
            pd.DataFrame: Validated and processed market data
        """
        if df is None or df.empty:
            self.logger.error("Market data is None or empty after loading.")
            raise ValueError("Market data cannot be None or empty.")

        # Expected columns (use lowercase as per _load_and_prepare_data normalization)
        expected_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not expected_cols.issubset(df.columns):
            missing_cols = expected_cols - set(df.columns)
            self.logger.error(f"Missing expected columns: {missing_cols}. Available: {df.columns.tolist()}")
            raise ValueError(f"DataFrame must include {expected_cols}")

        # Ensure data types are correct
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle NaNs: forward fill then backfill for robustness
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # Drop any rows that still have NaNs in critical columns (should be rare after ffill/bfill)
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        
        if df.empty:
            self.logger.error("DataFrame became empty after NaN handling in _validate_data.")
            raise ValueError("DataFrame empty after NaN handling.")

        # Feature engineering for validation (example: moving averages)
        # Ensure these use lowercase 'close' as well
        if hasattr(self, 'config') and hasattr(self.config, 'MOVING_AVERAGES'):
            for col, window in self.config.MOVING_AVERAGES.items():
                # Ensure the source column for rolling operations is 'close' (lowercase)
                df[col] = df['close'].rolling(window=window).mean()
        
        # Re-fill NaNs that might have been introduced by rolling operations
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        self.logger.info("Data validation and initial processing complete.")
        return df

    def _load_and_prepare_data(
        self, 
        symbol: str, 
        period: str = '6mo', 
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        Load and prepare historical price data for the trading environment.
        
        Args:
            symbol (str): Trading symbol to fetch data for
            period (str): Historical data period (default: '6mo')
            interval (str): Data interval (default: '1h')
        
        Returns:
            pd.DataFrame: Processed historical price data
        """
        # Validate inputs
        if not symbol:
            raise ValueError("Symbol must be provided")
        
        # Use data provider to fetch historical data
        try:
            # Fetch historical data
            df = self.data_provider.fetch_all_historical_data(
                symbol=symbol, 
                interval=interval, 
                days=180  # Consistent with 6-month period
            )
            
            # Validate data
            if df is None or len(df) < 1000:
                raise ValueError(f"Insufficient data for {symbol}")
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.date_range(
                    start=pd.Timestamp.now() - pd.Timedelta(days=180), 
                    periods=len(df), 
                    freq=interval
                )
            
            # Final NaN Handling
            df.ffill(inplace=True)
            df.fillna(0, inplace=True)
            
            self.logger.debug(f"_load_and_prepare_data returning df with columns: {df.columns.tolist() if df is not None else 'None'}")
            return df
        except Exception as e:
            # Log and re-raise with context
            self.logger.error(f"Data loading error for {symbol}: {e}")
            raise

    def _preprocess_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced preprocessing of raw price data with comprehensive technical indicators
        
        Args:
            df_raw (pd.DataFrame): Raw price data from data provider, 
                                   assumed to have lowercase columns ('open', 'high', 'low', 'close', 'volume')
        
        Returns:
            pd.DataFrame: Processed dataframe with extensive technical indicators
        """
        self.logger.info(f"Starting data preprocessing for {self.symbol}...")
        # Use a copy to avoid SettingWithCopyWarning
        df = df_raw.copy()

        # Ensure necessary columns are present
        required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        for col in required_ohlcv:
            if col not in df.columns:
                self.logger.error(f"Missing required column {col} in raw data for preprocessing.")
                # Fill with 0 or handle error appropriately, for now, raise error if critical
                raise ValueError(f"Missing critical OHLCV column: {col}")

        # Calculate Log Returns robustly
        # Replace 0s with a very small number to avoid log(0) or division by zero
        safe_close = df['close'].replace(0, 1e-9) # 1e-9 is a tiny number
        safe_close_shifted = safe_close.shift(1).replace(0, 1e-9)
        df['log_returns'] = np.log(safe_close / safe_close_shifted)

        # Calculate Heikin-Ashi candlesticks
        df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['ha_open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2 # Note: First ha_open will be NaN
        df.loc[df.index[0], 'ha_open'] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2 # Initialize first ha_open
        df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
        df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)

        # TTM Squeeze specific features (ensure column names are correct)
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        kc = ta.volatility.KeltnerChannel(high=df['high'], low=df['low'], close=df['close'], window=20, window_atr=1.5)
        df['bb_upperband'] = bb.bollinger_hband()
        df['bb_lowerband'] = bb.bollinger_lband()
        df['kc_upperband'] = kc.keltner_channel_hband()
        df['kc_lowerband'] = kc.keltner_channel_lband()
        df['squeeze_on'] = (df['bb_lowerband'] > df['kc_lowerband']) & (df['bb_upperband'] < df['kc_upperband'])
        df['squeeze_off'] = (df['bb_lowerband'] < df['kc_lowerband']) & (df['bb_upperband'] > df['kc_upperband'])
        # Momentum for TTM - using a simple 12-period momentum for 'close'
        df['ttm_momentum'] = df['close'].diff(12)

        # Features from config.py (ensure consistency)
        if hasattr(self, 'config') and self.config:
            feature_config = self.config.RL_AGENT_FEATURES
            # ... (ensure any df access here uses lowercase columns)

        # --- Clean all generated indicator columns for NaNs and Infs ---
        # List of all columns that are considered technical indicators based on self.technical_indicators
        # This ensures we only clean the ones we intend to use.
        indicator_cols_to_clean = self.technical_indicators # these are defined in __init__
        
        for col in indicator_cols_to_clean:
            if col in df.columns:
                df[col] = np.nan_to_num(df[col], nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
            else:
                # This case should ideally not happen if technical_indicators list is correct
                self.logger.warning(f"Column {col} specified in self.technical_indicators not found in DataFrame during cleaning.")

        # Fill any remaining NaNs that might have been introduced by shifts/rolling operations at the beginning
        df.fillna(method='bfill', inplace=True) # Backfill first to handle initial NaNs
        df.fillna(0, inplace=True) # Fill any remaining NaNs with 0 (e.g., if entire column was NaN)


        self.logger.info(f"Data preprocessing complete for {self.symbol}. DataFrame shape: {df.shape}")
        return df

    def _get_observation(self):
        """
        Generate a comprehensive state representation for the RL agent
        
        Returns:
            np.ndarray: Normalized state vector with multi-perspective features
        """
        if self.current_step < self.lookback_window -1: # Ensure enough data for lookback
            # If not enough data, return a zero vector matching observation space shape
            self.logger.warning(f"Not enough data for lookback at step {self.current_step}. Returning zero observation.")
            return np.zeros(self.observation_space.shape)

        current_data_full_lookback = self.df_processed.iloc[self.current_step - self.lookback_window + 1 : self.current_step + 1]
        if current_data_full_lookback.empty or len(current_data_full_lookback) < self.lookback_window:
            self.logger.error(f"RESET: current_data_full_lookback is empty or too short at step {self.current_step} for lookback {self.lookback_window}")
            return np.zeros(self.observation_space.shape)
            
        current_data = self.df_processed.iloc[self.current_step]

        # Check for all required columns from self.technical_indicators
        missing_indicators = [col for col in self.technical_indicators if col not in current_data]
        if missing_indicators:
            for col in missing_indicators:
                 self.logger.error(f"RESET: Missing technical indicator column in df_processed: {col}. Available: {self.df_processed.columns.tolist()}")
            # Potentially return zero observation or raise error
            # This indicates an issue in _preprocess_data or technical_indicators list
            return np.zeros(self.observation_space.shape) 

        # Technical indicators from the preprocessed data
        # These should align with self.technical_indicators and the 31-feature space memory
        indicator_features = []
        # Example: if self.technical_indicators = ['sma_20', 'rsi', 'macd', ...]
        for indicator_name in self.technical_indicators:
            indicator_features.append(current_data[indicator_name])

        # Portfolio features
        portfolio_features = [
            self.portfolio_balance / self.initial_balance,  # Normalized balance
            self.crypto_held / (self.initial_balance / current_data['close'] if current_data['close'] > 0 else 1), # Normalized holdings
            self.portfolio_value / self.initial_balance, # Normalized total portfolio value
            (current_data['close'] * self.crypto_held) / self.initial_balance if self.initial_balance > 0 else 0 # Current value of crypto holdings, normalized
        ]

        # Combine all features
        # The combined list should have 31 features as per MEMORY[9a356701-c8fa-4463-b534-ce392b0caf43]
        # Ensure `self.technical_indicators` contains 27 items.
        observation = np.array(indicator_features + portfolio_features, dtype=np.float32)

        if len(observation) != self.observation_space.shape[0]:
            self.logger.error(
                f"Observation length ({len(observation)}) does not match observation space ({self.observation_space.shape[0]}). "
                f"Indicator features: {len(indicator_features)}, Portfolio features: {len(portfolio_features)}. "
                f"Expected 27 indicator features."
            )
            # Fallback to zeros to prevent crashing, but this is an error condition.
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        return observation

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state vector using min-max scaling
        
        Args:
            state_vector (list): Raw state features
        
        Returns:
            list: Normalized state features
        """
        # Clip extreme values to prevent numerical instability
        normalized_state = []
        for value in state:
            # Handle categorical/binary features differently
            if value in [0, 1]:
                normalized_state.append(float(value))
            else:
                # Min-max normalization with clipping
                try:
                    normalized_value = np.clip(
                        (value - self._state_min_max.get(len(normalized_state), -1e10)) / 
                        (self._state_min_max.get(len(normalized_state), 1e10) - self._state_min_max.get(len(normalized_state), -1e10)),
                        -1, 1
                    )
                    normalized_state.append(normalized_value)
                except Exception:
                    # Fallback normalization if specific min-max not available
                    normalized_state.append(np.clip(value / 100000, -1, 1))
        
        return normalized_state
    
    def _get_current_price(self):
        return self.df_processed['close'].iloc[self.current_step]

    def _update_portfolio(self, action: int, current_price: float) -> None:
        """
        Update the portfolio based on the action taken.
        
        Args:
            action (int): Action taken (0: Hold, 1: Buy, 2: Sell)
            current_price (float): Current price of the asset
        """
        # Ensure current_price is valid
        current_price = max(1e-10, current_price)
        
        # Compute transaction amount (25% of current balance)
        transaction_amount = self.portfolio_balance * 0.25
        
        # Compute transaction fees
        transaction_fee = transaction_amount * self.transaction_fee_percent
        
        if action == 1:  # Buy
            # Calculate amount of crypto that can be bought after fees
            crypto_to_buy = (transaction_amount - transaction_fee) / current_price
            
            # Update portfolio
            self.portfolio_balance -= (transaction_amount)
            self.crypto_held += crypto_to_buy
        
        elif action == 2:  # Sell
            # Sell all held crypto
            if self.crypto_held > 0:
                sell_amount = self.crypto_held * current_price
                sell_proceeds = sell_amount - (sell_amount * self.transaction_fee_percent)
                
                # Update portfolio
                self.portfolio_balance += sell_proceeds
                self.crypto_held = 0
        
        # Calculate total portfolio value
        self.portfolio_value = self.portfolio_balance + (self.crypto_held * current_price)
        
        # Update portfolio history
        self._portfolio_history = {
            'balance': [self.portfolio_balance],
            'crypto_held': [self.crypto_held],
            'total_value': [self.portfolio_value],
            'returns': [0.0]
        }
        
        # Calculate returns
        if len(self._portfolio_history['total_value']) > 1:
            last_value = self._portfolio_history['total_value'][-2]
            current_return = (self.portfolio_value - last_value) / last_value
            self._portfolio_history['returns'].append(current_return)
        
        # Clip portfolio history to prevent excessive memory usage
        max_history_length = 1000
        for key in self._portfolio_history:
            if len(self._portfolio_history[key]) > max_history_length:
                self._portfolio_history[key] = self._portfolio_history[key][-max_history_length:]

    def _calculate_exponential_reward(self, current_portfolio_value, previous_portfolio_value):
        """
        Advanced exponential reward calculation with multi-factor analysis
        
        Reward Components:
        1. Exponential Return Scaling
        2. Stability Bonus
        3. Drawdown Penalty
        4. Trading Efficiency Bonus
        
        Args:
            current_portfolio_value (float): Current total portfolio value
            previous_portfolio_value (float): Previous total portfolio value
        
        Returns:
            float: Calculated exponential reward
        """
        # Prevent division by zero and extreme values
        previous_portfolio_value = max(previous_portfolio_value, 1e-10)
        
        # 1. Exponential Return Component
        # Compute percentage return with logarithmic transformation
        returns = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
        log_return = np.log(1 + returns)
        
        # Quadratic scaling for positive returns, cubic penalty for negative
        if returns >= 0:
            return_reward = np.sign(log_return) * (log_return ** 2)
        else:
            return_reward = -np.abs(returns) ** 3
        
        # 2. Stability Bonus
        # Compute coefficient of variation for portfolio history
        portfolio_history = self.portfolio_history[-30:]  # Last 30 steps
        if len(portfolio_history) > 1:
            portfolio_std = np.std(portfolio_history)
            portfolio_mean = np.mean(portfolio_history)
            
            # Lower coefficient of variation means more stable performance
            stability_factor = 1 / (1 + (portfolio_std / portfolio_mean))
        else:
            stability_factor = 1.0
        
        # 3. Drawdown Penalty
        # Track maximum portfolio value and compute exponential drawdown penalty
        max_portfolio_value = max(self.portfolio_history + [current_portfolio_value])
        drawdown = (max_portfolio_value - current_portfolio_value) / max_portfolio_value
        drawdown_penalty = -np.exp(drawdown * 5)  # Exponential penalty for significant drawdowns
        
        # 4. Trading Efficiency Bonus
        # Penalize excessive trading
        trade_frequency_penalty = -0.1 * len(self.trade_history)
        
        # Optional: Sentiment Integration
        sentiment_multiplier = 1.0
        if self.use_sentiment:
            try:
                # Fetch sentiment for the current trading symbol
                sentiment_data = self.sentiment_analyzer.aggregate_sentiment(self.symbol.split('-')[0])
                
                # Sentiment reward scaling
                sentiment_score = sentiment_data.get('sentiment_score', 0)
                
                # Modify base reward based on sentiment
                # Positive sentiment amplifies reward, negative sentiment dampens reward
                sentiment_multiplier = 1 + (sentiment_score * 0.5)  # +/- 50% reward modification
            
            except Exception as e:
                # Fallback if sentiment analysis fails
                self.logger.warning(f"Sentiment analysis failed: {e}")
        
        # Combine reward components
        total_reward = (
            return_reward +  # Primary exponential return
            (stability_factor * 0.5) +  # Stability bonus
            drawdown_penalty +  # Drawdown penalty
            trade_frequency_penalty  # Trading efficiency penalty
        ) * sentiment_multiplier
        
        # Clip reward to prevent extreme values
        return float(np.clip(total_reward, -10, 10))

    def _calculate_reward(self, current_portfolio_value: float) -> float:
        """
        Calculate reward for the current step.
        
        Args:
            current_portfolio_value (float): Current portfolio value
        
        Returns:
            float: Reward value
        """
        # If this is the first step, return 0
        if not self.portfolio_history:
            return 0.0
        
        # Use exponential reward calculation
        previous_portfolio_value = self.portfolio_history[-1]
        
        return self._calculate_exponential_reward(
            current_portfolio_value, 
            previous_portfolio_value
        )

    def _get_portfolio_state(self):
        """
        Enhanced portfolio state representation
        
        Returns:
            np.ndarray: Detailed portfolio state
        """
        # Basic portfolio metrics
        portfolio_returns = np.array(self.portfolio_history)
        
        portfolio_metrics = [
            self.portfolio_value,  # Current portfolio value
            len(self.portfolio_history),  # Number of steps
            np.mean(portfolio_returns) if len(portfolio_returns) > 0 else 0,  # Mean return
            np.std(portfolio_returns) if len(portfolio_returns) > 0 else 0,  # Return volatility
            np.min(portfolio_returns) if len(portfolio_returns) > 0 else 0,  # Minimum return
            np.max(portfolio_returns) if len(portfolio_returns) > 0 else 0,  # Maximum return
        ]
        
        return np.array(portfolio_metrics, dtype=np.float32)

    def _get_market_state(self):
        return {
            'current_price': self._get_current_price(),
            'previous_price': self.df_processed['close'].iloc[self.current_step - 1]
        }

    def _detect_arbitrage(self):
        """Scan order books for cross-exchange opportunities"""
        try:
            # Get prices from 3 exchanges simultaneously
            rh_price = self.data_provider.get_price('robinhood', self.symbol)
            cb_price = self.data_provider.get_price('coinbase', self.symbol)
            bn_price = self.data_provider.get_price('binance', self.symbol)
            
            # Calculate spreads
            spreads = {
                'rh_cb': (rh_price - cb_price) / cb_price,
                'rh_bn': (rh_price - bn_price) / bn_price,
                'cb_bn': (cb_price - bn_price) / bn_price
            }
            
            # Return max arbitrage opportunity
            return max(spreads.values(), key=abs)
            
        except Exception as e:
            logging.warning(f"Arbitrage scan failed: {e}")
            return 0.0

    def _detect_flash_crash(self):
        """Identify anomalous price movements"""
        prices = self.df_processed['close'].iloc[
            max(0, self.current_step-10):self.current_step+1
        ].values
        
        # Calculate instantaneous velocity
        changes = np.diff(prices) / prices[:-1]
        if len(changes) == 0:
            return False
            
        # Trigger if >3σ move occurs
        std = np.std(changes)
        last_change = changes[-1]
        return abs(last_change) > 3 * std and abs(last_change) > 0.05

    def _chaos_position_size(self, action):
        """Dynamic sizing based on market stability"""
        fd = self._calculate_fractal_dimension(
            self.df_processed['close'].iloc[self.current_step-50:self.current_step].values
        )
        
        # Scale position based on market randomness (1.0=Brownian, 1.5=Chaotic)
        chaos_factor = max(0.1, 2 - fd)  # More aggressive in trending markets
        return action * chaos_factor

    def _map_continuous_action(self, action):
        """
        Advanced action mapping with non-linear transformation
        
        Args:
            action (float): Raw action from policy (-1 to 1)
        
        Returns:
            float: Transformed trading action
        """
        # Sigmoid-like transformation for smoother action scaling
        scaled_action = np.tanh(action * 2)  # Compress to [-1, 1]
        
        # Add chaos-inspired dynamic sizing
        market_volatility = self._calculate_market_volatility()
        chaos_factor = 1 / (1 + np.exp(-market_volatility))
        
        return scaled_action * chaos_factor
    
    def _calculate_market_volatility(self):
        """
        Calculate market volatility for dynamic position sizing
        
        Returns:
            float: Market volatility metric
        """
        recent_prices = self.df_processed['close'].iloc[
            max(0, self.current_step-30):self.current_step+1
        ].values
        volatility = np.std(recent_prices) / np.mean(recent_prices)
        return volatility

    def step(self, action):
        """
        Execute a trading action and return the next observation, reward, done flag, and additional info.
        
        Args:
            action (int): Trading action (0: Hold, 1: Buy, 2: Sell)
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Increment step counter
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.df_processed) - 1
        truncated = False
        
        # Get current row data
        current_row = self.df_processed.iloc[self.current_step]
        
        # Update current price
        self.current_price = current_row['close']
        
        # Trading logic
        trade_cost = 0.001  # 0.1% transaction fee
        
        if action == 1:  # Buy
            # Buy as much as possible with current balance
            max_crypto_to_buy = (self.portfolio_balance * (1 - trade_cost)) / self.current_price
            if max_crypto_to_buy > 0:
                self.crypto_held += max_crypto_to_buy
                self.portfolio_balance -= max_crypto_to_buy * self.current_price * (1 + trade_cost)
        
        elif action == 2:  # Sell
            # Sell all held crypto
            if self.crypto_held > 0:
                sell_value = self.crypto_held * self.current_price * (1 - trade_cost)
                self.portfolio_balance += sell_value
                self.crypto_held = 0
        
        # Calculate portfolio value
        self.portfolio_value = (
            self.portfolio_balance + 
            (self.crypto_held * self.current_price)
        )
        
        # Update portfolio history
        self.portfolio_history.append(self.portfolio_value)
        
        # Calculate reward using exponential reward mechanism
        reward = self._calculate_exponential_reward(
            self.portfolio_value, 
            self.previous_portfolio_value
        )
        
        # Update previous portfolio value
        self.previous_portfolio_value = self.portfolio_value
        
        # Prepare observation vector
        # Technical indicators are already in self.df_processed at current_row
        # Portfolio features need to be appended

        # Find actual column names for technical indicators, case-insensitive
        available_columns_in_df = {col.lower(): col for col in self.df_processed.columns}
        matched_technical_columns = []
        for tc_col_name in self.technical_indicators: # Use the list from __init__
            lower_tc_col_name = tc_col_name.lower()
            if lower_tc_col_name in available_columns_in_df:
                matched_technical_columns.append(available_columns_in_df[lower_tc_col_name])
            else:
                # This should ideally not happen if _preprocess_data ran correctly and all columns were generated
                self.logger.error(f"Missing technical indicator column in df_processed: {tc_col_name}. Available: {list(available_columns_in_df.keys())}")
                # Fallback: add a zero or NaN, or raise error. For now, let's log and it will likely result in NaN/0 from _extract_scalar
                matched_technical_columns.append(tc_col_name) # Add the name, _extract_scalar might handle missing gracefully or error
    
        observation_feature_count = len(self.technical_indicators) + self.portfolio_feature_count
        observation = np.zeros(observation_feature_count, dtype=np.float32)
        
        # Extract scalar values, handling nested or complex data
        def _extract_scalar(value):
            """Extract a scalar value from potentially nested data"""
            if isinstance(value, (list, np.ndarray)):
                return float(value[0]) if len(value) > 0 else 0.0
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0
        
        # Fill with technical indicator data
        for i, col_name in enumerate(matched_technical_columns):
            if col_name in current_row:
                observation[i] = _extract_scalar(current_row[col_name])
            else:
                observation[i] = 0.0 # Fallback for missing columns after logging
    
        # Add portfolio metrics (last 4 features)
        portfolio_offset = len(self.technical_indicators)
        observation[portfolio_offset + 0] = self.portfolio_balance
        observation[portfolio_offset + 1] = self.portfolio_value
        observation[portfolio_offset + 2] = self.crypto_held
        observation[portfolio_offset + 3] = self.current_price # Or a normalized version if needed
    
        # Ensure observation matches observation_space (np.clip will use the space defined in __init__)
        observation = np.clip(
            observation, 
            self.observation_space.low, 
            self.observation_space.high
        )
        
        # Prepare info dictionary
        info = {
            'current_price': self.current_price,
            'portfolio_value': self.portfolio_value,
            'crypto_held': self.crypto_held
        }
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        """
        Reset the trading environment to its initial state
        
        Args:
            seed (int, optional): Random seed for reproducibility
        
        Returns:
            Tuple[np.ndarray, dict]: Initial observation and info dictionary
        """
        super().reset(seed=seed)
        
        # Reset core trading variables
        self.current_step = 0
        self.portfolio_balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.crypto_held = 0.0
        
        # Reset trading state
        self.current_price = self.df_processed['close'].iloc[0]
        self.previous_portfolio_value = self.initial_balance
        
        # Reset portfolio tracking
        self.portfolio_history = [self.initial_balance]
        self.trade_history = []
        
        # Reset reward tracking
        self.cumulative_reward = 0.0
        self.max_portfolio_value = self.initial_balance
        
        # Prepare initial observation
        # Technical indicators are in self.df_processed at the first row (self.current_step is 0)
        # Portfolio features are based on initial state
        current_row_for_reset = self.df_processed.iloc[self.current_step] # self.current_step is 0

        # Find actual column names for technical indicators, case-insensitive
        available_columns_in_df_reset = {col.lower(): col for col in self.df_processed.columns}
        matched_technical_columns_reset = []
        for tc_col_name_reset in self.technical_indicators: # Use the list from __init__
            lower_tc_col_name_reset = tc_col_name_reset.lower()
            if lower_tc_col_name_reset in available_columns_in_df_reset:
                matched_technical_columns_reset.append(available_columns_in_df_reset[lower_tc_col_name_reset])
            else:
                self.logger.error(f"RESET: Missing technical indicator column in df_processed: {tc_col_name_reset}. Available: {list(available_columns_in_df_reset.keys())}")
                matched_technical_columns_reset.append(tc_col_name_reset)

        observation_feature_count_reset = len(self.technical_indicators) + self.portfolio_feature_count
        initial_observation = np.zeros(observation_feature_count_reset, dtype=np.float32)
        
        # Extract scalar values, handling nested or complex data
        def _extract_scalar(value):
            """Extract a scalar value from potentially nested data"""
            if isinstance(value, (list, np.ndarray)):
                return float(value[0]) if len(value) > 0 else 0.0
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0
        
        # Fill with technical indicator data from the first row
        for i, col_name_reset in enumerate(matched_technical_columns_reset):
            if col_name_reset in current_row_for_reset:
                initial_observation[i] = _extract_scalar(current_row_for_reset[col_name_reset])
            else:
                initial_observation[i] = 0.0 # Fallback
    
        # Add initial portfolio metrics (last 4 features)
        portfolio_offset_reset = len(self.technical_indicators)
        initial_observation[portfolio_offset_reset + 0] = self.portfolio_balance # initial_balance
        initial_observation[portfolio_offset_reset + 1] = self.portfolio_value   # initial_balance
        initial_observation[portfolio_offset_reset + 2] = self.crypto_held      # 0.0
        initial_observation[portfolio_offset_reset + 3] = self.current_price    # price at step 0
    
        # Ensure observation matches observation_space (np.clip will use the space defined in __init__)
        initial_observation = np.clip(
            initial_observation, 
            self.observation_space.low, 
            self.observation_space.high
        )
        
        return initial_observation, {}

    def render(self, mode='human'):
        """
        Render the environment state.
        
        Args:
            mode (str): Rendering mode
        """
        if mode == 'human':
            current_price = self.df_processed['close'].iloc[self.current_step]
            print(f"--- Step: {self.current_step} ---")
            print(f"  Price: {current_price:.4f}")
            print(f"  Balance: {self.portfolio_value:.2f}")
            print(f"  Crypto Held: {self.crypto_held:.6f}")
            print(f"  Net Worth: {self.portfolio_value:.2f}")
    
    def close(self):
        """
        Close the environment and clean up resources.
        """
        pass

    def __str__(self):
        return f"CryptoTradingEnv({self.symbol})"

    def __repr__(self):
        return f"CryptoTradingEnv({self.symbol})"

# --- Example Usage (Optional: for testing the environment directly) ---
if __name__ == '__main__':
    # Load some sample data (replace with your actual data loading)
    try:
        # Example using yfinance
        import yfinance as yf
        ticker = "BTC-USD"
        data = yf.download(ticker, start="2022-01-01", end="2023-01-01", interval="1d")
        # Ensure column names are lower case initially if yf returns capitalized
        data.columns = [col.lower() for col in data.columns]
        logger.info(f"Loaded {ticker} data: {data.shape[0]} rows")
        logger.info(data.head())

    except ImportError:
        logger.warning("yfinance not installed. Creating dummy data.")
        # Create dummy data if yfinance is not available
        dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
        data = pd.DataFrame({
            'open': np.random.rand(500) * 100 + 40000,
            'high': np.random.rand(500) * 100 + 40100,
            'low': np.random.rand(500) * 100 + 39900,
            'close': np.random.rand(500) * 100 + 40000,
            'volume': np.random.rand(500) * 1000 + 10000
        }, index=dates)

    if data.empty:
        logger.error("Failed to load data. Exiting.")
    else:
        # Ensure 'Date' is the index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
            elif 'Date' in data.columns: # Check for capitalized 'Date' too
                 data['Date'] = pd.to_datetime(data['Date'])
                 data.set_index('Date', inplace=True)
            else:
                logger.error("Cannot determine date column to set as index.")
                exit()

        # Instantiate the environment
        # TTM features will be calculated inside __init__ -> _preprocess_data
        logger.info("Instantiating CryptoTradingEnv for testing...")
        try:
            env = CryptoTradingEnv(
                                   market_data=data,
                                   initial_balance=10000,
                                   transaction_fee_percent=0.001,
                                   data_period='6mo',
                                   data_interval='1h',
                                   use_sentiment=True,
                                   lookback_window=30
                                   )
            logger.info("Environment instantiated successfully.")
            logger.info(f"Feature columns used by env: ")
            logger.info(f"Processed DataFrame columns: {env.df_processed.columns.tolist()}")


            # Test reset
            logger.info("Testing env.reset()...")
            obs, info = env.reset()
            logger.info(f"Initial Observation Shape: {obs.shape}")
            logger.info(f"Initial Info: {info}")
            logger.info(f"Observation space contains observation: {env.observation_space.contains(obs)}")


            # Test step loop with random actions
            terminated = False
            truncated = False
            total_reward = 0
            steps = 0
            # Calculate max steps correctly based on the length of the *processed* data
            # max_steps = len(env.processed_df) - env.lookback_window - 1
            max_steps = len(env.df_processed) - env.lookback_window # Start step is lookback_window - 1
            logger.info(f"Running test loop for max {max_steps} steps...")

            while not terminated and not truncated and steps < max_steps:
                action = env.action_space.sample() # Random action
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                # env.render() # Uncomment to render each step
                steps += 1
                if steps % 100 == 0: # Log progress every 100 steps
                    logger.info(f"Step {steps}/{max_steps}, Net Worth: {info.get('net_worth', 'N/A'):.2f}")

            logger.info(f"Test loop finished after {steps} steps.")
            logger.info(f"Final Info: {info}")
            logger.info(f"Total Reward: {total_reward}")

            env.close()
            logger.info("Environment closed.")

        except Exception as e:
             logger.error(f"Error during environment testing: {e}", exc_info=True)

             import traceback
             traceback.print_exc()

    print(f"\nFinished test loop after {steps} steps.")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final Net Worth: {env.portfolio_value:.2f}")
    env.close()
