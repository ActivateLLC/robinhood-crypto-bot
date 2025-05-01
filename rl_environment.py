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
        
        # Observation space: technical indicators, portfolio state
        observation_columns = [
            'MA_50', 'MA_200', 'RSI', 'MACD', 'Signal_Line', 
            'Volatility', 'log_returns'
        ]
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(observation_columns) + 2,),  # +2 for balance and portfolio value
            dtype=np.float32
        )
        
        # Reset environment to initial state
        self.reset()

    def _validate_data(self, df):
        """
        Validate and preprocess market data for trading environment
        
        Args:
            df (pd.DataFrame): Input market data
        
        Returns:
            pd.DataFrame: Validated and processed market data
        """
        # Required columns with flexible validation
        required_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'MA_50', 'MA_200', 'RSI', 'MACD', 'Signal_Line'
        ]
        
        # Optional columns
        optional_columns = [
            'Dividends', 'Stock Splits', 'Volatility', 'log_returns'
        ]
        
        # Add missing required columns with default values
        for col in required_columns:
            if col not in df.columns:
                if col.startswith('MA_'):
                    # Moving averages
                    window = int(col.split('_')[1])
                    df[col] = df['Close'].rolling(window=window).mean()
                elif col == 'RSI':
                    # RSI calculation
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df[col] = 100.0 - (100.0 / (1.0 + rs))
                elif col == 'MACD':
                    # MACD calculation
                    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                    df[col] = exp1 - exp2
                elif col == 'Signal_Line':
                    # Signal line calculation
                    df[col] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Add optional columns if missing
        for col in optional_columns:
            if col not in df.columns:
                if col == 'Dividends':
                    df[col] = 0
                elif col == 'Stock Splits':
                    df[col] = 0
                elif col == 'Volatility':
                    df[col] = df['Close'].pct_change().rolling(window=30).std()
                elif col == 'log_returns':
                    df[col] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        # Verify columns after processing
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Could not generate required columns: {missing_columns}")
        
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
            
            return df
        except Exception as e:
            # Log and re-raise with context
            self.logger.error(f"Data loading error for {symbol}: {e}")
            raise

    def _preprocess_data(self, df_raw):
        """
        Advanced preprocessing of raw price data with comprehensive technical indicators
        
        Args:
            df_raw (pd.DataFrame): Raw price data from data provider
        
        Returns:
            pd.DataFrame: Processed dataframe with extensive technical indicators
        """
        # Create a copy to avoid modifying original data
        df = df_raw.copy()
        
        # Standardize column names to uppercase for case-insensitive matching
        df.columns = [col.upper() for col in df.columns]
        
        # Mapping of expected columns
        column_mapping = {
            'OPEN': 'Open',
            'HIGH': 'High', 
            'LOW': 'Low', 
            'CLOSE': 'Close', 
            'VOLUME': 'Volume'
        }
        
        # Rename columns if they exist in a different case
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Check for required columns, create if missing
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                # If column is missing, try to derive from existing columns
                if col == 'Open' and 'MA10' in df.columns:
                    df[col] = df['MA10']  # Use moving average as fallback
                elif col == 'Close' and 'MA30' in df.columns:
                    df[col] = df['MA30']
                else:
                    # Last resort: generate synthetic data
                    df[col] = np.linspace(100, 200, len(df))
        
        # 1. Basic Moving Averages and Exponential Moving Averages
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_30'] = df['Close'].rolling(window=30).mean()
        df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['ema_30'] = df['Close'].ewm(span=30, adjust=False).mean()
        df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # 2. Trend Confirmation Indicator
        df['price_above_ema50'] = (df['Close'] > df['ema_50']).astype(int)
        
        # 3. Volatility Indicators
        df['volatility_10'] = df['Close'].rolling(window=10).std()
        df['volatility_30'] = df['Close'].rolling(window=30).std()
        
        # 4. Volume Analysis
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        
        # 5. Heikin-Ashi Candles Calculation
        df['ha_close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        
        # Initialize Heikin-Ashi columns
        df['ha_open'] = df['Open'].copy()
        df['ha_high'] = df['High'].copy()
        df['ha_low'] = df['Low'].copy()
        
        # Recalculate Heikin-Ashi Open, High, Low
        for i in range(1, len(df)):
            prev_ha_open = df.loc[df.index[i-1], 'ha_open']
            prev_ha_close = df.loc[df.index[i-1], 'ha_close']
            
            # Heikin-Ashi Open
            df.loc[df.index[i], 'ha_open'] = (prev_ha_open + prev_ha_close) / 2
            
            # Heikin-Ashi High and Low
            df.loc[df.index[i], 'ha_high'] = max(
                df.loc[df.index[i], 'High'], 
                df.loc[df.index[i], 'ha_open'], 
                df.loc[df.index[i], 'ha_close']
            )
            df.loc[df.index[i], 'ha_low'] = min(
                df.loc[df.index[i], 'Low'], 
                df.loc[df.index[i], 'ha_open'], 
                df.loc[df.index[i], 'ha_close']
            )
        
        # 6. Momentum Indicators
        # RSI Calculation
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        
        relative_strength = avg_gain / avg_loss
        df['rsi'] = 100.0 - (100.0 / (1.0 + relative_strength))
        
        # MACD Calculation
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 7. TTM Scalper-like Indicators
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        df['bb_upper'] = df['bb_middle'] + 2 * df['Close'].rolling(window=20).std()
        df['bb_lower'] = df['bb_middle'] - 2 * df['Close'].rolling(window=20).std()
        
        # Keltner Channels
        atr = df['High'] - df['Low']
        df['kc_middle'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['kc_upper'] = df['kc_middle'] + 1.5 * atr.rolling(window=20).mean()
        df['kc_lower'] = df['kc_middle'] - 1.5 * atr.rolling(window=20).mean()
        
        # Squeeze Detection (Simplified)
        df['ttm_squeeze'] = ((df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])).astype(int)
        
        # Momentum for Squeeze
        df['ttm_momentum'] = df['Close'] - df['Close'].rolling(window=12).mean()
        
        # Categorical Signal Generation
        df['ttm_signal'] = 0  # Default: No Signal
        df.loc[df['ttm_squeeze'] & (df['ttm_momentum'] > 0), 'ttm_signal'] = 1  # Buy Setup
        df.loc[df['ttm_squeeze'] & (df['ttm_momentum'] < 0), 'ttm_signal'] = 2  # Sell Setup
        
        # Squeeze Release Detection
        squeeze_ended = (~df['ttm_squeeze'].astype(bool)) & df['ttm_squeeze'].shift(1).astype(bool)
        df.loc[squeeze_ended & (df['ttm_momentum'].shift(1) > 0), 'ttm_signal'] = 3  # Fired Long
        df.loc[squeeze_ended & (df['ttm_momentum'].shift(1) < 0), 'ttm_signal'] = 4  # Fired Short
        
        # 8. Returns and Log Returns
        df['returns'] = df['Close'].pct_change().fillna(0)
        df['log_returns'] = np.log(1 + df['returns']).fillna(0)
        
        # Final NaN Handling
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)
        
        return df

    def _get_observation(self):
        """
        Generate a comprehensive state representation for the RL agent
        
        Returns:
            np.ndarray: Normalized state vector with multi-perspective features
        """
        # Get current timestep data
        current_data = self.df_processed.iloc[self.current_step]
        
        # 1. Price and Trend Features
        price_features = [
            current_data['Close'],  # Current price
            current_data['sma_10'],  # Short-term moving average
            current_data['sma_30'],  # Medium-term moving average
            current_data['ema_10'],  # Short-term exponential moving average
            current_data['ema_30'],  # Medium-term exponential moving average
            current_data['price_above_ema50']  # Trend confirmation
        ]
        
        # 2. Heikin-Ashi Features
        ha_features = [
            current_data['ha_close'],
            current_data['ha_open'],
            current_data['ha_high'],
            current_data['ha_low']
        ]
        
        # 3. TTM Scalper Features
        ttm_features = [
            current_data['ttm_squeeze'],  # Squeeze status (0 or 1)
            current_data['ttm_momentum'],  # Momentum value
            current_data['ttm_signal']  # Categorical signal (0-4)
        ]
        
        # 4. Momentum Indicators
        momentum_features = [
            current_data['rsi'],  # Relative Strength Index
            current_data['macd'],  # MACD line
            current_data['macd_signal'],  # MACD signal line
            current_data['macd_hist']  # MACD histogram
        ]
        
        # 5. Volatility and Volume Features
        volatility_volume_features = [
            current_data['volatility_10'],  # Short-term volatility
            current_data['volatility_30'],  # Medium-term volatility
            current_data['Volume'],  # Current volume
            current_data['volume_ratio']  # Volume relative to 30-day moving average
        ]
        
        # 6. Bollinger and Keltner Channel Features
        channel_features = [
            current_data['bb_middle'],
            current_data['bb_upper'],
            current_data['bb_lower'],
            current_data['kc_middle'],
            current_data['kc_upper'],
            current_data['kc_lower']
        ]
        
        # 7. Portfolio and Trading Context
        portfolio_features = [
            self.portfolio_balance,  # Available cash
            self.crypto_held,  # Crypto holdings
            self.portfolio_value,  # Total portfolio value
            current_data['Close'] * self.crypto_held  # Current value of crypto holdings
        ]
        
        # Combine all features
        raw_state = (
            price_features + 
            ha_features + 
            ttm_features + 
            momentum_features + 
            volatility_volume_features + 
            channel_features + 
            portfolio_features
        )
        
        # Normalize the state vector
        state_vector = self._normalize_state(raw_state)
        
        return np.array(state_vector, dtype=np.float32)
    
    def _normalize_state(self, state_vector):
        """
        Normalize state vector using min-max scaling
        
        Args:
            state_vector (list): Raw state features
        
        Returns:
            list: Normalized state features
        """
        # Clip extreme values to prevent numerical instability
        normalized_state = []
        for value in state_vector:
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
        return self.df_processed['Close'].iloc[self.current_step]

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
            'previous_price': self.df_processed['Close'].iloc[self.current_step - 1]
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
        prices = self.df_processed['Close'].iloc[
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
            self.df_processed['Close'].iloc[self.current_step-50:self.current_step].values
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
        recent_prices = self.df_processed['Close'].iloc[
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
        self.current_price = current_row['Close']
        
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
        
        # Prepare observation vector with fixed size
        observation_columns = [
            'MA_50', 'MA_200', 'RSI', 'MACD', 'Signal_Line', 
            'Volatility', 'Log_Returns'
        ]
        
        # Find actual column names, case-insensitive
        available_columns = {col.lower(): col for col in self.df_processed.columns}
        matched_columns = []
        
        for col in observation_columns:
            lower_col = col.lower()
            if lower_col in available_columns:
                matched_columns.append(available_columns[lower_col])
            else:
                raise ValueError(f"Could not find column matching: {col}")
        
        # Construct observation vector with fixed size
        observation_size = 9  # 7 technical indicators + 2 portfolio metrics
        observation = np.zeros(observation_size, dtype=np.float32)
        
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
        
        # Fill with available data
        for i, col in enumerate(matched_columns[:7]):
            observation[i] = _extract_scalar(current_row[col])
        
        # Add portfolio metrics
        observation[7] = self.portfolio_balance
        observation[8] = self.portfolio_value
        
        # Ensure observation matches observation_space
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
        self.current_price = self.df_processed['Close'].iloc[0]
        self.previous_portfolio_value = self.initial_balance
        
        # Reset portfolio tracking
        self.portfolio_history = [self.initial_balance]
        self.trade_history = []
        
        # Reset reward tracking
        self.cumulative_reward = 0.0
        self.max_portfolio_value = self.initial_balance
        
        # Prepare initial observation
        observation_columns = [
            'MA_50', 'MA_200', 'RSI', 'MACD', 'Signal_Line', 
            'Volatility', 'Log_Returns'
        ]
        
        # Find actual column names, case-insensitive
        available_columns = {col.lower(): col for col in self.df_processed.columns}
        matched_columns = []
        
        for col in observation_columns:
            lower_col = col.lower()
            if lower_col in available_columns:
                matched_columns.append(available_columns[lower_col])
            else:
                raise ValueError(f"Could not find column matching: {col}")
        
        # Select first row of data
        current_row = self.df_processed.iloc[0]
        
        # Construct observation vector, handling nested or complex data
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
        
        # Construct observation vector with fixed size
        observation_size = 9  # 7 technical indicators + 2 portfolio metrics
        initial_observation = np.zeros(observation_size, dtype=np.float32)
        
        # Fill with available data
        for i, col in enumerate(matched_columns[:7]):
            initial_observation[i] = _extract_scalar(current_row[col])
        
        # Add portfolio metrics
        initial_observation[7] = self.portfolio_balance
        initial_observation[8] = self.portfolio_value
        
        # Ensure observation matches observation_space
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
            current_price = self.df_processed['Close'].iloc[self.current_step]
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
