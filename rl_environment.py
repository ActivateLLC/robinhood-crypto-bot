import gymnasium as gym
import numpy as np
import pandas as pd
import logging
import ta

logger = logging.getLogger(__name__)

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn'])
    from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class CryptoTradingEnv(gym.Env):  
    """Cryptocurrency Trading Environment that follows Gymnasium interface"""
    """
    A custom Gymnasium environment for Reinforcement Learning-based cryptocurrency trading.

    This environment uses historical OHLCV data, incorporates technical indicators,
    handles transaction fees, and provides rewards based on profit/loss with
    penalties for holding losers and buying into losing positions.

    Action Space (Continuous, 1 action):
        -1 = 100% short, 0 = cash, +1 = 100% long

    Observation Space (Box):
        Flattened array containing a lookback window of:
        - Normalized OHLCV data
        - Normalized technical indicators (RSI, MACD signal, Bollinger Band width)
        - Normalized cash balance
        - Normalized crypto holdings
    """
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, 
                 symbol='BTC-USD', 
                 initial_balance=10000, 
                 transaction_fee_percent=0.001, 
                 lookback_window=30,
                 dsr_window=14,
                 sharpe_epsilon=1e-10,
                 seed=42):
        """
        Initialize the Cryptocurrency Trading Environment.
        
        Args:
            symbol (str): Trading symbol
            initial_balance (float): Starting portfolio balance
            transaction_fee_percent (float): Transaction fee percentage
            lookback_window (int): Number of historical steps to consider
            dsr_window (int): Window size for Differential Sharpe Ratio calculation
            sharpe_epsilon (float): Small constant to prevent division by zero
            seed (int): Random seed for reproducibility
        """
        # Call parent constructor
        super().__init__()
        
        # Set random seed for reproducibility
        self.seed = seed
        
        # Create a deterministic random number generator
        self.rng = np.random.default_rng(seed)
        
        # Environment parameters
        self.symbol = symbol
        self.initial_portfolio_value = max(1, initial_balance)
        self.portfolio_value = self.initial_portfolio_value
        self.transaction_fee_percent = transaction_fee_percent
        self.lookback_window = max(5, lookback_window)
        self.dsr_window = max(5, dsr_window)
        self.sharpe_epsilon = max(1e-15, sharpe_epsilon)
        
        # Trading state variables
        self.current_step = 0
        self.crypto_held = 0
        
        # Reward caching mechanism
        self._reward_cache = {}
        
        # Data and trading state
        self.df_processed = None
        self.start_step = 0
        self.total_trades = 0
        
        # Logging and tracking
        logger = logging.getLogger(__name__)
        
        # Fetch and prepare data
        self._load_and_prepare_data()
        
        # Define action and observation spaces
        # Observation space: historical features + portfolio features
        # 9 features * lookback_window + 4 portfolio features
        obs_features = 9  # close, open, high, low, volume, MA10, MA30, returns, log_returns
        obs_size = (obs_features * self.lookback_window) + 4
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-10, high=10, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Ensure reproducibility
        self._np_random, _ = gym.utils.seeding.np_random(seed)
        
        # Log initialization
        logger.info(f"Loaded {symbol} data: {len(self.df_processed)} rows")
        logger.info(self.df_processed.head())

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate the input DataFrame for the trading environment.
        
        Args:
            df (pd.DataFrame): Input DataFrame with price and volume data
        
        Raises:
            ValueError: If data does not meet validation criteria
        """
        # Check if DataFrame is empty
        if df is None or df.empty:
            raise ValueError("Input DataFrame cannot be None or empty")
        
        # Ensure all required columns are present
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure all volumes are positive
        if not (df['volume'] > 0).all():
            logger.warning("Some volumes are zero or negative. Applying absolute value.")
            df['volume'] = np.abs(df['volume'])

    def _preprocess_data(self, df: pd.DataFrame):
        """
        Preprocess the input DataFrame, calculating technical indicators.
        
        Args:
            df (pd.DataFrame): Input DataFrame with OHLCV data
        
        Returns:
            pd.DataFrame: Processed DataFrame with technical indicators
        """
        # Flatten multi-level column names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [f'{col[0]}_{col[1]}' for col in df.columns]
        
        # Ensure columns are lowercase and remove any quotes or parentheses
        df.columns = [col.lower().strip("'()") for col in df.columns]
        
        # Specific handling for yfinance column structure
        if len(df.columns) == 12 and len(set(df.columns)) == 6:
            # Columns are in the format ['adj close', 'btc-usd', 'close', 'btc-usd', ...]
            # Reconstruct the DataFrame with correct columns
            new_columns = ['adj close', 'close', 'high', 'low', 'open', 'volume']
            df.columns = new_columns
        
        # Rename columns if necessary
        column_map = {
            'adj close': 'close',
            'adj close_btc-usd': 'close',
            'open_btc-usd': 'open', 
            'high_btc-usd': 'high', 
            'low_btc-usd': 'low', 
            'volume_btc-usd': 'volume'
        }
        df.rename(columns=column_map, inplace=True)
        
        # Select required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Manually select and rename columns
        selected_data = {}
        for req_col in required_cols:
            # Try multiple variations
            possible_cols = [
                req_col,
                req_col.capitalize(),
                req_col.upper(),
                f'{req_col}_btc-usd',
                f'({req_col}, btc-usd)',
                f"'{req_col}, btc-usd'",
                f'{req_col} (btc-usd)',
                f'{req_col} btc-usd'
            ]
            
            for col in possible_cols:
                if col in df.columns:
                    selected_data[req_col] = df[col]
                    break
            else:
                # If no column found, try to extract from the existing columns
                matching_cols = [col for col in df.columns if req_col in col]
                if matching_cols:
                    selected_data[req_col] = df[matching_cols[0]]
                else:
                    # If still no column found, raise an error
                    raise ValueError(f"Could not find column for {req_col}. Available columns: {df.columns}")
        
        # Create a new DataFrame with selected columns
        df = pd.DataFrame(selected_data)
        
        # Convert to numeric, coercing errors to NaN
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values
        df.dropna(subset=required_cols, inplace=True)
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(1 + df['returns'])
        
        # Calculate technical indicators
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA30'] = df['close'].rolling(window=30).mean()
        
        # Drop rows with NaN values after indicator calculation
        df.dropna(inplace=True)
        
        # Select features, keeping original columns
        features = df[['close', 'open', 'high', 'low', 'volume', 
                       'RSI', 'MA10', 'MA30', 'returns', 'log_returns']]
        
        return features

    def _load_and_prepare_data(self):
        # Load some sample data (replace with your actual data loading)
        try:
            # Example using yfinance
            import yfinance as yf
            ticker = "BTC-USD"
            data = yf.download(ticker, start="2022-01-01", end="2023-01-01", interval="1d", auto_adjust=False)
            
            # Ensure column names are lower case initially
            data.columns = [str(col).lower() for col in data.columns]
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
            raise ValueError("No data available for processing")
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
                    raise ValueError("No date column found")

            # Preprocess data
            self.df_processed = self._preprocess_data(data.copy())

    def _compute_order_book_imbalance(self):
        """
        Compute a simple order book imbalance metric.
        This is a placeholder that simulates order book imbalance.
        In a real-world scenario, this would come from an actual order book API.
        
        Returns:
            float: Order book imbalance metric
        """
        # Simulated order book imbalance based on recent price movements
        recent_returns = self.df_processed['returns'].iloc[max(0, self.current_step-10):self.current_step+1]
        
        # Compute buy/sell pressure based on returns
        buy_pressure = np.sum(recent_returns[recent_returns > 0])
        sell_pressure = np.abs(np.sum(recent_returns[recent_returns < 0]))
        
        # Compute imbalance ratio
        imbalance = (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure + 1e-10)
        
        return float(np.clip(imbalance, -1, 1))

    def _compute_volatility_risk_score(self):
        """
        Compute a dynamic volatility risk score.
        
        Returns:
            float: Volatility risk score between -1 and 1
        """
        # Compute rolling volatility
        start_idx = max(0, self.current_step - 30)  # 30-day rolling window
        recent_returns = self.df_processed['returns'].iloc[start_idx:self.current_step+1]
        
        # Compute volatility metrics
        volatility = np.std(recent_returns)
        max_drawdown = np.min(np.cumsum(recent_returns))
        
        # Normalize and combine volatility metrics
        normalized_volatility = np.clip(volatility * 10, -1, 1)  # Scale and clip
        normalized_drawdown = np.clip(max_drawdown, -1, 1)
        
        # Combine metrics with more weight on volatility
        risk_score = 0.7 * normalized_volatility + 0.3 * normalized_drawdown
        
        return float(risk_score)

    def _get_observation(self):
        """
        Generate an enhanced observation vector with advanced features.
        
        Returns:
            np.ndarray: Normalized observation vector
        """
        # Ensure we have enough historical data
        if self.current_step < 0 or self.current_step >= len(self.df_processed):
            raise ValueError(f"Invalid current step: {self.current_step}")
        
        # Compute start and end indices for historical data
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step + 1
        
        # Extract historical price data with fixed precision
        historical_data = self.df_processed.iloc[start_idx:end_idx]
        
        # Compute technical indicators with fixed precision
        # 9 features: close, open, high, low, volume, MA10, MA30, returns, log_returns
        features_to_normalize = [
            'close', 'open', 'high', 'low', 'volume', 
            'MA10', 'MA30', 'returns', 'log_returns'
        ]
        
        # Robust normalization with safety checks
        def robust_normalize(arr, epsilon=1e-10):
            # Handle empty or constant arrays
            if len(arr) == 0 or np.all(arr == arr[0]):
                return np.zeros_like(arr, dtype=np.float32)
            
            # Remove any non-finite values
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Compute mean and standard deviation with safety checks
            mean = float(np.round(np.mean(arr), 6))
            std = float(np.round(max(np.std(arr), epsilon), 6))
            
            # Normalize and clip
            normalized = np.round((arr - mean) / std, 6)
            
            # Clip to prevent extreme values
            return np.clip(normalized, -10, 10)
        
        # Normalize all features
        normalized_features = []
        for feature in features_to_normalize:
            feature_data = historical_data[feature].to_numpy()
            
            # Ensure consistent length
            if len(feature_data) > self.lookback_window:
                feature_data = feature_data[-self.lookback_window:]
            elif len(feature_data) < self.lookback_window:
                feature_data = np.pad(
                    feature_data, 
                    (self.lookback_window - len(feature_data), 0), 
                    mode='constant', 
                    constant_values=0
                )
            
            normalized_features.append(robust_normalize(feature_data))
        
        # Combine normalized features
        feature_vector = np.concatenate(normalized_features)
        
        # Pad or truncate to fixed observation size
        obs_size = self.observation_space.shape[0] - 4  # Subtract 4 for portfolio features
        if len(feature_vector) > obs_size:
            feature_vector = feature_vector[:obs_size]
        elif len(feature_vector) < obs_size:
            feature_vector = np.pad(
                feature_vector, 
                (0, obs_size - len(feature_vector)), 
                mode='constant', 
                constant_values=0
            )
        
        # Add advanced features
        order_book_imbalance = self._compute_order_book_imbalance()
        volatility_risk_score = self._compute_volatility_risk_score()
        
        # Modify portfolio features to include risk metrics
        current_price = float(np.round(max(1e-10, self._get_current_price()), 6))
        portfolio_features = np.array([
            float(np.round(np.clip(self.portfolio_value / self.initial_portfolio_value, -10, 10), 6)),
            float(np.round(np.clip(self.crypto_held * current_price / self.initial_portfolio_value, -10, 10), 6)),
            order_book_imbalance,
            volatility_risk_score
        ])
        
        # Combine features
        full_observation = np.concatenate([feature_vector, portfolio_features])
        
        return full_observation.astype(np.float32)

    def _get_current_price(self):
        return self.df_processed['close'].iloc[self.current_step]

    def _update_entry_price(self, shares_bought, price_paid):
        """ Calculates the new average entry price after a buy. """
        if shares_bought <= 0:
            return # No change if no shares bought

        current_total_value = self.entry_price * (self.crypto_held - shares_bought) # Value before buy
        new_purchase_value = shares_bought * price_paid
        total_shares = self.crypto_held # Holdings already updated in step() before calling this

        if total_shares > 1e-9: # Avoid division by zero
            self.entry_price = (current_total_value + new_purchase_value) / total_shares
        else:
             self.entry_price = 0 # Should not happen if shares_bought > 0

    def _calculate_reward(self, action):
        """
        Enhanced reward calculation with advanced risk management.
        """
        # Ensure action is a numpy array
        if not isinstance(action, np.ndarray):
            action = np.array([action], dtype=np.float32)
        
        # Validate action shape
        if action.shape != (1,):
            action = action.reshape(1)
        
        # Compute current portfolio metrics
        current_price = self._get_current_price()
        prev_portfolio_value = self.portfolio_value
        prev_crypto_held = self.crypto_held
        
        # Compute trading action
        trade_fraction = float(np.round(action[0], 6))
        trade_amount = abs(trade_fraction) * prev_portfolio_value
        trade_crypto_amount = trade_amount / current_price
        
        # Perform trade simulation
        if trade_fraction > 0:  # Buy
            crypto_bought = trade_crypto_amount * (1 - self.transaction_fee_percent)
            self.crypto_held += crypto_bought
            self.portfolio_value -= trade_amount
        elif trade_fraction < 0:  # Sell
            crypto_sold = min(trade_crypto_amount, self.crypto_held)
            sell_amount = crypto_sold * current_price * (1 - self.transaction_fee_percent)
            self.crypto_held -= crypto_sold
            self.portfolio_value += sell_amount
        
        # Compute new portfolio value
        new_portfolio_value = (
            self.portfolio_value + 
            self.crypto_held * current_price
        )
        
        # Compute returns and volatility
        start_idx = max(0, self.current_step - self.dsr_window)
        end_idx = self.current_step + 1
        recent_returns = self.df_processed.iloc[start_idx:end_idx]['returns'].to_numpy()
        
        # Safety checks for calculations
        if len(recent_returns) == 0:
            recent_returns = np.array([0])
        
        # Market volatility component
        market_volatility = self._calculate_market_volatility()
        
        # Compute portfolio metrics
        portfolio_return = float(np.round(
            (new_portfolio_value - prev_portfolio_value) / max(prev_portfolio_value, 1e-6), 
            6
        ))
        mean_return = float(np.round(np.mean(recent_returns), 6))
        std_return = float(np.round(max(np.std(recent_returns), self.sharpe_epsilon), 6))
        
        # Compute Differential Sharpe Ratio with volatility adjustment
        dsr = float(np.round(
            (portfolio_return - mean_return) / (std_return * (1 + market_volatility)), 
            6
        ))
        
        # Advanced risk management penalties
        # Compute volatility risk score
        volatility_risk = self._compute_volatility_risk_score()
        order_book_imbalance = self._compute_order_book_imbalance()
        
        # Dynamic trade penalty based on market conditions
        trade_penalty = 0.001 * abs(action[0])  # Base trade penalty
        trade_penalty *= (1 + abs(order_book_imbalance))  # Increase penalty during market imbalance
        
        # Dynamic risk penalty
        risk_penalty = 0.005 * abs(volatility_risk)
        
        # Holding penalty with dynamic adjustment
        holding_penalty = 0.001 if self.crypto_held > 0 else 0
        holding_penalty *= (1 + abs(volatility_risk))
        
        # Combine rewards and penalties
        reward = float(np.round(
            np.tanh(dsr) - trade_penalty - risk_penalty - holding_penalty, 
            6
        ))
        
        # Log detailed reward components for debugging
        logger.debug(f"Reward Calculation Details:")
        logger.debug(f"Portfolio Return: {portfolio_return}")
        logger.debug(f"Market Volatility: {market_volatility}")
        logger.debug(f"DSR: {dsr}")
        logger.debug(f"Order Book Imbalance: {order_book_imbalance}")
        logger.debug(f"Volatility Risk Score: {volatility_risk}")
        logger.debug(f"Trade Penalty: {trade_penalty}")
        logger.debug(f"Risk Penalty: {risk_penalty}")
        logger.debug(f"Holding Penalty: {holding_penalty}")
        logger.debug(f"Final Reward: {reward}")
        
        return reward

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
        Execute a single step in the environment.
        
        Args:
            action (np.ndarray or float): Trading action

        Returns:
            tuple: observation, reward, terminated, truncated, info
        """
        # Ensure action is a numpy array
        if not isinstance(action, np.ndarray):
            action = np.array([action], dtype=np.float32)
        
        # Validate action shape
        if action.shape != (1,):
            action = action.reshape(1)
        
        # Clip action to valid range
        action = np.clip(action, -1, 1)
        
        # Increment current step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.df_processed) - 1
        truncated = False
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Get next observation
        observation = self._get_observation()
        
        # Prepare deterministic info dictionary
        info = {
            'portfolio_value': float(np.round(self.portfolio_value, 2)),
            'crypto_held': float(np.round(self.crypto_held, 6)),
            'current_price': float(np.round(self._get_current_price(), 2)),
            'total_trades': self.total_trades,
            'episode_step': self.current_step,
            'action': float(np.round(action[0], 6))
        }
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional reset options

        Returns:
            tuple: Initial observation and info dictionary
        """
        # Set random seed if provided
        super().reset(seed=seed)
        
        # Reset environment variables with fixed precision
        self.current_step = 0
        self.portfolio_value = float(np.round(self.initial_portfolio_value, 2))
        self.crypto_held = 0.0
        self.total_trades = 0
        
        # Clear reward cache
        self._reward_cache = {}
        
        # Get initial observation
        observation = self._get_observation()
        
        # Prepare deterministic info dictionary
        info = {
            'portfolio_value': float(np.round(self.portfolio_value, 2)),
            'crypto_held': 0.0,
            'current_price': float(np.round(self._get_current_price(), 2)),
            'total_trades': 0,
            'episode_step': 0
        }
        
        return observation, info
    
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
                                   initial_balance=10000,
                                   lookback_window=30,
                                   transaction_fee_percent=0.001,
                                   dsr_window=14,
                                   sharpe_epsilon=1e-10,
                                   seed=42
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
