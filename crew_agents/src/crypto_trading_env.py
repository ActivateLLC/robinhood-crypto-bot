import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
import ccxt
from typing import Dict, Any, Tuple, Optional
import logging
from datetime import datetime, timedelta

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Initialize logger
logging.basicConfig(level=logging.INFO)

class CryptoTradingEnvironment(gym.Env):
    """
    Cryptocurrency Trading Environment for Reinforcement Learning
    Implements Gymnasium Env interface for training trading agents
    """
    def __init__(
        self, 
        symbol: str = 'BTC-USD',
        initial_capital: float = 10000,
        lookback_window: int = 24,  # 24 hours of history
        trading_fee: float = 0.0005,  # Reduced fee for more frequent trading
        df: Optional[pd.DataFrame] = None
    ):
        super().__init__()
        
        # Initialize lookback window
        self.lookback_window = lookback_window
        
        # Action space: 0=Hold, 1=Buy 25%, 2=Buy 50%, 3=Buy 100%, 4=Sell 25%, 5=Sell 50%, 6=Sell 100%
        self.action_space = gym.spaces.Discrete(7)
        
        # Observation space remains the same
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(11,),
            dtype=np.float32
        )
        
        # Trading parameters
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.crypto_holdings = 0
        self.trading_fee = trading_fee
        self.max_drawdown = 0.2  # 20% max drawdown before reset
        
        # Data handling
        if df is not None:
            self.historical_data = df
        else:
            self.historical_data = self._fetch_historical_data()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create console handler and set level to INFO
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Add formatter to ch
        ch.setFormatter(formatter)
        
        # Add ch to logger
        self.logger.addHandler(ch)
        
        # Trading state variables
        self.current_step = 0
        self.portfolio_value = initial_capital
        self.max_portfolio_value = initial_capital
        
        # Ensure initial observation is prepared
        self._initial_observation = self._generate_observation(
            self.historical_data.iloc[:max(1, self.lookback_window)]
        )
    
    def _fetch_historical_data(self) -> pd.DataFrame:
        """
        Advanced historical data fetching with comprehensive fallback

        Returns:
            pd.DataFrame: Validated historical price data
        """
        # Priority-ordered data sources with specific handling
        data_sources = [
            ('yfinance', self._fetch_yfinance_data),
            ('CCXT Exchanges', self._fetch_ccxt_data),
            ('Simulated', self._generate_simulated_data)
        ]

        from crew_agents.src.error_monitoring_agent import AdvancedErrorMonitoringAgent
        error_monitor = AdvancedErrorMonitoringAgent()

        for source_name, source_method in data_sources:
            try:
                data = source_method()
                if not data.empty and len(data) > 0:
                    return data
            except Exception as e:
                error_monitor.log_error(f"Data fetch failed for {source_name}: {str(e)}")
        
        # If all sources fail, generate simulated data
        return self._generate_simulated_data()
    
    def _fetch_yfinance_data(self) -> pd.DataFrame:
        """
        Fetch data using yfinance with specific configuration
        
        Returns:
            pd.DataFrame: Historical price data
        """
        # Calculate date range within the last 730 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        # Fetch hourly data within the last 730 days
        data = yf.download(
            self.symbol, 
            start=start_date, 
            end=end_date, 
            interval='1h', 
            progress=False
        )
        
        # Ensure data is not empty
        if data.empty:
            raise ValueError("Empty dataset from yfinance")
        
        # Rename columns to standard format
        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        
        # Add additional technical indicators
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(1 + data['returns'])
        
        return data
    
    def _fetch_ccxt_data(self) -> pd.DataFrame:
        """
        Fetch data from CCXT exchanges with robust error handling
        
        Returns:
            pd.DataFrame: Historical price data
        """
        exchanges = [
            ccxt.coinbase,   # Primary exchange
            ccxt.gateio,     # Alternative exchange
            ccxt.kraken      # Backup exchange
        ]
        
        for ExchangeClass in exchanges:
            try:
                exchange = ExchangeClass()
                
                # Use 'BTC/USDT' for most exchanges
                ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=2000)
                
                if ohlcv:
                    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                    data.set_index('timestamp', inplace=True)
                    
                    return data
            
            except Exception as exchange_error:
                self.logger.warning(f"Data fetch error from {ExchangeClass.__name__}: {exchange_error}")
        
        raise ValueError("Unable to fetch data from CCXT exchanges")
    
    def _fetch_coingecko_data(self) -> pd.DataFrame:
        """
        Fetch historical data using CoinGecko API
        
        Returns:
            pd.DataFrame: Historical price data
        """
        import requests
        
        # Extract base cryptocurrency from symbol
        base_crypto = self.symbol.split('-')[0]
        
        url = f"https://api.coingecko.com/api/v3/coins/{base_crypto.lower()}/market_chart?vs_currency=usd&days=730"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Extract price data
            prices = pd.DataFrame(data['prices'], columns=['Timestamp', 'Close'])
            prices['Timestamp'] = pd.to_datetime(prices['Timestamp'], unit='ms')
            prices.set_index('Timestamp', inplace=True)
            
            # Add additional columns
            prices['Open'] = prices['Close'].shift(1)
            prices['High'] = prices['Close'].rolling(window=5).max()
            prices['Low'] = prices['Close'].rolling(window=5).min()
            prices['Volume'] = 0  # CoinGecko doesn't provide volume in this endpoint
            
            return prices
        
        except Exception as e:
            self.logger.error(f"CoinGecko data fetch error: {e}")
            raise
    
    def _fetch_alphavantage_data(self) -> pd.DataFrame:
        """
        Fetch historical data using Alpha Vantage API
        
        Returns:
            pd.DataFrame: Historical price data
        """
        import requests
        
        # Retrieve API key from environment
        api_key = os.getenv('ALPHAVANTAGE_API_KEY')
        if not api_key:
            raise ValueError("Alpha Vantage API key not found")
        
        base_crypto = self.symbol.split('-')[0]
        
        url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={base_crypto}&market=USD&apikey={api_key}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Extract time series data
            time_series = data.get('Time Series (Digital Currency Daily)', {})
            
            if not time_series:
                raise ValueError("No time series data available")
            
            prices = []
            for date, values in time_series.items():
                prices.append({
                    'Timestamp': pd.to_datetime(date),
                    'Open': float(values['1a. open (USD)']),
                    'High': float(values['2a. high (USD)']),
                    'Low': float(values['3a. low (USD)']),
                    'Close': float(values['4a. close (USD)']),
                    'Volume': float(values['5. volume'])
                })
            
            prices_df = pd.DataFrame(prices)
            prices_df.set_index('Timestamp', inplace=True)
            prices_df.sort_index(inplace=True)
            
            return prices_df
        
        except Exception as e:
            self.logger.error(f"Alpha Vantage data fetch error: {e}")
            raise
    
    def _fetch_ccxt_data_advanced(self) -> pd.DataFrame:
        """
        Advanced CCXT data fetching with multiple exchange fallback
        
        Returns:
            pd.DataFrame: Historical price data
        """
        exchanges = [
            'coinbase', 'kraken', 
            'kucoin', 'bitfinex', 'gemini'
        ]
        
        for exchange_id in exchanges:
            try:
                exchange = getattr(ccxt, exchange_id)()
                exchange.load_markets()
                
                # Fetch OHLCV data
                ohlcv = exchange.fetch_ohlcv(
                    self.symbol, 
                    timeframe='1h',  # 1-hour candles
                    limit=730  # 2 years of data
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
                df.set_index('Timestamp', inplace=True)
                
                # Validate data
                if self._validate_data_source(df):
                    return df
            
            except Exception as e:
                self.logger.warning(f"CCXT data fetch error from {exchange_id}: {e}")
        
        raise ValueError("Unable to fetch data from any CCXT exchange")
    
    def _validate_data_source(self, data: pd.DataFrame) -> bool:
        """
        Comprehensive validation of historical data
        
        Args:
            data (pd.DataFrame): Input historical price data
        
        Returns:
            bool: Whether data meets minimum requirements
        """
        # Minimum data requirements
        min_data_points = 1000  # Approximately 1 month of hourly data
        min_date_range = 180  # Days
        
        # Check data is not empty
        if data is None or data.empty:
            self.logger.warning("Data source returned empty dataset")
            return False
        
        # Check number of data points
        if len(data) < min_data_points:
            self.logger.warning(f"Insufficient data points: {len(data)} < {min_data_points}")
            return False
        
        # Check date range
        date_range = (data.index.max() - data.index.min()).days
        if date_range < min_date_range:
            self.logger.warning(f"Insufficient date range: {date_range} days < {min_date_range}")
            return False
        
        # Check for NaN values
        nan_percentage = data.isna().mean().mean() * 100
        if nan_percentage > 5:  # More than 5% NaN values
            self.logger.warning(f"High NaN percentage: {nan_percentage:.2f}%")
            return False
        
        return True
    
    def _advanced_data_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced data preprocessing with multiple technical indicators
        
        Args:
            data (pd.DataFrame): Raw historical price data
        
        Returns:
            pd.DataFrame: Preprocessed data with technical indicators
        """
        # Ensure required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Missing required price columns")
        
        # Technical indicators
        def _calculate_rsi(prices, periods=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        def _calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
            exp1 = prices.ewm(span=fast_period, adjust=False).mean()
            exp2 = prices.ewm(span=slow_period, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=signal_period, adjust=False).mean()
            return macd, signal
        
        # Add technical indicators
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(1 + data['returns'])
        data['rsi'] = _calculate_rsi(data['close'])
        data['macd'], data['macd_signal'] = _calculate_macd(data['close'])
        
        # Volatility indicators
        data['volatility'] = data['close'].rolling(window=20).std()
        
        # Normalize volume
        data['normalized_volume'] = (data['volume'] - data['volume'].mean()) / data['volume'].std()
        
        # Drop initial NaN rows from indicator calculations
        data.dropna(inplace=True)
        
        return data
    
    def _generate_simulated_data(self, days=365) -> pd.DataFrame:
        """
        Generate advanced simulated market data with realistic characteristics
        
        Args:
            days (int): Number of days of simulated data
        
        Returns:
            pd.DataFrame: Simulated market data
        """
        import numpy as np
        import pandas as pd
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate timestamps
        timestamps = pd.date_range(end=pd.Timestamp.now(), periods=days*24, freq='H')
        
        # Simulate price movement with more realistic characteristics
        initial_price = 30000  # Starting price
        daily_volatility = 0.03  # 3% daily volatility
        trend_strength = 0.001  # Slight upward trend
        
        # Generate price series with mean-reverting random walk
        prices = [initial_price]
        for _ in range(len(timestamps) - 1):
            # Mean-reverting component
            mean_reversion = (initial_price - prices[-1]) * 0.01
            
            # Random walk with trend and volatility
            random_change = np.random.normal(trend_strength, daily_volatility) * prices[-1]
            
            # Combine components
            price_change = mean_reversion + random_change
            prices.append(prices[-1] + price_change)
        
        # Create DataFrame with realistic price ranges
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + np.random.uniform(0.001, 0.01)) for p in prices],
            'low': [p * (1 - np.random.uniform(0.001, 0.01)) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(mean=np.log(500), sigma=1, size=len(timestamps))
        })
        
        data.set_index('timestamp', inplace=True)
        
        return data
    
    def _calculate_advanced_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced technical indicators for more nuanced trading decisions
        
        Args:
            data (pd.DataFrame): Historical price data
        
        Returns:
            pd.DataFrame: Enhanced dataset with advanced indicators
        """
        # Existing indicators
        data['returns'] = data['Close'].pct_change()
        data['log_returns'] = np.log(1 + data['returns'])
        
        # Moving Averages
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        
        # Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        relative_strength = avg_gain / avg_loss
        data['RSI'] = 100.0 - (100.0 / (1.0 + relative_strength))
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
        data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()
        
        # MACD - Moving Average Convergence Divergence
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Stochastic Oscillator
        low_14 = data['Low'].rolling(window=14).min()
        high_14 = data['High'].rolling(window=14).max()
        data['%K'] = 100 * (data['Close'] - low_14) / (high_14 - low_14)
        data['%D'] = data['%K'].rolling(window=3).mean()
        
        return data.dropna()
    
    def _advanced_action_selection(self, observation: np.ndarray) -> int:
        """
        Advanced action selection using multiple technical indicators
        
        Args:
            observation (np.ndarray): Current market state observation
        
        Returns:
            int: Selected trading action
        """
        # Unpack observation features
        close_price, rsi, macd, signal_line, stoch_k, stoch_d = observation[-6:]
        
        # Complex trading logic
        buy_signals = 0
        sell_signals = 0
        
        # RSI-based signal
        if rsi < 30:  # Oversold condition
            buy_signals += 1
        elif rsi > 70:  # Overbought condition
            sell_signals += 1
        
        # MACD signal
        if macd > signal_line:  # Bullish crossover
            buy_signals += 1
        elif macd < signal_line:  # Bearish crossover
            sell_signals += 1
        
        # Stochastic Oscillator
        if stoch_k < 20 and stoch_d < 20:  # Oversold
            buy_signals += 1
        elif stoch_k > 80 and stoch_d > 80:  # Overbought
            sell_signals += 1
        
        # Decision logic
        if buy_signals > sell_signals:
            return 1  # Buy
        elif sell_signals > buy_signals:
            return 2  # Sell
        else:
            return 0  # Hold
    
    def _generate_observation(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate advanced observation vector
        
        Args:
            data (pd.DataFrame): Historical price data
        
        Returns:
            np.ndarray: Observation vector for RL agent
        """
        # Select most recent data point
        latest_data = data.iloc[-1]
        
        # Convert to dict to handle both Series and scalar values
        latest_dict = latest_data.to_dict() if hasattr(latest_data, 'to_dict') else latest_data
        
        # Handle potential missing columns with safe defaults
        observation = np.array([
            float(latest_dict.get('Close', 0)),
            float(latest_dict.get('returns', 0)),
            float(latest_dict.get('log_returns', 0)),
            float(latest_dict.get('MA_20', latest_dict.get('Close', 0))),
            float(latest_dict.get('MA_50', latest_dict.get('Close', 0))),
            float(latest_dict.get('MA_200', latest_dict.get('Close', 0))),
            float(latest_dict.get('RSI', 50)),
            float(latest_dict.get('MACD', 0)),
            float(latest_dict.get('Signal_Line', 0)),
            float(latest_dict.get('BB_Upper', latest_dict.get('Close', 0))),
            float(latest_dict.get('BB_Lower', latest_dict.get('Close', 0)))
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self, action: int, current_price: float) -> float:
        """
        Refined reward calculation with more balanced approach
        """
        # Current portfolio value
        current_value = float(self.current_capital + (self.crypto_holdings * current_price))
        
        # 1. Base ROI calculation (less aggressive scaling)
        roi = (current_value / self.initial_capital - 1) * 100
        roi_reward = np.tanh(roi / 10)  # Sigmoid-like scaling
        
        # 2. Momentum component (more conservative)
        price_change = current_price / float(self.historical_data.iloc[max(0, self.current_step-1)]['Close']) - 1
        momentum_bonus = np.tanh(price_change * 50)
        
        # 3. Trading action efficiency
        action_efficiency = 0
        if action in [1,2,3] and self.current_capital > 0:  # Buy actions
            action_efficiency = 0.1
        elif action in [4,5,6] and self.crypto_holdings > 0:  # Sell actions
            action_efficiency = 0.1
        
        # 4. Drawdown protection (less punitive)
        max_portfolio_value = float(max(self.max_portfolio_value, self.initial_capital))
        drawdown = 1 - (current_value / max_portfolio_value)
        drawdown_penalty = -np.tanh(drawdown * 5)
        
        # Combine components with careful weighting
        total_reward = (
            roi_reward * 0.6 + 
            momentum_bonus * 0.2 + 
            action_efficiency * 0.1 +
            drawdown_penalty * 0.1
        )
        
        # Logging for debugging
        if self.current_step % 100 == 0:
            self.logger.info(f"Step {self.current_step}: ROI={roi:.2f}, Portfolio=${current_value:.2f}, Reward={total_reward:.4f}")
        
        return total_reward
    
    def _validate_state(self):
        """
        Validate the current trading environment state
        Raises:
            ValueError: If state is invalid
        """
        # Check numeric validity
        if not isinstance(self.current_capital, (int, float)):
            raise ValueError(f"Invalid current_capital: {self.current_capital}")
        
        if not isinstance(self.crypto_holdings, (int, float)):
            raise ValueError(f"Invalid crypto_holdings: {self.crypto_holdings}")
        
        # Prevent negative holdings
        if self.current_capital < 0:
            self.logger.warning(f"Negative capital detected: ${self.current_capital}")
            self.current_capital = 0
        
        if self.crypto_holdings < 0:
            self.logger.warning(f"Negative crypto holdings detected: {self.crypto_holdings}")
            self.crypto_holdings = 0
        
        # Check data integrity
        if self.historical_data is None or len(self.historical_data) == 0:
            raise ValueError("No historical data available")
        
        # Validate current step
        if self.current_step < 0 or self.current_step >= len(self.historical_data):
            raise ValueError(f"Invalid current step: {self.current_step}")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute trading action with comprehensive error handling
        
        Args:
            action (int): Trading action to execute
        
        Returns:
            Tuple of observation, reward, terminated, truncated, and info
        """
        try:
            # Validate action
            if action not in range(7):
                raise ValueError(f"Invalid action: {action}")
            
            # Safety check for current step
            if self.current_step >= len(self.historical_data):
                self.logger.warning(f"Current step {self.current_step} exceeds historical data length")
                return self._initial_observation, 0, True, False, {"error": "step_out_of_bounds"}
            
            # Extensive pre-trade logging
            self.logger.debug(f"Pre-trade State Details:")
            self.logger.debug(f"Current Step: {self.current_step}")
            self.logger.debug(f"Current Capital: ${self.current_capital:.4f}")
            self.logger.debug(f"Crypto Holdings: {self.crypto_holdings:.8f}")
            
            # Fetch current price
            current_price = float(self.historical_data.iloc[self.current_step]['Close'])
            pre_portfolio_value = self.current_capital + (self.crypto_holdings * current_price)
            
            # Sell actions with enhanced diagnostics
            if action in [4,5,6]:
                # Precise handling of crypto holdings
                self.crypto_holdings = max(0, float(self.crypto_holdings))
                
                # Detailed logging for sell action
                self.logger.debug(f"Sell Action Diagnostic:")
                self.logger.debug(f"Current Price: ${current_price:.4f}")
                self.logger.debug(f"Crypto Holdings Before Sell: {self.crypto_holdings:.8f}")
                
                # Extremely precise threshold for sell
                sell_threshold = 1e-10  # Much smaller threshold
                if self.crypto_holdings <= sell_threshold:
                    self.logger.warning(
                        f"Cannot sell: Insufficient crypto holdings. "
                        f"Current holdings: {self.crypto_holdings:.8f}, "
                        f"Threshold: {sell_threshold}"
                    )
                    # Return initial observation to prevent data slicing issues
                    return self._initial_observation, 0, False, False, {"sell_failure_reason": "insufficient_holdings"}
                
                sell_percentage = [0.25, 0.5, 1.0][action-4]
                position_size = sell_percentage * self.crypto_holdings
                
                # Additional safeguards with detailed logging
                if position_size <= sell_threshold or current_price <= 0:
                    self.logger.warning(
                        f"Invalid sell conditions: "
                        f"position_size={position_size:.8f}, "
                        f"price=${current_price:.4f}, "
                        f"total_holdings={self.crypto_holdings:.8f}, "
                        f"sell_percentage={sell_percentage}"
                    )
                    # Return initial observation to prevent data slicing issues
                    return self._initial_observation, 0, False, False, {"sell_failure_reason": "invalid_sell_conditions"}
                
                sell_value = position_size * current_price
                fee = sell_value * self.trading_fee
                
                self.logger.info(f"SELL Action: {['25%', '50%', '100%'][action-4]}")
                self.logger.info(f"Position Size: {position_size:.8f}")
                self.logger.info(f"Sell Value: ${sell_value:.4f}")
                self.logger.info(f"Fee: ${fee:.4f}")
                
                self.current_capital += (sell_value - fee)
                self.crypto_holdings -= position_size
                
                # Post-sell logging
                self.logger.debug(f"Crypto Holdings After Sell: {self.crypto_holdings:.8f}")
            
            # Generate observation and calculate reward
            observation = self._generate_observation(
                self.historical_data.iloc[max(0, self.current_step-self.lookback_window):self.current_step+1]
            )
            reward = self._calculate_reward(action, current_price)
            
            # Increment current step
            self.current_step += 1
            
            # Check for episode termination
            terminated = self.current_step >= len(self.historical_data) - 1
            
            return observation, reward, terminated, False, {}
        
        except Exception as e:
            self.logger.error(f"Critical error in trading step: {e}", exc_info=True)
            # Return initial observation to prevent data slicing issues
            return self._initial_observation, 0, True, False, {"error": str(e)}
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the trading environment
        
        Args:
            seed (int, optional): Random seed for reproducibility
            options (Dict, optional): Additional reset options
        
        Returns:
            Tuple of initial observation and info dictionary
        """
        # Set random seed if provided
        super().reset(seed=seed)
        
        # Reset state variables
        self.current_step = self.lookback_window
        self.portfolio_value = self.initial_capital
        self.max_portfolio_value = self.initial_capital
        
        # Return initial observation and info
        return self._initial_observation, {}
    
    def render(self):
        """
        Render environment state (optional)
        """
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        print(f"Crypto Holdings: {self.crypto_holdings}")
        print(f"Capital: ${self.current_capital:.2f}")

def main():
    """
    Example usage of CryptoTradingEnvironment
    """
    env = CryptoTradingEnvironment(symbol='BTC-USD')
    
    # Run a sample episode
    obs, info = env.reset()
    terminated, truncated = False, False
    
    while not (terminated or truncated):
        action = env._advanced_action_selection(obs)  # Advanced action selection
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break

if __name__ == "__main__":
    main()
