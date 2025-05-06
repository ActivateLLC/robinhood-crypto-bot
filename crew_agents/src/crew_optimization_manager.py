import os
import sys
import time
import json
import logging
import traceback
import joblib
from typing import Dict, Any, List, Optional
import datetime # <-- Add import

# --- Early Logging --- #
import logging
initial_logger = logging.getLogger(f"{__name__}_initial")
initial_logger.info("Crew Optimization Manager Script - Top Level Reached.")
# --- End Early Logging --- #

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yfinance as yf
import ccxt
import ta
from sklearn.preprocessing import StandardScaler

import multiprocessing
import subprocess
import gymnasium
from stable_baselines3 import PPO
import torch
import optuna

from alt_crypto_data import AltCryptoDataProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append('/Users/activate/Dev/robinhood-crypto-bot')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/activate/Dev/robinhood-crypto-bot/logs/crew_optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install missing Python dependencies"""
    dependencies = [
        'gymnasium', 'optuna', 'stable-baselines3', 
        'ta-lib', 'torch', 'numpy', 'pandas'
    ]
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
        except Exception as e:
            print(f"Could not install {dep}: {e}")

install_dependencies()

class AgentCommunicationHub:
    """
    Centralized communication hub for coordinating agent interactions
    """
    def __init__(self):
        self.shared_context = {
            'market_insights': {},
            'risk_assessments': {},
            'feature_engineering_results': {},
            'optimization_history': []
        }
        self.communication_log = []
    
    def broadcast_message(self, sender: str, message: Dict[str, Any], recipients: List[str] = None):
        """
        Broadcast a message across agents
        
        Args:
            sender (str): Name of the sending agent
            message (Dict): Message content
            recipients (List[str], optional): Specific agents to receive the message
        """
        timestamp = time.time()
        communication_entry = {
            'timestamp': timestamp,
            'sender': sender,
            'message': message,
            'recipients': recipients or 'ALL'
        }
        self.communication_log.append(communication_entry)
        
        # Update shared context based on message type
        if message.get('type') == 'market_insight':
            self.shared_context['market_insights'][sender] = message
        elif message.get('type') == 'risk_assessment':
            self.shared_context['risk_assessments'][sender] = message
        
        logger.info(f"Agent Communication: {sender} broadcast message: {message}")
    
    def get_shared_context(self, context_type: str = None):
        """
        Retrieve shared context or specific context type
        
        Args:
            context_type (str, optional): Specific context type to retrieve
        
        Returns:
            Shared context or specific context
        """
        if context_type:
            return self.shared_context.get(context_type, {})
        return self.shared_context
    
    def generate_collaborative_insights(self):
        """
        Generate collaborative insights by aggregating agent communications
        
        Returns:
            Dict of synthesized insights
        """
        collaborative_insights = {
            'market_sentiment': self._analyze_market_sentiment(),
            'risk_aggregation': self._aggregate_risk_assessments(),
            'optimization_trends': self._extract_optimization_trends()
        }
        return collaborative_insights
    
    def _analyze_market_sentiment(self):
        """
        Analyze market sentiment from agent communications
        """
        market_insights = self.shared_context.get('market_insights', {})
        sentiment_scores = [
            insight.get('sentiment_score', 0) 
            for insight in market_insights.values()
        ]
        
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        
        if avg_sentiment > 0.5:
            return 'Strongly Bullish'
        elif avg_sentiment > 0:
            return 'Moderately Bullish'
        elif avg_sentiment == 0:
            return 'Neutral'
        elif avg_sentiment > -0.5:
            return 'Moderately Bearish'
        else:
            return 'Strongly Bearish'
    
    def _aggregate_risk_assessments(self):
        """
        Aggregate risk assessments from different agents
        """
        risk_assessments = self.shared_context.get('risk_assessments', {})
        
        aggregated_risks = {
            'volatility': [],
            'max_drawdown': [],
            'correlation_risk': []
        }
        
        for assessment in risk_assessments.values():
            for risk_type, value in assessment.items():
                if risk_type in aggregated_risks:
                    aggregated_risks[risk_type].append(value)
        
        return {
            risk_type: {
                'mean': np.mean(values) if values else 0,
                'max': np.max(values) if values else 0,
                'min': np.min(values) if values else 0
            }
            for risk_type, values in aggregated_risks.items()
        }
    
    def _extract_optimization_trends(self):
        """
        Extract optimization trends from communication history
        """
        optimization_history = self.shared_context.get('optimization_history', [])
        
        trends = {
            'most_successful_strategies': {},
            'emerging_patterns': {}
        }
        
        # Analyze optimization history for trends
        for entry in optimization_history:
            # Placeholder for trend extraction logic
            pass
        
        return trends

class CrewOptimizationAgent:
    """
    Advanced multi-agent optimization system for cryptocurrency trading
    """
    def __init__(self, symbol: str = 'BTC-USD', initial_capital: float = 100000):
        """
        Initialize comprehensive optimization framework
        
        Args:
            symbol (str): Primary trading symbol
            initial_capital (float): Starting investment capital
        """
        self.symbol = symbol
        self.initial_capital = initial_capital
        
        # Initialize specialized agents
        self.agents = {
            'data_analyst': DataAnalystAgent(symbol),
            'feature_engineer': FeatureEngineeringAgent(symbol),
            'hyperparameter_tuner': HyperparameterTuningAgent(symbol, logger=logger), # Pass logger
            'risk_manager': RiskManagementAgent(symbol),
            'model_trainer': ModelTrainingAgent(symbol),
            'portfolio_optimizer': PortfolioOptimizationAgent(symbol)
        }
        
        # Collaborative insights storage
        self.optimization_history = []
        self.global_performance_metrics = {
            'cumulative_roi': [],
            'max_drawdowns': [],
            'sharpe_ratios': []
        }
        
        # Add communication hub
        self.communication_hub = AgentCommunicationHub()
        
        # Modify agent initialization to enable communication
        for name, agent in self.agents.items():
            agent.communication_hub = self.communication_hub
    
    def run_optimization_pipeline(self) -> Dict[str, Any]:
        """
        Advanced collaborative optimization pipeline
        
        Returns:
            Comprehensive optimization results
        """
        try:
            # Start timing the optimization process
            start_time = time.time()
            logger.info(f"Starting Optimization Pipeline for {self.symbol}")
            
            # Instantiate Data Provider early for reuse
            try:
                initial_logger.info("Attempting to import AltCryptoDataProvider...")
                data_provider = AltCryptoDataProvider()
                logger.info("AltCryptoDataProvider initialized.")
            except Exception as dp_error:
                logger.error(f"Failed to initialize AltCryptoDataProvider: {dp_error}", exc_info=True)
                return {'error': 'Data Provider Initialization Failed', 'details': str(dp_error)}

            # 1. Advanced Market Data Analysis
            # Pass the data_provider instance
            market_data = self.agents['data_analyst'].analyze_market_data(data_provider=data_provider)
            
            # --- Check if market_data is valid --- #
            if market_data is None or market_data.empty:
                logger.error(f"Market data analysis failed or returned empty data for {self.symbol}. Skipping further processing.")
                return {
                    'error': 'Market Data Analysis Failed',
                    'symbol': self.symbol,
                    'details': 'analyze_market_data returned None or empty DataFrame.'
                }
            # -------------------------------------- #
            
            # 2. Sophisticated Feature Engineering
            enhanced_features = self.agents['feature_engineer'].engineer_features(market_data)

            # 3. Multi-Objective Hyperparameter Tuning
            # Pass data_provider and features
            best_hyperparams = self.agents['hyperparameter_tuner'].tune_hyperparameters(
                data_provider=data_provider,
                features=enhanced_features,
                objectives=['roi', 'risk', 'stability'] # Example objectives
            )

            # 4. Comprehensive Risk Assessment
            risk_profile = self.agents['risk_manager'].assess_risk(
                hyperparameters=best_hyperparams,
                initial_capital=self.initial_capital
            )
            
            # 5. Advanced Model Training
            # Pass data_provider, features, hyperparameters, and risk_profile
            trained_model = self.agents['model_trainer'].train_model(
                data_provider=data_provider,
                features=enhanced_features,
                hyperparameters=best_hyperparams,
                risk_profile=risk_profile
            )
            
            # 6. Portfolio Optimization
            portfolio_strategy = self.agents['portfolio_optimizer'].optimize_portfolio(
                model=trained_model,
                risk_profile=risk_profile,
                initial_capital=self.initial_capital
            )
            
            # Compile comprehensive results
            optimization_results = {
                'timestamp': time.time(),
                'duration': time.time() - start_time,
                'symbol': self.symbol,
                'initial_capital': self.initial_capital,
                'market_data': market_data,
                'features': enhanced_features,
                'hyperparameters': best_hyperparams,
                'risk_profile': risk_profile,
                'trained_model': trained_model,
                'portfolio_strategy': portfolio_strategy
            }
            
            # Record optimization history
            self.optimization_history.append(optimization_results)
            
            # Update global performance metrics
            self._update_performance_metrics(optimization_results)
            
            # Broadcast market insights
            self.communication_hub.broadcast_message(
                sender='crew_optimization_agent',
                message={
                    'type': 'market_insight',
                    'symbol': self.symbol,
                    'market_data': market_data,
                    'features': enhanced_features
                }
            )
            
            # Broadcast risk profile
            self.communication_hub.broadcast_message(
                sender='crew_optimization_agent',
                message={
                    'type': 'risk_assessment',
                    **risk_profile
                }
            )
            
            # Generate collaborative insights
            collaborative_insights = self.communication_hub.generate_collaborative_insights()
            logger.info(f"Collaborative Insights: {collaborative_insights}")
            
            logger.info(f"Optimization Pipeline Completed Successfully for {self.symbol}")
            logger.info(f"Optimization Duration: {optimization_results['duration']:.2f} seconds")
            
            # Return the final trained model within the results
            optimization_results['trained_model'] = trained_model # Ensure model is in results
            return optimization_results
        
        except Exception as e:
            logger.critical(f"Catastrophic Optimization Pipeline Failure: {e}")
            logger.critical(traceback.format_exc())
            return {'error': 'Catastrophic Optimization Failure', 'details': str(e)}
    
    def _update_performance_metrics(self, results: Dict[str, Any]):
        """
        Update global performance tracking metrics
        
        Args:
            results (Dict): Optimization results
        """
        try:
            # Extract and store key performance indicators
            self.global_performance_metrics['cumulative_roi'].append(
                results.get('portfolio_strategy', {}).get('cumulative_roi', 0)
            )
            self.global_performance_metrics['max_drawdowns'].append(
                results.get('risk_profile', {}).get('max_drawdown', 0)
            )
            self.global_performance_metrics['sharpe_ratios'].append(
                results.get('portfolio_strategy', {}).get('sharpe_ratio', 0)
            )
            
            logger.info("Performance metrics updated successfully")
        
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")

class DataAnalystAgent:
    """Agent responsible for market data analysis"""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.communication_hub = None
    
    def analyze_market_data(self, data_provider: AltCryptoDataProvider) -> Optional[pd.DataFrame]:
        """Analyze market data using yfinance and ta libraries"""
        try:
            logger.info(f"Analyzing market data for {self.symbol}...")
            # Define date range - Last 40 days 
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=40)

            # Fetch historical data
            # Use start/end dates instead of period for hourly data 
            data = data_provider.fetch_historical_data(
                symbol=self.symbol, 
                start=start_date.strftime('%Y-%m-%d'), 
                end=end_date.strftime('%Y-%m-%d'), 
                interval='1h', 
                progress=False
            )

            if data.empty:
                raise ValueError(f"No market data available for {self.symbol}")
            
            # Data cleaning and preprocessing
            data = self._preprocess_market_data(data)
            
            # Feature engineering
            data = self._add_technical_indicators(data)
            
            # Normalize and scale features
            data = self._normalize_features(data)
            
            logger.info(f"Market data analysis completed for {self.symbol}")
            logger.info(f"Data shape: {data.shape}")
            
            return data
        
        except Exception as e:
            logger.error(f"Market data analysis failed: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _fetch_alternative_data(self, days: int, interval: str):
        """
        Fetch data from alternative sources if primary source fails
        
        Args:
            days (int): Number of historical days
            interval (str): Data interval
        
        Returns:
            DataFrame with market data
        """
        try:
            import ccxt
            
            # Use CCXT for cryptocurrency data
            exchange = ccxt.binance()
            ohlcv = exchange.fetch_ohlcv(self.symbol, interval, limit=days*24)
            
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
            
            return data
        
        except Exception as e:
            logger.error(f"Alternative data fetch failed: {e}")
            return pd.DataFrame()
    
    def _preprocess_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced data preprocessing
        
        Args:
            data (DataFrame): Raw market data
        
        Returns:
            Preprocessed DataFrame
        """
        # Remove duplicate indices
        data = data[~data.index.duplicated(keep='first')]
        
        # Handle missing values
        data.dropna(inplace=True)
        
        # Ensure correct data types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove extreme outliers
        for col in numeric_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced technical indicators
        
        Args:
            data (DataFrame): Preprocessed market data
        
        Returns:
            DataFrame with additional technical indicators
        """
        try:
            # Ensure 'Close' is a Series
            close_price = data['Close']
            if not isinstance(close_price, pd.Series):
                close_price = pd.Series(close_price.values.flatten(), index=data.index)

            # Calculate technical indicators using the ensured Series
            data['SMA_50'] = ta.trend.SMAIndicator(close=close_price, window=50).sma_indicator()
            data['SMA_200'] = ta.trend.SMAIndicator(close=close_price, window=200).sma_indicator()
            data['RSI'] = ta.momentum.RSIIndicator(close=close_price, window=14).rsi()
            # MACD
            macd_indicator = ta.trend.MACD(close=close_price)
            data['MACD'] = macd_indicator.macd()
            data['MACD_signal'] = macd_indicator.macd_signal()
            
            # Bollinger Bands
            bollinger_indicator = ta.volatility.BollingerBands(close=close_price)
            data['Bollinger_High'] = bollinger_indicator.bollinger_hband()
            data['Bollinger_Low'] = bollinger_indicator.bollinger_lband()
            # Stochastic Oscillator
            stoch_indicator = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=close_price)
            data['Stoch_Signal'] = stoch_indicator.stoch_signal()

            # Feature Scaling (Example: Standard Scaler)
            from sklearn.preprocessing import StandardScaler
            
            # Select features for normalization
            features_to_normalize = [
                'Open', 'High', 'Low', 'Close', 'Volume', 
                'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_signal',
                'Bollinger_High', 'Bollinger_Low', 'Stoch_Signal'
            ]
            
            # Initialize scaler
            scaler = StandardScaler()
            
            # Normalize selected features
            data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])
            
            return data
        
        except Exception as e:
            logger.error(f"Technical indicator calculation failed: {e}")
            return data

class FeatureEngineeringAgent:
    """Agent responsible for feature engineering"""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.communication_hub = None
    
    def engineer_features(self, market_data):
        """
        Create advanced technical indicators
        
        Args:
            market_data (pd.DataFrame): Raw market data
        
        Returns:
            Enhanced feature DataFrame
        """
        try:
            import pandas as pd
            import numpy as np
            import ta
            
            # Ensure market_data is a DataFrame
            if not isinstance(market_data, pd.DataFrame):
                raise ValueError("Market data must be a pandas DataFrame")
            
            # Compute technical indicators
            market_data['RSI'] = ta.momentum.RSIIndicator(market_data['Close']).rsi()
            market_data['MACD'] = ta.trend.MACD(market_data['Close']).macd()
            market_data['Signal_Line'] = ta.trend.MACD(market_data['Close']).macd_signal()
            
            # Volatility indicators
            market_data['Bollinger_High'] = ta.volatility.BollingerBands(market_data['Close']).bollinger_hband()
            market_data['Bollinger_Low'] = ta.volatility.BollingerBands(market_data['Close']).bollinger_lband()
            
            # Log feature engineering details
            logger.info("Feature Engineering Completed")
            logger.info(f"Added Features: RSI, MACD, Signal Line, Bollinger Bands")
            
            return market_data
        
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            logger.error(traceback.format_exc())
            return market_data

class HyperparameterTuningAgent:
    """Agent responsible for hyperparameter optimization"""
    def __init__(self, symbol: str, logger):
        self.symbol = symbol
        self.search_space = {
            # PPO specific
            'n_steps': optuna.distributions.IntDistribution(128, 2048, step=64),
            'gamma': optuna.distributions.FloatDistribution(0.9, 0.9999, log=True),
            # 'vf_coef': optuna.distributions.UniformDistribution(0.2, 0.8),
        }
        self.logger = logger # Store logger instance
    
    def tune_hyperparameters(self, data_provider: AltCryptoDataProvider, features: pd.DataFrame, objectives: List[str], n_trials: int = 10):
        """
        Use Optuna for hyperparameter tuning

        Args:
            data_provider (AltCryptoDataProvider): Data provider instance
            features (pd.DataFrame): Engineered features (used for environment creation)
            objectives (list): Optimization objectives (currently unused, placeholder)
            n_trials (int): Number of Optuna trials

        Returns:
            dict: Best hyperparameters found
        """
        self.logger.info(f"Starting Hyperparameter Tuning for {self.symbol} ({n_trials} trials)")

        # --- Define Objective Function Nested --- #
        def objective(trial):
            try:
                # 1. Define Hyperparameter Search Space
                hyperparameters = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
                    'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
                    'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                    'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
                    'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
                    'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
                    'n_epochs': trial.suggest_int('n_epochs', 5, 20),
                    # Add other env params if needed, e.g., lookback_window
                    # 'lookback_window': trial.suggest_int('lookback_window', 12, 48)
                }

                # 2. Create Environment Instance for this trial
                # Use captured data_provider and features from the outer scope
                # Ensure features is not empty
                if features is None or features.empty:
                     logger.warning(f"Trial {trial.number}: Features data is empty. Skipping trial.")
                     return -1e9 # Return poor score if no features

                env = CryptoTradingEnvironment(
                    data_provider=data_provider, # Captured from outer scope
                    symbol=self.symbol,
                    initial_capital=10000.0, # Use a standard balance for comparison
                    trading_fee=0.001,      # Use a standard fee for comparison
                    df=features,              # Captured from outer scope
                    # live_trading=False # Default
                    # lookback_window=hyperparameters.get('lookback_window', 24) # If tuning lookback
                )

                # Check if data loading/processing within env failed (redundant if features is checked, but safe)
                if env.historical_data is None or env.historical_data.empty:
                    logger.warning(f"Trial {trial.number}: Environment failed to initialize with data. Skipping trial.")
                    return -1e9

                # Wrap in a VecEnv
                from stable_baselines3.common.env_util import make_vec_env
                vec_env = make_vec_env(lambda: env, n_envs=1)

                # 3. Create and Train PPO Model
                from stable_baselines3 import PPO
                model = PPO(
                    "MlpPolicy",
                    vec_env,
                    verbose=0,
                    device='auto',
                    # Filter only PPO specific hyperparameters
                    **{k: v for k, v in hyperparameters.items() if k in ['learning_rate', 'n_steps', 'batch_size', 'gamma', 'gae_lambda', 'ent_coef', 'clip_range', 'n_epochs']}
                )

                training_timesteps = 10000 # Reduced timesteps for faster tuning
                model.learn(total_timesteps=training_timesteps)

                # 4. Evaluate the Trained Model
                obs, _ = vec_env.reset()
                cumulative_reward = 0
                # Get initial portfolio value from the environment after reset
                final_portfolio_value = env.initial_capital # Start with initial capital
                n_steps_eval = len(env.historical_data) - env.lookback_window # Evaluate over valid steps

                if n_steps_eval <= 0:
                    logger.warning(f"Trial {trial.number}: Not enough data points ({len(env.historical_data)}) for evaluation with lookback {env.lookback_window}. Skipping eval.")
                    final_portfolio_value = env.initial_capital # Return initial capital if no eval possible
                else:
                    for step in range(n_steps_eval):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = vec_env.step(action)
                        done = terminated or truncated
                        cumulative_reward += reward[0]
                        # Get portfolio value from info if available
                        if isinstance(info, list) and len(info) > 0 and 'portfolio_value' in info[0]:
                            final_portfolio_value = info[0]['portfolio_value']

                        if done:
                            # Ensure final value is captured correctly on done
                            if isinstance(info, list) and len(info) > 0 and 'portfolio_value' in info[0]:
                                final_portfolio_value = info[0]['portfolio_value']
                            else: # Fallback to env property if not in info
                                final_portfolio_value = env.portfolio_value
                            break

                    # If loop finished without done, capture final value from env state
                    if not done:
                         final_portfolio_value = env.portfolio_value

                # Cleanup environment
                vec_env.close()

                self.logger.debug(f"Trial {trial.number}: Params={trial.params}, Final Value={final_portfolio_value:.2f}, Cum Reward={cumulative_reward:.2f}")

                # Handle potential NaN or non-positive portfolio values
                if np.isnan(final_portfolio_value) or final_portfolio_value <= 0:
                    self.logger.warning(f"Trial {trial.number} resulted in invalid final portfolio value: {final_portfolio_value}. Returning poor score.")
                    return -1e9 # Return a very poor score

                # Return final portfolio value as the objective to maximize
                return final_portfolio_value

            except Exception as e:
                self.logger.warning(f"Trial {trial.number} failed during execution: {e}", exc_info=True)
                # Consider logging traceback here if needed: self.logger.warning(traceback.format_exc())
                return -1e9 # Return a very poor score on any exception
        # --- End of Nested Objective Function --- #

        try:
            # Create Optuna study
            study = optuna.create_study(direction='maximize')
            # Run optimization using the nested objective function
            study.optimize(objective, n_trials=n_trials, timeout=600) # Added timeout

            self.logger.info("Hyperparameter Tuning Completed")
            self.logger.info(f"Best Trial Value (Final Portfolio): {study.best_value}")
            self.logger.info(f"Best Hyperparameters: {study.best_params}")
            return study.best_params

        except Exception as e:
            self.logger.error(f"Optuna hyperparameter tuning failed: {e}", exc_info=True)
            # Return default or empty dict on failure
            return {}

class RiskManagementAgent:
    """Agent focused on risk analysis and mitigation"""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.communication_hub = None
    
    def assess_risk(self, hyperparameters, initial_capital):
        """
        Assess trading strategy risk
        
        Args:
            hyperparameters (dict): Hyperparameters to evaluate
            initial_capital (float): Starting investment capital
        
        Returns:
            Risk profile dictionary
        """
        try:
            # Risk calculation logic
            risk_profile = {
                'max_drawdown_tolerance': 0.2,  # 20% max drawdown
                'volatility_threshold': 0.3,    # 30% max volatility
                'risk_adjusted_score': None
            }
            
            # Log risk assessment
            logger.info("Risk Management Assessment")
            logger.info(f"Hyperparameters Risk Profile: {risk_profile}")
            
            return risk_profile
        
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {}

class ModelTrainingAgent:
    """Agent responsible for model training"""
    def __init__(self, symbol: str):
        self.symbol = symbol # Keep symbol for context/logging if needed
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{symbol}")

    def train_model(self, data_provider: AltCryptoDataProvider, features: pd.DataFrame, hyperparameters: Dict, risk_profile: Dict):
        """
        Train reinforcement learning model
        
        Args:
            data_provider (AltCryptoDataProvider): Data provider instance
            features (pd.DataFrame): Engineered features
            hyperparameters (dict): Optimized hyperparameters
            risk_profile (dict): Risk assessment profile
        
        Returns:
            Trained model
        """
        try:
            import gymnasium
            from stable_baselines3 import PPO
            import torch
            
            # Create custom gym environment
            from crew_agents.src.crypto_trading_env import CryptoTradingEnvironment
            
            # --- Environment Initialization ---
            # Extract potential environment-related params from hyperparameters IF THEY WERE TUNED
            # Defaults will be used from CryptoTradingEnvironment if not tuned.
            env_kwargs = {}
            if 'initial_capital' in hyperparameters: env_kwargs['initial_capital'] = hyperparameters['initial_capital']
            if 'lookback_window' in hyperparameters: env_kwargs['lookback_window'] = hyperparameters['lookback_window']
            if 'trading_fee' in hyperparameters: env_kwargs['trading_fee'] = hyperparameters['trading_fee']
            # Add other potential env params here if they are added to Optuna search space
            
            self.logger.info(f"Initializing CryptoTradingEnvironment with specific args and kwargs: {env_kwargs}")
            env = CryptoTradingEnvironment(
                df=features,                # Pass historical data
                data_provider=data_provider, # Pass the data provider instance
                symbol=self.symbol,         # Pass the symbol
                # Pass any tuned env parameters, otherwise defaults are used
                **env_kwargs
            )
            self.logger.info("CryptoTradingEnvironment initialized for training.")
            
            # --- PPO Model Initialization ---
            # Filter hyperparameters for only those relevant to PPO
            ppo_hyperparams = {
                k: v for k, v in hyperparameters.items() 
                if k in [
                    'n_steps', 'gamma', 'learning_rate', 'vf_coef', 'max_grad_norm', 
                    'gae_lambda', 'ent_coef', 'clip_range', 'batch_size', 'n_epochs'
                ] # Add other valid PPO params if tuned
            }
            self.logger.info(f"Initializing PPO model with hyperparameters: {ppo_hyperparams}")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1, 
                tensorboard_log=f"{self.model_save_dir}/tensorboard/",
                # Use only the relevant hyperparameters for the PPO model
                **ppo_hyperparams 
            )
            
            # --- Model Training ---
            model.learn(total_timesteps=50000)
            
            # Log training details
            logger.info("Model Training Completed")
            logger.info(f"Training Symbol: {self.symbol}")
            logger.info(f"Hyperparameters Used: {hyperparameters}")
            
            return model
        
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            logger.error(traceback.format_exc())
            return None

class PortfolioOptimizationAgent:
    """
    Advanced portfolio optimization and strategy generation
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.communication_hub = None
    
    def optimize_portfolio(self, model, risk_profile: Dict[str, Any], initial_capital: float) -> Dict[str, Any]:
        """
        Generate sophisticated portfolio optimization strategy
        
        Args:
            model: Trained reinforcement learning model
            risk_profile (Dict): Comprehensive risk assessment
            initial_capital (float): Starting investment amount
        
        Returns:
            Portfolio optimization strategy
        """
        try:
            # Advanced portfolio allocation strategy
            portfolio_strategy = {
                'initial_capital': initial_capital,
                'allocation_strategy': self._determine_allocation(risk_profile),
                'rebalancing_frequency': self._calculate_rebalancing_frequency(risk_profile),
                'risk_adjusted_position_sizing': self._calculate_position_sizing(risk_profile),
                'cumulative_roi': self._simulate_portfolio_performance(model, initial_capital),
                'sharpe_ratio': self._calculate_sharpe_ratio(model)
            }
            
            logger.info("Portfolio Optimization Completed")
            logger.info(f"Portfolio Strategy: {portfolio_strategy}")
            
            return portfolio_strategy
        
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {}
    
    def _determine_allocation(self, risk_profile: Dict[str, Any]) -> Dict[str, float]:
        """
        Determine optimal asset allocation based on risk profile
        
        Args:
            risk_profile (Dict): Comprehensive risk assessment
        
        Returns:
            Asset allocation percentages
        """
        volatility = risk_profile.get('volatility', 0.2)
        
        # Dynamic allocation based on risk tolerance
        if volatility < 0.1:  # Low volatility
            return {
                'crypto': 0.7,
                'stablecoin': 0.3
            }
        elif volatility < 0.2:  # Moderate volatility
            return {
                'crypto': 0.5,
                'stablecoin': 0.3,
                'cash': 0.2
            }
        else:  # High volatility
            return {
                'crypto': 0.3,
                'stablecoin': 0.4,
                'cash': 0.3
            }
    
    def _calculate_rebalancing_frequency(self, risk_profile: Dict[str, Any]) -> str:
        """
        Determine optimal portfolio rebalancing frequency
        
        Args:
            risk_profile (Dict): Comprehensive risk assessment
        
        Returns:
            Rebalancing frequency
        """
        max_drawdown = risk_profile.get('max_drawdown', 0.1)
        
        if max_drawdown < 0.05:  # Conservative strategy
            return 'monthly'
        elif max_drawdown < 0.1:  # Balanced strategy
            return 'quarterly'
        else:  # Aggressive strategy
            return 'semi-annually'
    
    def _calculate_position_sizing(self, risk_profile: Dict[str, Any]) -> float:
        """
        Calculate risk-adjusted position sizing
        
        Args:
            risk_profile (Dict): Comprehensive risk assessment
        
        Returns:
            Maximum position size percentage
        """
        volatility = risk_profile.get('volatility', 0.2)
        
        # Inverse relationship between volatility and position size
        return max(0.05, 0.2 - (volatility * 0.5))
    
    def _simulate_portfolio_performance(self, model, initial_capital: float) -> float:
        """
        Simulate portfolio performance using trained model
        
        Args:
            model: Trained reinforcement learning model
            initial_capital (float): Starting investment amount
        
        Returns:
            Cumulative Return on Investment (ROI)
        """
        # Placeholder for advanced portfolio performance simulation
        # In a real implementation, this would use the trained model to simulate trades
        return 0.15  # 15% ROI as an example
    
    def _calculate_sharpe_ratio(self, model) -> float:
        """
        Calculate Sharpe ratio for portfolio performance
        
        Args:
            model: Trained reinforcement learning model
        
        Returns:
            Sharpe ratio
        """
        # Placeholder for Sharpe ratio calculation
        # Requires historical returns and risk-free rate
        return 1.5  # Example Sharpe ratio

def main():
    """
    Main execution for crew-based optimization
    """
    # --- Main Function Logging --- #
    main_logger = logging.getLogger("CrewOptimizationManager_Main")
    main_logger.info("Entered main function of crew_optimization_manager.")
    # --- End Main Function Logging --- #
    try:
        # Initialize crew optimization agent
        # Ensure the TRADING_SYMBOL env var is set by the launcher
        trading_symbol = os.environ.get('TRADING_SYMBOL', 'BTC-USD') # Default if not set
        logger.info(f"Initializing CrewOptimizationAgent for symbol: {trading_symbol}")
        crew_agent = CrewOptimizationAgent(symbol=trading_symbol, initial_capital=100000)
        
        # Run optimization pipeline
        logger.info(f"Running optimization pipeline for {trading_symbol}...")
        optimization_results = crew_agent.run_optimization_pipeline()
        
        # Define model save path and directory
        model_save_dir = '/Users/activate/Dev/robinhood-crypto-bot/models'
        model_save_path = os.path.join(model_save_dir, 'ppo_crypto_trader_live.zip')
        log_save_path = '/Users/activate/Dev/robinhood-crypto-bot/logs/optimization_results.pkl'

        # Create directories if they don't exist
        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_save_path), exist_ok=True)

        # Save optimization results dictionary (excluding the model object if too large)
        try:
            results_to_save = optimization_results.copy()
            if 'trained_model' in results_to_save:
                del results_to_save['trained_model'] # Avoid saving large model object in pickle
            joblib.dump(results_to_save, log_save_path)
            logger.info(f"Optimization results dictionary saved to {log_save_path}")
        except Exception as dump_error:
            logger.error(f"Error saving optimization results dictionary: {dump_error}")

        # Explicitly save the trained model
        trained_model = optimization_results.get('trained_model')
        if trained_model:
            logger.info(f"Found trained model object in results. Type: {type(trained_model)}")
            try:
                logger.info(f"Attempting to save model to {model_save_path}...")
                trained_model.save(model_save_path)
                logger.info(f"Model saving process completed for path: {model_save_path}") 
                # Verify file existence immediately after saving attempt
                if os.path.exists(model_save_path):
                    logger.info(f"Verified: Model file exists at {model_save_path}")
                else:
                    logger.error(f"Verification failed: Model file NOT found at {model_save_path} immediately after save call.")
            except Exception as save_error:
                logger.error(f"Error occurred during model.save() operation for {model_save_path}: {save_error}", exc_info=True) 
        elif 'error' not in optimization_results:
             logger.warning("Optimization pipeline finished, but no trained model found in results to save.")
        else:
            logger.error("Optimization pipeline failed, skipping model save.")

        logger.info(f"Crew Optimization Pipeline for {trading_symbol} Completed.")
    
    except Exception as e:
        logger.error(f"Crew optimization failed: {e}")
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()
