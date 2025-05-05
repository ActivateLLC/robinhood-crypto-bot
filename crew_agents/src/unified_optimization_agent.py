import sys
import os

# --- Add project root to sys.path EARLY ---
# Calculate project root relative to this file's location
# Go up three levels: src -> crew_agents -> robinhood-crypto-bot
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------------------

# --- Now import project modules ---
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Import necessary components using explicit relative imports
from trading_intelligence import TradingIntelligenceEngine
from crypto_trading_env import CryptoTradingEnvironment

import logging
import logging.handlers
import json
import numpy as np
import pandas as pd
import optuna
import torch
import gymnasium as gym
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import traceback
from dotenv import load_dotenv


class UnifiedOptimizationAgent:
    """
    Comprehensive Cryptocurrency Trading Optimization Agent
    Integrates ROI Optimization, Reinforcement Learning, and Advanced Market Analysis
    """
    
    def __init__(
        self, 
        symbols: List[str] = ['BTC-USD', 'ETH-USD', 'BNB-USD'],
        config_path: Optional[str] = None,
        log_dir: Optional[str] = None
    ):
        """
        Initialize Unified Optimization Agent
        
        Args:
            symbols (List[str]): Cryptocurrencies to optimize
            config_path (str, optional): Path to configuration file
            log_dir (str, optional): Directory for logging results
        """
        self.symbols = symbols
        self.log_dir = log_dir or os.path.join(project_root, 'logs', 'unified_optimization')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Determine run mode from environment variables
        self.live_trading = os.getenv('LIVE_TRADING', 'false').lower() == 'true'
        self.broker_type = os.getenv('BROKER_TYPE', 'simulation').lower()
        if self.live_trading and self.broker_type not in ['robinhood', 'ibkr', 'simulation']: # Allow simulation override even if LIVE_TRADING is true
            raise ValueError(f"Unsupported BROKER_TYPE '{self.broker_type}' specified in environment for live trading.")
        if not self.live_trading:
            self.broker_type = 'simulation' # Force simulation if LIVE_TRADING is false

        # Initialize data and intelligence providers
        self.trading_intelligence = TradingIntelligenceEngine()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load or create configuration
        self.config_path = config_path or os.path.join(project_root, 'configs', 'unified_optimization_config.json')
        self.config = self._load_or_create_config()
        # Store lookback_days from config
        self.lookback_days = self.config.get('data_settings', {}).get('lookback_days', 365) # Default to 365 if not found
        self.logger.info(f"Using lookback period: {self.lookback_days} days")
        
        # Performance tracking
        self.performance_history = {symbol: [] for symbol in symbols}
        self.optimization_count = 0
    
    def _setup_logging(self) -> logging.Logger:
        """
        Configure comprehensive logging
        
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger('UnifiedOptimizationAgent')
        logger.setLevel(logging.INFO)
        
        # Rotating file handler
        log_file = os.path.join(self.log_dir, 'unified_optimization.log')
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        logger.addHandler(file_handler)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        
        return logger
    
    def _load_or_create_config(self) -> Dict[str, Any]:
        """
        Load existing configuration or create default
        
        Returns:
            Dict: Configuration parameters
        """
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                # Load existing config, provide defaults for missing keys
                loaded_config = json.load(f)
                loaded_config.setdefault('optimization_settings', {})
                loaded_config['optimization_settings'].setdefault('max_trials', 500)
                loaded_config['optimization_settings'].setdefault('timeout_hours', 24)
                loaded_config['optimization_settings'].setdefault('risk_tolerance', 0.2)
                loaded_config.setdefault('hyperparameters', {})
                loaded_config['hyperparameters'].setdefault('learning_rates', {})
                loaded_config['hyperparameters'].setdefault('batch_sizes', {})
                loaded_config['hyperparameters'].setdefault('gamma_values', {})
                loaded_config['hyperparameters'].setdefault('exploration_rates', {})
                loaded_config.setdefault('data_settings', {})
                loaded_config['data_settings'].setdefault('lookback_days', 365)
                return loaded_config
        
        default_config = {
            'optimization_settings': {
                'max_trials': 500,
                'timeout_hours': 24,
                'risk_tolerance': 0.2
            },
            'hyperparameters': {
                'learning_rates': {'BTC-USD': 0.0008, 'ETH-USD': 0.001, 'BNB-USD': 0.0009},
                'batch_sizes': {'BTC-USD': 128, 'ETH-USD': 256, 'BNB-USD': 192},
                'gamma_values': {'BTC-USD': 0.99, 'ETH-USD': 0.95, 'BNB-USD': 0.97},
                'exploration_rates': {'BTC-USD': 0.1, 'ETH-USD': 0.15, 'BNB-USD': 0.12}
            },
            'data_settings': { 
                'lookback_days': 365
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        return default_config
    
    def _create_trading_environment(self, symbol: str) -> gym.Env:
        """
        Create robust trading environment
        
        Args:
            symbol (str): Cryptocurrency trading pair
        
        Returns:
            gym.Env: Validated trading environment
        """
        self.logger.info(f"Creating environment for {symbol} | Live: {self.live_trading}, Broker: {self.broker_type}")
        try:
            # Pass the live_trading and broker_type flags read during init
            env = CryptoTradingEnvironment(
                symbol=symbol,
                live_trading=self.live_trading,
                broker_type=self.broker_type
            )
            
            # Validate environment
            self._validate_environment(env)
            
            return env
        except Exception as e:
            self.logger.error(f"Environment creation failed for {symbol}: {e}")
            raise
    
    def _validate_environment(self, env: gym.Env):
        """
        Perform comprehensive environment validation
        
        Args:
            env (gym.Env): Trading environment to validate
        
        Raises:
            ValueError: If environment fails validation
        """
        # Observation space validation
        assert isinstance(env.observation_space, gym.spaces.Box), "Invalid observation space"
        
        # Action space validation
        assert isinstance(env.action_space, gym.spaces.Discrete), "Invalid action space"
        
        # Initial state validation
        initial_state = env.reset()
        assert initial_state is not None, "Failed to reset environment"
    
    def optimize_hyperparameters(self, symbol: str) -> Dict[str, Any]:
        """
        Advanced hyperparameter optimization using Optuna
        
        Args:
            symbol (str): Cryptocurrency trading pair
        
        Returns:
            Dict: Best hyperparameters
        """
        def objective(trial):
            try:
                # Hyperparameter suggestions
                hyperparams = {
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                    'n_steps': trial.suggest_int('n_steps', 2048, 16384),
                    'gamma': trial.suggest_uniform('gamma', 0.9, 0.9999),
                    'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 1.0),
                    'ent_coef': trial.suggest_loguniform('ent_coef', 1e-4, 1e-1),
                    'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4)
                }
                
                # Create environment
                env = self._create_trading_environment(symbol)
                
                # Initialize and train model
                model = PPO(
                    "MlpPolicy", 
                    env, 
                    learning_rate=hyperparams['learning_rate'],
                    batch_size=hyperparams['batch_size'],
                    n_steps=hyperparams['n_steps'],
                    gamma=hyperparams['gamma'],
                    gae_lambda=hyperparams['gae_lambda'],
                    ent_coef=hyperparams['ent_coef'],
                    clip_range=hyperparams['clip_range']
                )
                
                # Train model
                model.learn(total_timesteps=20000)
                
                # Evaluate model performance
                mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
                
                return mean_reward
            
            except Exception as e:
                self.logger.error(f"Optimization trial failed: {e}")
                return float('-inf')
        
        # Create and run Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(
            objective, 
            n_trials=self.config['optimization_settings']['max_trials'],
            timeout=self.config['optimization_settings']['timeout_hours'] * 3600
        )
        
        # Log and save best parameters
        best_params = study.best_params
        self.logger.info(f"Best Hyperparameters for {symbol}: {best_params}")
        
        # Update configuration
        self.config['hyperparameters']['learning_rates'][symbol] = best_params['learning_rate']
        self.config['hyperparameters']['batch_sizes'][symbol] = best_params['batch_size']
        
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        return best_params
    
    def run_optimization_pipeline(self):
        """
        Execute comprehensive optimization pipeline
        
        Returns:
            Dict: Optimization results for all symbols
        """
        results = {}
        
        for symbol in self.symbols:
            try:
                # Fetch market data
                self.logger.info(f"Fetching market data for {symbol}...")
                # Removed: market_data = self.data_provider.fetch_price_history(symbol=symbol, days=self.lookback_days)

                # Perform analysis using Trading Intelligence Engine
                self.logger.info(f"Generating trading insights for {symbol}...")
                # Removed: insights_analysis = self.trading_intelligence.generate_comprehensive_trading_insights(historical_data=market_data)

                # Optimize hyperparameters
                best_hyperparams = self.optimize_hyperparameters(symbol)
                
                # Combine results
                results[symbol] = {
                    'best_hyperparameters': best_hyperparams
                }
                
                self.logger.info(f"Optimization completed for {symbol}")
            
            except Exception as e:
                self.logger.error(f"Optimization failed for {symbol}: {e}")
                results[symbol] = {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        # Save comprehensive results
        results_path = os.path.join(
            self.log_dir, 
            f'optimization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def main():
    """
    Main entry point for Unified Optimization Agent
    """
    # Construct the absolute path to the .env file
    dotenv_path = os.path.join(project_root, '.env')
    print(f"DEBUG: Attempting to load .env from: {dotenv_path}") # Add debug print for path
    # Explicitly load environment variables from the specified path
    load_dotenv(dotenv_path=dotenv_path, override=True) # Use override=True just in case
    
    optimization_agent = UnifiedOptimizationAgent()
    results = optimization_agent.run_optimization_pipeline()
    
    # Print key results
    for symbol, data in results.items():
        print(f"\n{symbol} Optimization Results:")
        print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
