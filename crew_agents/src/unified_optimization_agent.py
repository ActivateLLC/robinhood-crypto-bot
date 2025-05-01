import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import optuna
import torch
import gymnasium as gym
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import traceback

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import custom modules
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from crew_agents.src.crypto_trading_env import CryptoTradingEnvironment
from alt_crypto_data import AltCryptoDataProvider
from trading_intelligence import TradingIntelligenceEngine

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
        
        # Initialize data and intelligence providers
        self.data_provider = AltCryptoDataProvider()
        self.trading_intelligence = TradingIntelligenceEngine()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load or create configuration
        self.config_path = config_path or os.path.join(project_root, 'configs', 'unified_optimization_config.json')
        self.config = self._load_or_create_config()
        
        # Performance tracking
        self.performance_history = {symbol: [] for symbol in symbols}
    
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
                return json.load(f)
        
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
        try:
            env = CryptoTradingEnvironment(symbol=symbol)
            
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
                market_data = self.data_provider.fetch_historical_data(symbol)
                
                # Perform ROI analysis
                roi_analysis = self.trading_intelligence.calculate_roi(market_data)
                
                # Optimize hyperparameters
                best_hyperparams = self.optimize_hyperparameters(symbol)
                
                # Combine results
                results[symbol] = {
                    'roi_analysis': roi_analysis,
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
    optimization_agent = UnifiedOptimizationAgent()
    results = optimization_agent.run_optimization_pipeline()
    
    # Print key results
    for symbol, data in results.items():
        print(f"\n{symbol} Optimization Results:")
        print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
