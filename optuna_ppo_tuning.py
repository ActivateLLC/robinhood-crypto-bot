import os
import sys
import logging
import numpy as np
import pandas as pd
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import data provider and environment
from alt_crypto_data import AltCryptoDataProvider
from rl_environment import CryptoTradingEnv

# Ensure the current directory is in the Python path
sys.path.append(os.getcwd())

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('optuna_ppo_tuning.log'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Supported cryptocurrencies for multi-asset training
CRYPTO_SYMBOLS = ['BTC-USD']  # Simplified to BTC-only

def create_env(df, symbol='BTC-USD', training=True):
    """
    Create a vectorized environment with additional safety checks
    """
    def env_creator():
        # Temporarily modify global data loading
        import rl_environment
        original_load_method = rl_environment.CryptoTradingEnv._load_and_prepare_data
        
        def mock_load_data(self, symbol=None, period=None, interval=None):
            self.df_processed = df.copy()
            return self.df_processed
        
        rl_environment.CryptoTradingEnv._load_and_prepare_data = mock_load_data
        
        env = CryptoTradingEnv(
            symbol=symbol, 
            initial_balance=10000,
            transaction_fee_percent=0.001,
            data_period='6mo',
            data_interval='1h'
        )
        
        # Restore original method
        rl_environment.CryptoTradingEnv._load_and_prepare_data = original_load_method
        
        return env
    
    env = DummyVecEnv([env_creator])
    
    # Add NaN checking wrapper for numerical stability
    env = VecCheckNan(env)
    
    return env

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    
    Args:
        trial (optuna.Trial): Optuna trial object for hyperparameter suggestion
    
    Returns:
        float: Evaluation metric (total portfolio value or Sharpe ratio)
    """
    # Hyperparameters to optimize with more robust sampling
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 64, 256, log=True)
    n_steps = trial.suggest_int('n_steps', 1024, 4096, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)
    ent_coef = trial.suggest_float('ent_coef', 1e-4, 1.0, log=True)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    
    # Configuration parameters
    symbol = 'BTC-USD'
    initial_balance = 10000
    
    # Create environment
    def create_env(symbol, training=True):
        env = CryptoTradingEnv(
            symbol=symbol,
            initial_balance=initial_balance,
            transaction_fee_percent=0.001,
            data_period='6mo',
            data_interval='1h'
        )
        return env
    
    # Vectorize environment
    env = DummyVecEnv([lambda: create_env(symbol)])
    
    # Create PPO model
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        verbose=0,
        seed=42
    )
    
    # Train model
    try:
        model.learn(total_timesteps=50000)
        
        # Evaluate model
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 1000:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
        
        # Close environment
        env.close()
        
        # Return negative total reward for minimization
        return -total_reward
    except Exception as e:
        # Log the error for debugging
        print(f"Trial {trial.number} failed: {e}")
        return float('inf')  # Penalize failed trials

def main():
    # Create a study object and specify the direction is 'maximize'
    study = optuna.create_study(
        study_name='PPO_Crypto_Tuning', 
        direction='maximize',
        storage='sqlite:///optuna_crypto_study.db',
        load_if_exists=True
    )
    
    # Run the optimization
    study.optimize(objective, n_trials=50)  # Increased trials
    
    # Print best trial details
    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

if __name__ == '__main__':
    main()
