import numpy as np
import pandas as pd
import logging
import pickle
import sys
import os
import random
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from gymnasium.utils.env_checker import check_env as gym_check_env

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import custom modules
from alt_crypto_data import AltCryptoDataProvider
from rl_environment import CryptoTradingEnv

# Stable Baselines imports
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_debug.log', mode='w', encoding='utf-8')
    ]
)
# Configure logger for specific modules
logging.getLogger('rl_environment').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Configuration ---
SYMBOL = "BTC" # Example symbol
DATA_DAYS = 730      # How many days of historical data to use for training (e.g., 2 years)
INITIAL_BALANCE = 10000 # Starting balance for the training environment
LOOKBACK_WINDOW = 30    # Lookback window for the environment's observation
TRAINING_TIMESTEPS = 50000 # Reduced from 250,000 for faster debugging
MODEL_ALGO = "PPO"        # Reverting to PPO algorithm
TRANSACTION_FEE = 0.001   # Transaction fee
DSR_WINDOW = 30           # DSR window
EVAL_EPISODES = 10
SAVE_FREQ = 10000

# --- File Paths ---
# Directory for saved RL models (trained agents)
MODEL_SAVE_DIR = "trained_models"
MODEL_FILENAME = f"{SYMBOL.replace('-','_')}_{MODEL_ALGO}_tuned_{TRAINING_TIMESTEPS}steps" # Reflects tuned params and 250k steps
model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME) # Path for the trained model .zip file

# Directory and path for normalization parameters (used by crypto_bot.py)
PARAMS_SAVE_DIR = "models" # Matches the directory expected by crypto_bot.py
PARAMS_FILENAME = "normalization_params.pkl" # Matches the default filename expected by crypto_bot.py
params_save_path = os.path.join(PARAMS_SAVE_DIR, PARAMS_FILENAME)

# Ensure the save directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PARAMS_SAVE_DIR, exist_ok=True) # Create the parameters directory too

def train_agent():
    """Fetches data, creates environment, trains, and saves the RL agent."""
    logger.info(f"Starting RL agent training for {SYMBOL}...")

    # 1. Load Data
    logger.info(f"Fetching {DATA_DAYS} days of historical data for {SYMBOL}...")
    data_provider = AltCryptoDataProvider()
    # Add error handling for data fetching
    try:
        # Note: fetch_price_history returns OHLCV
        df_raw = data_provider.fetch_price_history(symbol=SYMBOL, days=DATA_DAYS)
        
        # Robust data validation
        if df_raw is None:
            logger.error(f"Could not fetch sufficient data for {SYMBOL}. Generating synthetic data.")
            # Create synthetic data
            dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='1h')
            df_raw = pd.DataFrame({
                'open': np.random.normal(50000, 5000, 500),
                'high': np.random.normal(50000, 5000, 500) + np.abs(np.random.normal(0, 1000, 500)),
                'low': np.random.normal(50000, 5000, 500) - np.abs(np.random.normal(0, 1000, 500)),
                'close': np.random.normal(50000, 5000, 500),
                'volume': np.abs(np.random.normal(1000, 500, 500))
            }, index=dates)
            
            logger.warning("Using synthetic data for training due to data fetch failure.")
        
        # Ensure columns are in the correct format
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        
        # Drop rows with NaN values
        df_raw.dropna(subset=required_cols, inplace=True)
        
        if df_raw.empty:
            logger.error("No valid data available after cleaning. Cannot proceed.")
            return
        
        logger.info(f"Successfully fetched {len(df_raw)} data points.")
    except Exception as e:
         logger.error(f"Error fetching data: {e}")
         return

    # 2. Create Environment
    logger.info("Initializing Crypto Trading Environment...")
    try:
        env = CryptoTradingEnv(
            initial_balance=INITIAL_BALANCE,
            transaction_fee_percent=TRANSACTION_FEE,
            lookback_window=LOOKBACK_WINDOW,
            dsr_window=DSR_WINDOW,
            sharpe_epsilon=1e-10,
            seed=42  # Fixed seed for reproducibility
        )
        logger.info("Environment created successfully.")
        logger.info(f"Observation space shape: {env.observation_space.shape}")
        logger.info(f"Action space: {env.action_space}")
        
        # Validate environment
        sb3_check_env(env)
        gym_check_env(env)
        logger.info("Environment validation passed.")
    except Exception as e:
        logger.error(f"Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        return

    # Evaluation configuration
    eval_callback = EvalCallback(
        env, 
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=SAVE_FREQ,
        deterministic=True,  # Consistent evaluation
        render=False,
        n_eval_episodes=EVAL_EPISODES
    )

    # Training configuration
    model_params = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': 5e-5,  # Further reduced learning rate
        'batch_size': 256,      # Increased batch size
        'gamma': 0.99,
        'n_steps': 2048,        # Adjusted n_steps for PPO
        'ent_coef': 0.01,       # Adjusted entropy coefficient for PPO
        'vf_coef': 0.5,         # Adjusted value function coefficient for PPO
        'max_grad_norm': 0.5,   # Adjusted max gradient norm for PPO
        'gae_lambda': 0.95,     # Adjusted GAE lambda for PPO
        'clip_range': 0.2,      # Adjusted clip range for PPO
        'verbose': 1,
        'seed': 42,
        'device': 'auto'
    }

    # Create PPO model with explicit configuration
    model = PPO(**model_params)

    # Perform training with callbacks
    try:
        model.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            callback=eval_callback,
            log_interval=10,
            reset_num_timesteps=True  # Reset timestep counter
        )
        
        # Save final model
        model_save_path = os.path.join(
            'trained_models', 
            f'BTC_PPO_advanced_risk_management_{TRAINING_TIMESTEPS}steps.zip'
        )
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
        # Evaluate final model performance
        mean_reward = np.mean([
            np.sum(evaluate_policy(model, env, n_eval_episodes=1, deterministic=True)[0]) 
            for _ in range(10)
        ])
        
        # Compute final portfolio metrics
        current_price = env._get_current_price()
        final_net_worth = env.portfolio_value + (env.crypto_held * current_price)
        
        logger.info("\n--- Training Performance Summary ---")
        logger.info(f"Initial Portfolio Value: ${env.initial_portfolio_value:.2f}")
        logger.info(f"Final Portfolio Value: ${final_net_worth:.2f}")
        logger.info(f"Total Return: {((final_net_worth - env.initial_portfolio_value) / env.initial_portfolio_value * 100):.2f}%")
        logger.info(f"Total Trades: {env.total_trades}")
        logger.info(f"Mean Reward: {mean_reward:.2f}")
        
        # Optional: Log to file for future reference
        with open(f'./training_results_{MODEL_FILENAME}.txt', 'w') as f:
            f.write(f"Initial Portfolio Value: ${env.initial_portfolio_value:.2f}\n")
            f.write(f"Final Portfolio Value: ${final_net_worth:.2f}\n")
            f.write(f"Total Return: {((final_net_worth - env.initial_portfolio_value) / env.initial_portfolio_value * 100):.2f}%\n")
            f.write(f"Total Trades: {env.total_trades}\n")
            f.write(f"Mean Reward: {mean_reward:.2f}\n")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        # Optionally save partially trained model? For now, just exit.
        env.close()
        return

    # 5. Clean up
    env.close()
    logger.info("Training script finished.")

if __name__ == "__main__":
    train_agent()
