import logging
import logging.handlers
import sys
import os

# Configure logging AS EARLY AS POSSIBLE
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.StreamHandler(sys.stdout), 
        logging.FileHandler('training_debug.log', mode='w', encoding='utf-8')
    ]
)

import numpy as np
import pandas as pd
import gymnasium as gym 
import traceback
import time
from stable_baselines3.common.callbacks import BaseCallback, CallbackList 

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path: 
    sys.path.insert(0, project_root)

# Import custom modules AFTER basicConfig
from alt_crypto_data import AltCryptoDataProvider
from rl_environment import CryptoTradingEnv

# Stable Baselines imports AFTER basicConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

# Get the logger for the current module (__main__)
logger = logging.getLogger(__name__)

# Configure additional logging for training_progress.log (specific to this main script's logger)
log_file = 'training_progress.log'
file_handler = logging.handlers.RotatingFileHandler(
    log_file, 
    maxBytes=10*1024*1024,  
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)

# --- Configuration ---
SYMBOL = "BTC-USD"  
DATA_DAYS = 180     
INITIAL_BALANCE = 10000  
LOOKBACK_WINDOW = 30     
TRAINING_TIMESTEPS = 50000  
MODEL_ALGO = "PPO"        
TRANSACTION_FEE = 0.001   
DSR_WINDOW = 30           
EVAL_EPISODES = 5
SAVE_FREQ = 10000

# Directory for saved RL models
MODEL_SAVE_DIR = "trained_models"

# Create save directory if it doesn't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def create_environment(symbol='BTC-USD', use_sentiment=False):
    """
    Create a Crypto Trading Environment with configurable parameters
    
    Args:
        symbol (str): Trading symbol (e.g., 'BTC-USD')
        use_sentiment (bool): Whether to use sentiment analysis
    
    Returns:
        CryptoTradingEnv: Configured trading environment
    """
    # Configuration parameters
    initial_balance = 10000
    transaction_fee = 0.001
    lookback_window = 30
    
    # Create environment
    env = CryptoTradingEnv(
        symbol=symbol, 
        initial_balance=initial_balance,
        transaction_fee_percent=transaction_fee,
        data_period='6mo',
        data_interval='1h',
        use_sentiment=use_sentiment,
        lookback_window=lookback_window
    )
    
    logger.info("Environment created successfully.")
    return env

def train_ppo_agent(symbol='BTC-USD', total_timesteps=50000, use_sentiment=True):
    """
    Train PPO agent with optional sentiment analysis integration
    
    Args:
        symbol (str): Trading pair symbol
        total_timesteps (int): Total training timesteps
        use_sentiment (bool): Enable sentiment-based reward modification
    
    Returns:
        PPO: Trained PPO model
    """
    # Create environment
    env = create_environment(symbol, use_sentiment)
    
    # Configure model hyperparameters
    model_params = {
        'learning_rate': 0.0008259,
        'batch_size': 128,
        'n_steps': 4096,
        'gamma': 0.920044,
        'gae_lambda': 0.835314,
        'ent_coef': 0.000284,
        'clip_range': 0.379411
    }
    
    # Initialize and train PPO model
    model = PPO('MlpPolicy', env, verbose=1, **model_params)
    model.learn(total_timesteps=total_timesteps)
    
    # Save trained model
    model_filename = f'trained_models/{symbol}_ppo_model_{total_timesteps}_sentiment_{use_sentiment}.zip'
    model.save(model_filename)
    
    return model

def train_agent(symbol_to_train=SYMBOL, custom_hyperparams=None, model_name_suffix=""):
    """Fetches data, creates environment, trains, and saves the RL agent."""
    logger.info(f"Starting RL agent training for {symbol_to_train}...")
    logger.info(f"Fetching {DATA_DAYS} days of historical data for {symbol_to_train}...")

    # Define model filename and path within the function for clarity
    _model_name_suffix = f"_{model_name_suffix}" if model_name_suffix else ""
    model_filename = f"{symbol_to_train.replace('-', '_')}_{MODEL_ALGO}{_model_name_suffix}_{TRAINING_TIMESTEPS}steps.zip"
    model_save_path = os.path.join(MODEL_SAVE_DIR, model_filename)

    try:
        # 1. Load Data
        logger.info("Initializing data provider...")
        
        # Fetch historical price data
        data_provider = AltCryptoDataProvider()
        
        try:
            df_raw = data_provider.fetch_price_history(symbol=symbol_to_train, days=DATA_DAYS)
            
            # Robust data validation
            if df_raw is None:
                logger.error(f"Could not fetch sufficient data for {symbol_to_train}. Generating synthetic data.")
                # Create synthetic data
                dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='1h')
                df_raw = pd.DataFrame({
                    'open': np.random.normal(50000, 5000, 500),
                    'high': np.random.normal(50000, 5000, 500) + np.abs(np.random.normal(0, 1000, 500)),
                    'low': np.random.normal(50000, 5000, 500) - np.abs(np.random.normal(0, 1000, 500)),
                    'close': np.random.normal(50000, 5000, 500),
                    'volume': np.abs(np.random.normal(1000, 500, 500))
                }, index=dates)
                
                logger.warning("Using synthetic data for training.")
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            logger.error(traceback.format_exc())
            return

        # 2. Create Environment
        logger.info("Initializing Crypto Trading Environment...")
        try:
            env = create_environment(symbol_to_train, use_sentiment=True)
            
            # Wrap in DummyVecEnv for Stable Baselines3
            env = DummyVecEnv([lambda: env])
            
            logger.info("Environment created successfully.")
        except Exception as e:
            logger.error(f"Environment creation error: {e}")
            logger.error(traceback.format_exc())
            raise

        # 3. Callbacks
        # Stop training if no improvement in performance
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=3, 
            min_evals=1, 
            verbose=1
        )
        
        # Evaluation callback
        eval_callback = EvalCallback(
            env, 
            best_model_save_path='./models/',
            log_path='./logs/',
            eval_freq=SAVE_FREQ,
            deterministic=True,
            render=False,
            n_eval_episodes=EVAL_EPISODES,
            callback_after_eval=stop_callback
        )

        # 4. Model Configuration
        base_model_params = {
            'policy': 'MlpPolicy',
            'env': env,
            'verbose': 1,
            'seed': 42
        }

        # Default hyperparameters (can be overridden by custom_hyperparams)
        default_hyperparams = {
            'learning_rate': 0.0008259,  
            'batch_size': 128,
            'n_steps': 4096,
            'gamma': 0.920044,
            'gae_lambda': 0.835314,
            'ent_coef': 0.000284,
            'clip_range': 0.379411,
        }

        model_params = {**base_model_params, **default_hyperparams}

        if custom_hyperparams:
            logger.info(f"Applying custom hyperparameters: {custom_hyperparams}")
            # Ensure all required keys are present in custom_hyperparams or use defaults
            # For PPO, essential ones like learning_rate, batch_size, n_steps etc. are typically tuned.
            model_params.update(custom_hyperparams) 

        logger.info(f"Using model parameters: {model_params}")

        # Create PPO model
        model = PPO(**model_params)

        # 5. Training
        try:
            # Custom callback for detailed logging
            class DetailedTrainingCallback(BaseCallback):
                def __init__(self, verbose=1):
                    super().__init__(verbose)
                    self.start_time = time.time()
                    self.episode_rewards = []
                    self.total_episodes = 0

                def _on_training_start(self) -> None:
                    logger.info("Training started")

                def _on_step(self) -> bool:
                    # Log progress every 1000 steps
                    if hasattr(self, 'num_timesteps') and self.num_timesteps % 1000 == 0:
                        current_time = time.time()
                        elapsed_time = current_time - self.start_time
                        steps_per_second = self.num_timesteps / elapsed_time
                        
                        progress_percentage = (self.num_timesteps / TRAINING_TIMESTEPS) * 100
                        
                        logger.info(f"Training Progress: {progress_percentage:.2f}%")
                        logger.info(f"Step {self.num_timesteps}/{TRAINING_TIMESTEPS}")
                        logger.info(f"Elapsed Time: {elapsed_time:.2f} seconds")
                        logger.info(f"Steps per Second: {steps_per_second:.2f}")
                    
                    return True

                def _on_rollout_end(self) -> None:
                    self.total_episodes += 1
                    
                    # Log episode details periodically
                    if self.total_episodes % 10 == 0:
                        logger.info(f"Total Episodes Completed: {self.total_episodes}")

                def _on_training_end(self) -> None:
                    logger.info("Training completed")

            # Create detailed training callback
            detailed_callback = DetailedTrainingCallback()

            # Combine callbacks
            combined_callback = CallbackList([detailed_callback, eval_callback])

            model.learn(
                total_timesteps=TRAINING_TIMESTEPS,
                callback=combined_callback,
                log_interval=10,
                reset_num_timesteps=True
            )
            
            # Save final model
            model.save(model_save_path)
            logger.info(f"Model saved to {model_save_path}")
            
            # Log final training summary
            logger.info("Training Summary:")
            logger.info(f"Total Training Steps: {TRAINING_TIMESTEPS}")
            if hasattr(detailed_callback, 'total_episodes'): 
                logger.info(f"Total Training Episodes: {detailed_callback.total_episodes}")
                if detailed_callback.episode_rewards: 
                    logger.info(f"Average Episode Reward: {np.mean(detailed_callback.episode_rewards):.2f}")
                    logger.info(f"Best Episode Reward: {np.max(detailed_callback.episode_rewards):.2f}")
                else:
                    logger.info("No episode rewards recorded.")
            else:
                logger.info("Detailed callback did not record episode information.")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(traceback.format_exc())
        finally:
            env.close()
            logger.info("Training script finished.")

    except Exception as e:
        logger.error(f"Training error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Optimized hyperparameters for BTC-USD from May 3rd run
    optimized_btc_hyperparams = {
        "learning_rate": 0.0011398979077466161,
        "batch_size": 32,  
        "n_steps": 9074,
        "gamma": 0.985762576774885,
        "gae_lambda": 0.810001434716043,
        "ent_coef": 0.0021744055223654243,
        "clip_range": 0.2929392092750005
        # 'policy' and other fixed params like 'verbose', 'seed', 'env' are handled in train_agent
    }

    train_agent(
        symbol_to_train="BTC-USD", 
        custom_hyperparams=optimized_btc_hyperparams, 
        model_name_suffix="optimized_may3"
    )
