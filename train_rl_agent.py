import numpy as np
import pandas as pd
import logging
import logging.handlers
import sys
import os
import gymnasium as gym
import traceback
import time
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import custom modules
from alt_crypto_data import AltCryptoDataProvider
from rl_environment import CryptoTradingEnv

# Stable Baselines imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_debug.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Configure logging with file rotation
log_file = 'training_progress.log'
file_handler = logging.handlers.RotatingFileHandler(
    log_file, 
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)

# Ensure logging to console and file
logger.setLevel(logging.INFO)

# --- Configuration ---
SYMBOL = "BTC-USD"  # Cryptocurrency trading pair
DATA_DAYS = 180     # How many days of historical data to use for training
INITIAL_BALANCE = 10000  # Starting balance for the training environment
LOOKBACK_WINDOW = 30     # Lookback window for the environment's observation
TRAINING_TIMESTEPS = 50000  # Number of training timesteps
MODEL_ALGO = "PPO"        # Reinforcement Learning Algorithm
TRANSACTION_FEE = 0.001   # Transaction fee percentage
DSR_WINDOW = 30           # Dynamic Stop-Loss and Risk Window
EVAL_EPISODES = 5
SAVE_FREQ = 10000

# Directory for saved RL models
MODEL_SAVE_DIR = "trained_models"
MODEL_FILENAME = f"{SYMBOL.replace('-','_')}_{MODEL_ALGO}_tuned_{TRAINING_TIMESTEPS}steps"
model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)

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
    
    # Compute total number of features
    total_features = (
        6 +  # Price and Trend Features
        4 +  # Heikin-Ashi Features
        3 +  # TTM Scalper Features
        4 +  # Momentum Indicators
        4 +  # Volatility and Volume Features
        6 +  # Bollinger and Keltner Channel Features
        4    # Portfolio Features
    )
    
    # Modify observation space to match total features
    env.observation_space = gym.spaces.Box(
        low=-1, 
        high=1, 
        shape=(total_features,), 
        dtype=np.float32
    )
    
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

def train_agent():
    """Fetches data, creates environment, trains, and saves the RL agent."""
    logger.info(f"Starting RL agent training for {SYMBOL}...")
    logger.info(f"Fetching {DATA_DAYS} days of historical data for {SYMBOL}...")

    try:
        # 1. Load Data
        logger.info("Initializing data provider...")
        
        # Fetch historical price data
        data_provider = AltCryptoDataProvider()
        
        try:
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
                
                logger.warning("Using synthetic data for training.")
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            logger.error(traceback.format_exc())
            return

        # 2. Create Environment
        logger.info("Initializing Crypto Trading Environment...")
        try:
            env = create_environment(SYMBOL, use_sentiment=True)
            
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
        model_params = {
            'policy': 'MlpPolicy',
            'env': env,
            'learning_rate': 0.0008259,  # From Optuna tuning
            'batch_size': 128,
            'n_steps': 4096,
            'gamma': 0.920044,
            'gae_lambda': 0.835314,
            'ent_coef': 0.000284,
            'clip_range': 0.379411,
            'verbose': 1,
            'seed': 42
        }

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
            logger.info(f"Total Training Episodes: {detailed_callback.total_episodes}")
            logger.info(f"Average Episode Reward: {np.mean(detailed_callback.episode_rewards):.2f}")
            logger.info(f"Best Episode Reward: {np.max(detailed_callback.episode_rewards):.2f}")
            
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
    train_agent()
