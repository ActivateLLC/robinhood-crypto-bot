# crew_agents/src/model_retrainer.py
import time
import logging
import os
import json
from typing import Dict, Any
import pandas as pd
from stable_baselines3 import PPO # Assuming PPO, adjust if different
from stable_baselines3.common.env_util import make_vec_env
import sys # For path adjustments

# Assuming these are correctly importable after path adjustments if needed
# Add project root to sys.path to find alt_crypto_data
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from alt_crypto_data import AltCryptoDataProvider
from crypto_trading_env import CryptoTradingEnvironment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelRetrainer:
    def __init__(
        self,
        model_path: str,
        experience_log_file: str,
        env_config: Dict[str, Any], # Config needed to recreate env (symbol, window_size, etc.)
        data_provider: AltCryptoDataProvider, # Pass the same provider instance
        retrain_trigger_size: int = 1000, # e.g., retrain every 1000 new experiences logged
        retrain_interval_seconds: int = 3600, # e.g., or check every hour
        retrain_steps: int = 10000, # Number of timesteps for each retraining session
        learning_rate: float = 3e-4 # Or load from model/config
    ):
        self.model_path = model_path
        self.experience_log_file = experience_log_file
        self.env_config = env_config
        self.data_provider = data_provider
        self.retrain_trigger_size = retrain_trigger_size
        self.retrain_interval_seconds = retrain_interval_seconds
        self.retrain_steps = retrain_steps
        self.learning_rate = learning_rate # Store learning rate

        self.last_log_size = self._get_log_size()
        self.last_retrain_time = time.time()

        # Load the model
        if os.path.exists(self.model_path):
            logger.info(f"Loading existing model from {self.model_path}")
            # We load the model structure here, but env needs to be set before learning
            # Pass custom_objects if learning_rate needs to be preserved/overridden during load
            self.model = PPO.load(self.model_path, custom_objects={'learning_rate': self.learning_rate})
        else:
            logger.error(f"Model file not found at {self.model_path}. Cannot initialize retrainer.")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def _get_log_size(self) -> int:
        """Gets the current number of lines (experiences) in the log file."""
        try:
            with open(self.experience_log_file, 'r') as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            logger.warning(f"Experience log file not found: {self.experience_log_file}. Starting count at 0.")
            return 0

    def _check_retrain_conditions(self) -> bool:
        """Checks if retraining should be triggered."""
        current_log_size = self._get_log_size()
        time_since_last_retrain = time.time() - self.last_retrain_time

        size_condition_met = (current_log_size - self.last_log_size) >= self.retrain_trigger_size
        time_condition_met = time_since_last_retrain >= self.retrain_interval_seconds

        if size_condition_met:
            logger.info(f"Retrain triggered: Log size increased by >= {self.retrain_trigger_size} entries ({self.last_log_size} -> {current_log_size}).")
            return True
        # Optional: Add time-based trigger as well
        # if time_condition_met:
        #     logger.info(f"Retrain triggered: Interval of {self.retrain_interval_seconds}s passed.")
        #     return True

        return False

    def retrain(self):
        """Performs the retraining process."""
        logger.info("Starting retraining cycle...")

        try:
            # 1. Recreate the environment with potentially updated data
            logger.info("Creating new environment instance for retraining...")
            env_kwargs = self.env_config.copy()
            env_kwargs['data_provider'] = self.data_provider # Pass the provider instance
            
            # Ensure env_kwargs keys match the CryptoTradingEnvironment __init__ params
            # Example check (add more as needed):
            required_keys = ['symbol', 'initial_capital', 'lookback_window']
            if not all(key in env_kwargs for key in required_keys):
                 logger.error(f"Missing required environment config keys: {required_keys}. Have: {env_kwargs.keys()}")
                 return # Skip retraining if config is invalid

            # Create a vectorized environment for potentially faster training
            # Use a lambda to pass arguments to the constructor
            vec_env = make_vec_env(lambda: CryptoTradingEnvironment(**env_kwargs), n_envs=1)

            # 2. Set the environment for the loaded model
            self.model.set_env(vec_env)
            # Re-set learning rate if needed
            self.model.learning_rate = self.learning_rate
            logger.info(f"Set environment for model. Learning rate: {self.model.learning_rate}")

            # 3. Continue learning
            logger.info(f"Continuing learning for {self.retrain_steps} timesteps...")
            self.model.learn(
                total_timesteps=self.retrain_steps,
                reset_num_timesteps=False # Crucial for continuous learning
            )
            logger.info("Learning complete.")

            # 4. Save the updated model
            # Ensure the directory exists before saving
            model_dir = os.path.dirname(self.model_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            logger.info(f"Saving updated model to {self.model_path}")
            self.model.save(self.model_path)

            # 5. Update state for next check
            self.last_log_size = self._get_log_size()
            self.last_retrain_time = time.time()
            logger.info("Retraining cycle finished successfully.")

        except Exception as e:
            logger.exception(f"Error during retraining cycle: {e}")
            # Optionally, attempt to close the vec_env if it was created
            try:
                if 'vec_env' in locals() and vec_env is not None:
                    vec_env.close()
            except Exception as close_e:
                logger.error(f"Error closing vec_env after retraining failure: {close_e}")

    def run(self):
        """Main loop to periodically check and trigger retraining."""
        logger.info(f"ModelRetrainer started. Monitoring '{self.experience_log_file}'. Checking every {self.retrain_interval_seconds / 10:.1f}s.")
        while True:
            try:
                if self._check_retrain_conditions():
                    self.retrain()
                else:
                    logger.debug("Retrain conditions not met. Sleeping...")

                # Sleep before next check
                # Use a fraction of the interval for more responsive checking if desired
                time.sleep(max(10, self.retrain_interval_seconds / 10)) # Sleep at least 10s
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Shutting down ModelRetrainer.")
                break
            except Exception as loop_e:
                logger.exception(f"Unexpected error in main run loop: {loop_e}. Retrying after delay.")
                time.sleep(60) # Wait a minute before retrying the loop

# Example usage (likely called from a separate script/process)
if __name__ == '__main__':
    # --- Configuration --- Requires adjustment based on actual project structure/paths
    # Determine base path relative to this file's location
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, '..', '..')) # Goes up two levels
    
    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "ppo_crypto_trader_live.zip")
    EXPERIENCE_LOG = os.path.join(PROJECT_ROOT, "logs", "live_experience.jsonl")
    ENV_LOG_PATH = os.path.join(PROJECT_ROOT, "logs") # For env logging if needed
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(EXPERIENCE_LOG), exist_ok=True)
    os.makedirs(ENV_LOG_PATH, exist_ok=True)

    SYMBOL = "BTC-USD" # Or load from config
    INITIAL_BALANCE = 10000
    WINDOW_SIZE = 24 # Match the value used in the environment __init__ defaults if applicable
    
    # Need to instantiate the DataProvider
    # Ensure NEWS_API_KEY is in .env in the root folder
    # Path adjustment already done at the top of the file
    data_provider_instance = AltCryptoDataProvider()

    # Environment configuration dictionary - MUST match CryptoTradingEnvironment __init__ params
    environment_config = {
        "symbol": SYMBOL,
        # "data_provider": data_provider_instance, # Added dynamically in retrain()
        "initial_capital": INITIAL_BALANCE,
        "lookback_window": WINDOW_SIZE,
        "trading_fee": 0.0005,
        "max_episode_steps": 1000, # Or match training
        "log_level": "DEBUG", # More verbose logging for the env during retraining
        "live_trading_mode": False # Typically retrain in simulation mode
        # Add other params like broker_config=None if needed by __init__
    }

    # Check if a model file exists to load, otherwise this will fail
    if not os.path.exists(MODEL_SAVE_PATH):
        logger.error(f"No model found at {MODEL_SAVE_PATH}. Please train an initial model first.")
        sys.exit(1) # Exit if no model to retrain
        
    # Check if the log file exists, create if not
    if not os.path.exists(EXPERIENCE_LOG):
        logger.warning(f"Experience log {EXPERIENCE_LOG} not found. Creating empty file.")
        with open(EXPERIENCE_LOG, 'w') as f:
            pass # Create empty file

    retrainer = ModelRetrainer(
        model_path=MODEL_SAVE_PATH,
        experience_log_file=EXPERIENCE_LOG,
        env_config=environment_config,
        data_provider=data_provider_instance,
        retrain_trigger_size=50, # Trigger sooner for testing
        retrain_interval_seconds=300, # Check every 5 mins for testing
        retrain_steps=1000, # Fewer steps for testing
        learning_rate=1e-4 # Example LR, adjust as needed
    )

    # Run the retraining loop
    retrainer.run()
