import os
import pandas as pd
import numpy as np
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.model_selection import train_test_split
import optuna
import logging
from dotenv import load_dotenv
import traceback
import sys
import matplotlib.pyplot as plt

from rl_environment import CryptoTradingEnv # Assuming rl_environment.py is in the same directory

# --- Constants & Configuration --- #
load_dotenv() # Load env vars if needed (e.g., API keys for data source)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Keep logging level INFO

SYMBOL = "BTC"
# Use a fixed period for repeatable tuning runs
DATA_START_DATE = "2022-01-01" # Start of data period
DATA_END_DATE = "2024-01-01"   # End of data period (exclusive for data fetching)
TRAIN_END_DATE = "2023-07-01"  # End of training data within Optuna (rest is validation)

# --- Hyperparameter Tuning Settings ---
N_TRIALS = 50 # Number of Optuna trials to run
STUDY_NAME = f"ppo-tuning-{SYMBOL.replace('-','_')}-fixedTTM" # New study name for fixed TTM
STORAGE_URL = "sqlite:///optuna_ppo_tuning.db" # Database to store results
TUNING_TIMESTEPS = 25000  # Reduced from 250000 for faster debugging

# --- Evaluation Settings ---
INITIAL_BALANCE = 10000 # Initial balance for evaluation environment within Optuna
LOOKBACK_WINDOW = 30    # Lookback window MUST match environment used in training/live bot
N_EVAL_EPISODES = 5     # Number of episodes to evaluate for each trial

# --- Data Fetching --- #
def get_historical_data(symbol, start_date, end_date):
    ticker = yf.Ticker(symbol)
    # Use 1h data for more granularity during tuning --> Changed back to 1d
    data = ticker.history(start=start_date, end=end_date, interval='1d')
    data.index = data.index.tz_localize(None) # Remove timezone for consistency
    # Ensure standard column names
    data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    data = data[['open', 'high', 'low', 'close', 'volume']]
    data.dropna(inplace=True)
    logging.info(f"Fetched {len(data)} rows of data for {symbol} from {start_date} to {end_date}")
    return data

# --- Optuna Objective Function --- #
def original_objective(trial: optuna.Trial) -> float:
    logging.info(f"\n--- Starting Optuna Trial {trial.number} ---")

    # --- Fetch and Split Data --- #
    try:
        df_full = get_historical_data(SYMBOL, DATA_START_DATE, DATA_END_DATE)
        # Split data into training and validation sets for this trial
        df_train = df_full[:TRAIN_END_DATE]
        df_val = df_full[TRAIN_END_DATE:]

        if df_train.empty or df_val.empty:
            logging.error("Data fetching or splitting resulted in empty DataFrame.")
            return -np.inf # Return a very low value if data is bad

    except Exception as e:
        logging.error(f"Error fetching or processing data: {e}")
        return -np.inf # Penalize trial heavily if data fails

    # --- Hyperparameter Suggestions --- #
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 20])

    model_params = {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
    }
    logging.info(f"Trial {trial.number} Parameters: {model_params}")

    # --- Environment Setup --- #
    try:
        train_env = DummyVecEnv([lambda: CryptoTradingEnv(df_train, initial_balance=INITIAL_BALANCE, lookback_window=LOOKBACK_WINDOW)])
        eval_env = DummyVecEnv([lambda: CryptoTradingEnv(df_val, initial_balance=INITIAL_BALANCE, lookback_window=LOOKBACK_WINDOW)])

        # Wrap environments with VecCheckNan to detect NaN/Inf issues
        train_env = VecCheckNan(train_env, raise_exception=True)
        eval_env = VecCheckNan(eval_env, raise_exception=True)
    except Exception as e:
        logging.error(f"Error creating environment: {e}")
        return -np.inf

    # --- Model Training and Evaluation --- #
    try:
        model = PPO("MlpPolicy", train_env, verbose=0, tensorboard_log=None, **model_params)

        # Use SB3 EvalCallback for evaluation during training
        # We want the *final* performance on the validation set after training
        # So, we'll train fully and then evaluate separately

        model.learn(total_timesteps=TUNING_TIMESTEPS, progress_bar=False)

        # Evaluation
        print(f"\n=== Evaluating Trial {trial.number} ===")
        episode_rewards = []
        for i in range(N_EVAL_EPISODES):
            obs = eval_env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = eval_env.step(action)
                total_reward += reward
                if not np.isfinite(reward):
                    print(f"\n--- NON-FINITE REWARD DETECTED ---")
                    print(f"Episode: {i+1}, Step: {eval_env.current_step}")
                    print(f"Action: {action}")
                    print(f"Observation: {obs}")
                    print(f"Reward: {reward}")
            episode_rewards.append(total_reward)
            print(f"Episode {i+1} Total Reward: {total_reward}")

        mean_reward = np.mean(episode_rewards)
        print(f"Mean Reward for Trial {trial.number}: {mean_reward}")
        logging.info(f"Trial {trial.number} Evaluation Mean Reward: {mean_reward}") # Keep general log too

        # Final check before returning to Optuna
        if np.isnan(mean_reward) or np.isinf(mean_reward):
            logging.warning(f"Mean reward is NaN or Inf ({mean_reward}). Returning -Infinity to Optuna.")
            return -np.inf

        return mean_reward

    except Exception as e:
        logging.error(f"Error during training or evaluation in trial {trial.number}: {e}")
        logging.error(traceback.format_exc()) # Log the full traceback
        # Ensure eval_env is closed if it was created
        if 'eval_env' in locals() and eval_env:
            eval_env.close()
        # Ensure train_env is closed if it was created
        if 'train_env' in locals() and train_env:
            train_env.close()
        # Return -infinity or raise optuna.TrialPruned() if appropriate
        return -float('inf')

def objective(trial):
    try:
        # Existing optimization logic
        val = original_objective(trial)
        
        if not np.isfinite(val):
            raise optuna.exceptions.TrialPruned()
            
        return float(val)
        
    except Exception as e:
        trial.set_user_attr("error", str(e))
        raise optuna.exceptions.TrialPruned()

# --- Run Optuna Study --- #
if __name__ == "__main__":
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(
        study_name=STUDY_NAME, # Use updated name
        storage=STORAGE_URL,
        direction="maximize", # Maximize Differential Sharpe Ratio
        load_if_exists=True # Resume study if it exists
    )

    print("\n=== Starting Optimization ===")
    print(f"Initial Sampler: {study.sampler.__class__.__name__}")
    print(f"Pruner: {study.pruner.__class__.__name__}")
    
    # Add progress reporter callback
    def log_progress(study, trial):
        print(f"\nTrial {trial.number} completed with value: {trial.value}")
        print(f"Best value so far: {study.best_value}")
        print(f"Params: {trial.params}")
        
        # Live monitoring
        if trial.number % 10 == 0:
            plt.figure(figsize=(12,4))
            plt.subplot(131)
            plt.hist([t.value for t in study.trials if t.value is not None and np.isfinite(t.value)])
            plt.title('Reward Distribution')
            
            plt.subplot(132)
            plt.plot([t.number for t in study.trials], [t.value for t in study.trials if t.value is not None])
            plt.title('Trial Progress')
            
            plt.subplot(133)
            plt.bar(['Valid','Invalid'], [
                sum(1 for t in study.trials if t.value is not None and np.isfinite(t.value)),
                sum(1 for t in study.trials if t.value is None or not np.isfinite(t.value))
            ])
            plt.title('Trial Validity')
            
            plt.tight_layout()
            plt.savefig(f'optimization_status_{trial.number}.png')
            plt.close()

    try:
        # Use n_jobs=1 for potentially better stability/logging
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[log_progress], timeout=None, n_jobs=1, gc_after_trial=True)
    except KeyboardInterrupt:
        logging.warning("Optimization stopped manually.")
    except Exception as e:
        logging.error(f"An error occurred during the Optuna study: {e}")

    logging.info("\n--- Optuna Study Complete ---")
    logging.info(f"Number of finished trials: {len(study.trials)}")
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best value (Mean Final Net Worth): {study.best_value:.2f}")
    logging.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logging.info(f"  {key}: {value}")

    # You can now take these best hyperparameters and use them in train_rl_agent.py
    # for a full training run on the complete dataset.

if __name__ == "__main__" and True:  # DEBUG MODE ENABLED
    # Debug test
    from rl_environment import CryptoTradingEnv
    df = get_historical_data(SYMBOL, "2023-01-01", "2023-03-01")
    env = CryptoTradingEnv(df)
    
    print("\n=== Running Environment Test ===")
    obs = env.reset()
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(f"Step {i}: Action={action}, Reward={reward:.4f}, Done={done}")
        if not np.isfinite(reward):
            print(f"\n--- INVALID REWARD AT STEP {i} ---")
            print(f"Observation: {obs}")
            print(f"Portfolio State: {env._get_portfolio_state()}")
            break
        if done:
            obs = env.reset()
