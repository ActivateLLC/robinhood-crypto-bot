import optuna
import numpy as np
import os
import json
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from rl_environment import CryptoTradingEnv
from alt_crypto_data import AltCryptoDataProvider
import gymnasium as gym

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('optuna_ppo_tuning.log'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def create_env(df, symbol='BTC-USD', training=True):
    """
    Create a vectorized environment with additional safety checks
    """
    env = DummyVecEnv([lambda: CryptoTradingEnv(df=df, symbol=symbol)])
    
    # Add NaN checking wrapper for numerical stability
    env = VecCheckNan(env)
    
    return env

def objective(trial):
    """
    Objective function for hyperparameter optimization
    """
    # Hyperparameter suggestions with more precise ranges
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 256)
    n_steps = trial.suggest_int('n_steps', 512, 4096)
    gamma = trial.suggest_float('gamma', 0.90, 0.9999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)
    ent_coef = trial.suggest_float('ent_coef', 1e-4, 0.1, log=True)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    
    # Fetch historical data with robust error handling
    data_provider = AltCryptoDataProvider()
    try:
        df = data_provider.fetch_all_historical_data('BTC-USD', interval='1h')
    except Exception as e:
        logger.error(f"Data fetching error: {e}")
        df = data_provider.generate_placeholder_data('BTC-USD', days=30)  # Use the public method
    
    # Create training and evaluation environments
    train_env = create_env(df, training=True)
    eval_env = create_env(df, training=False)
    
    # Validate environment
    try:
        check_env(train_env.envs[0])
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return float('-inf')
    
    # Create model with suggested hyperparameters
    model = PPO(
        'MlpPolicy', 
        train_env, 
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        verbose=0
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=max(100, n_steps),
        deterministic=True,
        render=False
    )
    
    try:
        # Reduced training timesteps for faster debugging
        model.learn(
            total_timesteps=25000, 
            callback=eval_callback
        )
        
        # Evaluate model performance
        mean_reward, std_reward = evaluate_policy(
            model, 
            eval_env, 
            n_eval_episodes=5,
            deterministic=True
        )
        
        logger.info(f"Trial {trial.number} Performance: {mean_reward:.4f} ± {std_reward:.4f}")
        
        return mean_reward
    
    except Exception as e:
        logger.error(f"Training failed for trial {trial.number}: {e}")
        return float('-inf')

def main():
    # Delete existing study to reset
    try:
        os.remove('optuna_ppo_tuning.db')
    except FileNotFoundError:
        pass

    # Create a new study object with more robust storage and pruning
    study = optuna.create_study(
        direction='maximize',
        storage='sqlite:///optuna_ppo_tuning.db',
        study_name='crypto_ppo_tuning',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, 
            n_warmup_steps=10, 
            interval_steps=1
        )
    )
    
    # Run optimization with early stopping and parallel processing
    study.optimize(
        objective, 
        n_trials=50,
        n_jobs=-1,
        timeout=3600,  # 1-hour timeout
        show_progress_bar=True
    )
    
    # Detailed logging of results
    logger.info('Optuna Tuning Complete')
    logger.info('Number of finished trials: %d', len(study.trials))
    
    # Best trial details
    trial = study.best_trial
    logger.info('Best trial:')
    logger.info('  Value: %f', trial.value)
    logger.info('  Params: ')
    for key, value in trial.params.items():
        logger.info(f'    {key}: {value}')
    
    # Save comprehensive trial results
    with open('optuna_ppo_trials_summary.json', 'w') as f:
        json.dump({
            'best_trial': {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params
            },
            'all_trials': [
                {
                    'number': t.number, 
                    'value': t.value, 
                    'params': t.params,
                    'state': str(t.state)
                } for t in study.trials
            ]
        }, f, indent=2)

if __name__ == '__main__':
    main()
