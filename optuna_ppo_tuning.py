import os
import sys
import logging
import numpy as np
import pandas as pd
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import data provider and environment
from alt_crypto_data import AltCryptoDataProvider
from rl_environment import CryptoTradingEnv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('optuna_ppo_tuning.log'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Supported cryptocurrencies for multi-asset training
CRYPTO_SYMBOLS = ['BTC-USD']  

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    
    Args:
        trial (optuna.Trial): Optuna trial object for hyperparameter suggestion
    
    Returns:
        float: Evaluation metric (e.g., Sharpe Ratio)
    """
    logger.info(f"Starting Trial {trial.number}")
    try:
        # Hyperparameters to optimize
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_int('batch_size', 32, 512, log=True) 
        n_steps = trial.suggest_int('n_steps', 512, 4096, log=True) 
        gamma = trial.suggest_float('gamma', 0.9, 0.9999)
        gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)
        ent_coef = trial.suggest_float('ent_coef', 1e-8, 0.1, log=True) 
        vf_coef = trial.suggest_float('vf_coef', 0.1, 0.9)             
        n_epochs = trial.suggest_int('n_epochs', 3, 20)                
        clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
        lookback_window = trial.suggest_int('lookback_window', 10, 90) 

        # Configuration parameters
        symbol = 'BTC-USD' 
        initial_balance = 10000
        transaction_fee_percent = 0.001
        data_period = '1y' 
        data_interval = '1h'

        # Create environment
        env_lambda = lambda: CryptoTradingEnv(
            symbol=symbol,
            initial_balance=initial_balance,
            transaction_fee_percent=transaction_fee_percent,
            data_period=data_period,
            data_interval=data_interval,
            lookback_window=lookback_window 
        )
        env = DummyVecEnv([env_lambda])
        env = VecCheckNan(env, raise_exception=True) 
        
        # Create PPO model
        model = PPO(
            "MlpPolicy", 
            env, 
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs, 
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,   
            clip_range=clip_range,
            verbose=0, 
            seed=42 
        )
        
        # Train model 
        total_timesteps_train = 50000 
        model.learn(total_timesteps=total_timesteps_train, progress_bar=False) 
        
        # Evaluate model: run for one full episode
        obs = env.reset()
        done = False
        final_info = None
        actual_env = env.envs[0]
        max_episode_steps = len(actual_env.df_processed) - actual_env.lookback_window -1

        for step_num in range(max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action) 
            if dones[0]:
                final_info = infos[0]
                break
        if not final_info: 
             final_info = infos[0] if infos else {'sharpe_ratio': -float('inf')} 

        env.close()
        
        sharpe_ratio = final_info.get('sharpe_ratio', -float('inf')) 
        logger.info(f"Trial {trial.number} finished. Sharpe Ratio: {sharpe_ratio:.4f}. Params: {json.dumps(trial.params)}")
        return sharpe_ratio

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        return -float('inf')  

def main():
    # Create a study object and specify the direction is 'maximize'
    study = optuna.create_study(
        study_name='PPO_Crypto_Tuning_v2', 
        direction='maximize',
        storage='sqlite:///optuna_crypto_study_v2.db', 
        load_if_exists=True
    )
    
    # Run the optimization
    study.optimize(objective, n_trials=50) 
    
    # Print best trial details
    logger.info(f"Study completed. Number of finished trials: {len(study.trials)}")
    if study.best_trial:
        logger.info('Best trial:')
        trial = study.best_trial
        logger.info(f'  Value (Sharpe Ratio): {trial.value:.4f}')
        logger.info('  Params: ')        
        for key, value in trial.params.items():
            logger.info(f'    {key}: {value}')
        
        # You could save the best model here if needed
        # Example: best_params = trial.params
        # (recreate env, model with best_params, train longer, save model)
    else:
        logger.info("No successful trials completed.")

if __name__ == '__main__':
    main()
