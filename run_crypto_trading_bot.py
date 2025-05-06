#!/usr/bin/env python3
import os
import sys
import logging
import signal
import time
import json
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from crew_agents.src.multi_agent_orchestrator import MultiAgentOrchestrator
from crew_agents.src.alt_crypto_data import AltCryptoDataProvider
from crew_agents.src.crypto_trading_env import CryptoTradingEnvironment as CryptoTradingEnv

# Create logs directory if it doesn't exist
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Console output
        logging.FileHandler(os.path.join(log_dir, 'crypto_trading_bot.log'), mode='w')  # Use final log file path
    ]
)

# Set specific loggers to different levels
logging.getLogger('CryptoTradingEnvironment').setLevel(logging.DEBUG)
logging.getLogger('stable_baselines3').setLevel(logging.INFO)
logging.getLogger('gymnasium').setLevel(logging.WARNING)

logger = logging.getLogger(__name__) # Get root logger
logger.info("--- Crypto Trading Bot Script Started ---")

# Global exception handler
def global_exception_handler(exc_type, exc_value, exc_traceback):
    logging.error(
        "Uncaught exception", 
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = global_exception_handler

def signal_handler(signum, frame):
    """
    Handle system signals for graceful shutdown
    """
    print(f"\n🛑 Received signal {signum}. Initiating graceful shutdown...")
    sys.exit(0)

def backtest_trading_model(symbol='BTC-USD', years=3):
    """
    Comprehensive backtesting of trading model
    
    Args:
        symbol (str): Cryptocurrency trading pair
        years (int): Number of historical years to backtest
    
    Returns:
        dict: Backtesting performance metrics
    """
    # Initialize data provider
    data_provider = AltCryptoDataProvider()
    
    # Fetch historical data
    historical_data = data_provider.fetch_crypto_data(
        symbol=symbol, 
        days=years*365, 
        interval='1h'
    )
    
    # Create trading environment
    env = DummyVecEnv([lambda: CryptoTradingEnv(
        df=historical_data, 
        initial_capital=100000, 
        trading_fee=0.001
    )])
    
    # Initialize and train PPO model
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0008259, 
        n_steps=4096,
        batch_size=128,
        gamma=0.92,
        gae_lambda=0.8353,
        ent_coef=0.00028,
        clip_range=0.3794
    )
    
    # Train the model
    model.learn(total_timesteps=int(1e5))
    
    # Evaluate model performance
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    
    # Detailed performance analysis
    obs = env.reset()
    done = False
    total_reward = 0
    portfolio_values = []
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        portfolio_values.append(info[0]['portfolio_value'])
    
    # Calculate performance metrics
    performance_metrics = {
        'symbol': symbol,
        'initial_capital': 100000,
        'final_portfolio_value': portfolio_values[-1],
        'total_return_percentage': ((portfolio_values[-1] - 100000) / 100000) * 100,
        'mean_reward': mean_reward,
        'reward_std': std_reward,
        'max_drawdown': np.min(portfolio_values) / 100000 - 1
    }
    
    # Log and save results
    log_dir = os.path.join(project_root, 'logs', 'backtesting')
    os.makedirs(log_dir, exist_ok=True)
    
    results_file = os.path.join(log_dir, f'backtest_results_{symbol}_{int(time.time())}.json')
    with open(results_file, 'w') as f:
        json.dump(performance_metrics, f, indent=2)
    
    return performance_metrics

def main():
    """
    Main entry point for the Crypto Trading Bot Backtesting
    """
    # Symbols to backtest
    symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD']
    
    # Backtest each symbol
    backtest_results = {}
    for symbol in symbols:
        logger.info(f"🔍 Backtesting {symbol}")
        try:
            result = backtest_trading_model(symbol)
            backtest_results[symbol] = result
            logger.info(f"✅ Backtest completed for {symbol}")
            logger.info(f"Performance: {result['total_return_percentage']:.2f}%")
        except Exception as e:
            logger.error(f"❌ Backtest failed for {symbol}: {e}")
    
    # Summarize results
    logger.info("\n🏁 Backtesting Summary:")
    for symbol, result in backtest_results.items():
        logger.info(f"{symbol}: Total Return = {result['total_return_percentage']:.2f}%")

if __name__ == "__main__":
    main()
