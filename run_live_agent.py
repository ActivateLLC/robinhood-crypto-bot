import os
import time
import logging
import json
from stable_baselines3 import PPO
from dotenv import load_dotenv
from crew_agents.src.crypto_trading_env import CryptoTradingEnvironment
from alt_crypto_data import AltCryptoDataProvider
import config

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]) # Add FileHandler if needed
logger = logging.getLogger(__name__)

MODEL_PATH = "models/best_model.zip"  # Path to the trained model
TRADING_SYMBOL = os.environ.get('TRADING_SYMBOL', 'BTC-USD') # Get symbol from env or default
INITIAL_CAPITAL = float(os.environ.get('INITIAL_CAPITAL', 1000.0)) # Ensure float
LOOKBACK_WINDOW = int(os.environ.get('LOOKBACK_WINDOW', 60)) # Ensure int
LIVE_TRADING_INTERVAL_SECONDS = 60 * 5 # How often to check and trade (e.g., every 5 minutes)

# Define the path for the state file
script_dir = os.path.dirname(os.path.abspath(__file__))
STATE_FILE_PATH = os.path.join(script_dir, 'live_agent_state.json')

def save_state(state_data):
    """Saves the current agent state to a JSON file."""
    try:
        with open(STATE_FILE_PATH, 'w') as f:
            json.dump(state_data, f, indent=4)
        # logger.debug(f"Agent state saved to {STATE_FILE_PATH}") # Can enable for verbose logging
    except Exception as e:
        logger.error(f"Error saving agent state to {STATE_FILE_PATH}: {e}", exc_info=True)

# --- Main Execution ---
def run_live_trading():
    logger.info("--- Starting Live Trading Agent ---")

    # 1. Check Configuration for Live Trading
    if not config.ENABLE_TRADING:
        logger.error("Live trading is disabled in config.py (ENABLE_TRADING=False). Exiting.")
        return
    if config.ENABLE_TRADING:
        # Use the correct variable names loaded from config.py
        if not config.ROBINHOOD_API_KEY or not config.ROBINHOOD_BASE64_PRIVATE_KEY:
            logger.error("Trading is enabled, but API_KEY or BASE64_PRIVATE_KEY is missing in environment variables (.env).")
            logger.error("Please set ROBINHOOD_API_KEY and ROBINHOOD_BASE64_PRIVATE_KEY.")
            return
    
    logger.info(f"Live Trading ENABLED for symbol: {TRADING_SYMBOL}")
    logger.info(f"Model Path: {MODEL_PATH}")
    logger.info(f"Trading Interval: {LIVE_TRADING_INTERVAL_SECONDS} seconds")

    # 2. Initialize Environment
    try:
        # Initialize the data provider (Corrected: remove symbol argument)
        data_provider = AltCryptoDataProvider()
        logger.info("AltCryptoDataProvider initialized.")

        env = CryptoTradingEnvironment(
            data_provider=data_provider,
            symbol=TRADING_SYMBOL,
            initial_capital=INITIAL_CAPITAL,
            live_trading=True, # IMPORTANT: Set to True
            lookback_window=LOOKBACK_WINDOW,
            log_experience=True, # Enable experience logging for potential retraining
            broker_type='robinhood' # Explicitly set broker type
        )
        logger.info("CryptoTradingEnvironment initialized in live mode.")
    except Exception as e:
        logger.exception(f"Error initializing CryptoTradingEnvironment: {e}")
        return

    # 3. Load Trained Agent
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}. Exiting.")
        return
    
    try:
        # Load the model - MAKE SURE PPO matches the algorithm used for best_model.zip
        model = PPO.load(MODEL_PATH, env=env) 
        logger.info(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        logger.exception(f"Error loading model from {MODEL_PATH}: {e}")
        return

    # 4. Run Live Trading Loop
    logger.info("Starting live trading loop...")
    obs, info = env.reset() 
    
    try:
        while True:
            # Get action from the agent
            action, _states = model.predict(obs, deterministic=True) # Use deterministic actions for live trading
            
            # Execute action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- State Saving --- 
            current_price = 0.0
            portfolio_value = 0.0
            try:
                # Get current price to calculate portfolio value
                # Use the environment's broker instance
                current_price = float(env.broker.get_current_price(TRADING_SYMBOL))
                portfolio_value = float(env.current_capital + (env.crypto_holdings * current_price))
            except Exception as price_err:
                logger.error(f"Could not fetch current price for state saving: {price_err}")
                # Use last known portfolio value from history if available
                if hasattr(env, 'portfolio_history') and env.portfolio_history:
                    portfolio_value = env.portfolio_history[-1]
                else:
                     portfolio_value = env.initial_capital # Fallback

            agent_state = {
                'timestamp': time.time(),
                'step': env.current_step,
                'last_action': int(action), # Ensure action is JSON serializable
                'current_price': current_price,
                'portfolio_value': portfolio_value,
                'capital': float(env.current_capital),
                'holdings': float(env.crypto_holdings),
                'info': info, # Include info dict from env.step()
                'symbol': TRADING_SYMBOL
            }
            save_state(agent_state)
            # --------------------

            logger.info(f"Step completed. Action: {action}, Info: {info}")
            
            # In live trading, the environment likely won't terminate or truncate itself
            # Handle external stop signals if needed (e.g., KeyboardInterrupt)

            # Wait for the next interval
            logger.debug(f"Sleeping for {LIVE_TRADING_INTERVAL_SECONDS} seconds...")
            time.sleep(LIVE_TRADING_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping live trading agent.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the live trading loop: {e}")
    finally:
        logger.info("Closing environment (if applicable)...")
        # env.close() # Implement close method if needed for cleanup
        logger.info("--- Live Trading Agent Stopped ---")


if __name__ == "__main__":
    run_live_trading()
