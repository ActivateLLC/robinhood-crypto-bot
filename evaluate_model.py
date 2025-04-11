import pandas as pd
import numpy as np
import logging
import os
from stable_baselines3 import PPO
from dotenv import load_dotenv
import pandas_ta as ta # Explicit import
from rl_environment import CryptoTradingEnv  # Add this import

# Assuming alt_crypto_data.py is in the same directory
from alt_crypto_data import AltCryptoDataProvider

# --- Constants based on previous debugging and environment ---
OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']
# Based on Memory d1708acb (8 features expected) and Memory 8c20a12f (indicators used)
# Assuming the 8 features are OHLCV + RSI + MACD_line + MACD_signal
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume', # OHLCV (5)
    'RSI_14',                                # RSI (1)
    'MACD_12_26_9',                          # MACD Line (1)
    'MACDs_12_26_9'                          # MACD Signal Line (1)
] # Total: 8 features expected by the loaded model

# --- Configuration ---
# Ensure these match the model you want to test and the environment settings
# --- START MODIFICATION: Revert model path --- #
MODEL_PATH = "trained_models/BTC_USD_PPO_DSR_250000steps.zip" # Reverted back to the DSR model
# --- END MODIFICATION --- #
SYMBOL = "BTC" # Symbol the model was trained for
EVALUATION_PERIOD_DAYS = 365 # How much total historical data to fetch (e.g., 1 year)
BACKTEST_SPLIT_RATIO = 0.75 # Use first 75% for normalization stats, rest for backtest
INITIAL_CAPITAL = 10000.0 # Starting capital for the backtest simulation
TRADE_AMOUNT_FIXED = 1000.0 # Fixed USD amount to trade per signal (simplified for backtest)
RL_LOOKBACK_WINDOW = 30 # MUST match the lookback window used during training/live bot

# --- Setup ---
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Load environment variables (might be needed for data provider keys if not yfinance)
load_dotenv()


def calculate_norm_stats(df):
    """
    Calculates mean and std dev for normalization based on the provided dataframe.
    Adds required indicators first.
    """
    df_with_indicators = df.copy()
    if not all(col in df_with_indicators.columns for col in OHLCV_COLS):
        logging.error(f"Normalization input DataFrame missing one or more OHLCV columns: {OHLCV_COLS}")
        return None

    logging.debug("Calculating indicators for normalization stats...")
    try:
        # Calculate all indicators that might be needed for the FEATURE_COLS
        df_with_indicators.ta.rsi(length=14, append=True)
        df_with_indicators.ta.macd(fast=12, slow=26, signal=9, append=True)
        # df_with_indicators.ta.bbands(length=20, std=2, append=True) # Example: calculate even if not in FEATURE_COLS
    except Exception as e:
        logging.error(f"Error calculating technical indicators during norm calc: {e}")
        return None
    logging.debug(f"Columns after adding indicators: {df_with_indicators.columns.tolist()}")

    # Select only the features the model expects
    if not all(col in df_with_indicators.columns for col in FEATURE_COLS):
        missing = [col for col in FEATURE_COLS if col not in df_with_indicators.columns]
        logging.error(f"DataFrame missing required feature columns for norm stats: {missing}")
        logging.error(f"Available columns: {df_with_indicators.columns.tolist()}")
        return None

    feature_df = df_with_indicators[FEATURE_COLS].copy()
    feature_df.dropna(inplace=True) # Drop NaNs specific to the feature set

    if feature_df.empty:
        logging.error("Normalization feature DataFrame became empty after selecting features/dropping NaNs.")
        return None

    logging.debug(f"Shape of feature_df for norm stats after dropna: {feature_df.shape}")

    means = feature_df.mean()
    stds = feature_df.std()
    # Avoid division by zero if std dev is zero for any feature
    stds[stds == 0] = 1e-8
    return {"means": means, "stds": stds}


def evaluate_model(model_path, symbol, lookback_window=30, num_episodes=10):
    """
    Comprehensive model evaluation with multiple performance metrics
    
    Args:
        model_path (str): Path to trained model
        symbol (str): Trading symbol
        lookback_window (int): Historical data window
        num_episodes (int): Number of evaluation episodes
    
    Returns:
        dict: Comprehensive performance metrics
    """
    # Fetch historical data
    data_provider = AltCryptoDataProvider()
    df = data_provider.fetch_all_historical_data(symbol, period='6mo', interval='1h')
    
    # Load trained model
    model = PPO.load(model_path)
    env = CryptoTradingEnv(df=df, symbol=symbol, lookback_window=lookback_window)
    
    # Performance tracking
    episode_returns = []
    sharpe_ratios = []
    max_drawdowns = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0
        episode_values = [env.initial_balance]
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            
            episode_return += reward
            episode_values.append(env.current_balance)
        
        # Calculate episode-level metrics
        episode_returns.append(episode_return)
        
        # Sharpe Ratio calculation
        returns_array = np.diff(episode_values) / episode_values[:-1]
        sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)  # Annualized
        sharpe_ratios.append(sharpe_ratio)
        
        # Max Drawdown calculation
        max_drawdown = max((1 - min(episode_values[i] / max(episode_values[:i+1])) * 100
                            for i in range(1, len(episode_values))))
        max_drawdowns.append(max_drawdown)
    
    # Aggregate metrics
    metrics = {
        'mean_return': np.mean(episode_returns),
        'return_std': np.std(episode_returns),
        'mean_sharpe_ratio': np.mean(sharpe_ratios),
        'mean_max_drawdown': np.mean(max_drawdowns),
        'success_rate': sum(r > 0 for r in episode_returns) / num_episodes
    }
    
    # Logging and visualization
    logging.info("Comprehensive Model Evaluation Results:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value}")
    
    return metrics


def run_backtest():
    """Runs the backtesting simulation."""
    logging.info(f"--- Starting Backtest ---")
    logging.info(f"Model: {MODEL_PATH}, Symbol: {SYMBOL}")
    logging.info(f"Evaluation Period: {EVALUATION_PERIOD_DAYS} days, Lookback Window: {RL_LOOKBACK_WINDOW} steps")

    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found: {MODEL_PATH}")
        return
    try:
        # Let Stable Baselines handle the environment loading/detection if saved with the model
        model = PPO.load(MODEL_PATH, env=None)
        logging.info(f"Successfully loaded model.")
        # Check observation space if possible (requires environment)
        # expected_obs_shape = (RL_LOOKBACK_WINDOW * len(FEATURE_COLS) + 2,) # +2 for portfolio state
        # logging.info(f"Model observation space: {model.observation_space}") # Might be None if env not loaded
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        return

    # 2. Load Data
    data_provider = AltCryptoDataProvider()
    # Fetch slightly more data initially to account for NaNs from indicators
    fetch_days = EVALUATION_PERIOD_DAYS + 60 # Add buffer
    df_hist_raw = data_provider.fetch_all_historical_data(SYMBOL, days=fetch_days, interval="1h")

    if df_hist_raw is None or df_hist_raw.empty:
        logging.error(f"Failed to fetch historical data for {SYMBOL}")
        return

    # 3. Prepare Data (Add Indicators & Handle NaNs)
    df_hist = df_hist_raw.copy()
    logging.info(f"Calculating indicators for {len(df_hist)} raw data points...")
    try:
        df_hist.ta.rsi(length=14, append=True)
        df_hist.ta.macd(fast=12, slow=26, signal=9, append=True)
        # df_hist.ta.bbands(length=20, std=2, append=True) # Add others if needed
        df_hist.dropna(inplace=True) # Drop rows with NaNs created by indicators
    except Exception as e:
        logging.error(f"Error adding indicators to historical data: {e}", exc_info=True)
        return

    if len(df_hist) < RL_LOOKBACK_WINDOW * 5: # Check if enough data remains
        logging.error(f"Insufficient data ({len(df_hist)} points) after indicator calculation for split/backtest.")
        return
    logging.info(f"Data prepared: {len(df_hist)} points after adding indicators and dropping NaNs.")

    # 4. Split Data & Calculate Normalization Stats
    split_index = int(len(df_hist) * BACKTEST_SPLIT_RATIO)
    norm_data_df = df_hist.iloc[:split_index]
    backtest_df = df_hist.iloc[split_index:]

    if len(norm_data_df) < RL_LOOKBACK_WINDOW or len(backtest_df) < RL_LOOKBACK_WINDOW:
        logging.error(f"Not enough data post-split for normalization ({len(norm_data_df)}) or backtest ({len(backtest_df)}).")
        return
    logging.info(f"Split: {len(norm_data_df)} points for norm stats, {len(backtest_df)} for backtest.")

    norm_stats = calculate_norm_stats(norm_data_df)
    if norm_stats is None:
        logging.error("Failed to calculate normalization stats from the designated data split.")
        return
    logging.info("Normalization stats calculated successfully.")
    logging.debug(f"Norm Means: {norm_stats['means']}")
    logging.debug(f"Norm Stds: {norm_stats['stds']}")

    # 5. Simulation Loop
    cash = INITIAL_CAPITAL
    holdings_qty = 0.0
    portfolio_value_history = [INITIAL_CAPITAL]
    portfolio_timestamps = [backtest_df.index[RL_LOOKBACK_WINDOW - 1]] # Start tracking value when prediction starts
    trades = []
    action_map = {0: 'buy_all', 1: 'buy_half', 2: 'hold', 3: 'sell_half', 4: 'sell_all'}
    expected_obs_shape = (RL_LOOKBACK_WINDOW * len(FEATURE_COLS) + 2,) # +2 for portfolio state

    logging.info("Starting simulation loop...")
    for i in range(RL_LOOKBACK_WINDOW, len(backtest_df)):
        current_timestamp = backtest_df.index[i]
        current_price = backtest_df['close'].iloc[i]

        if pd.isna(current_price) or current_price <= 0:
            logging.warning(f"Skipping {current_timestamp}: Invalid price ({current_price})")
            portfolio_value_history.append(portfolio_value_history[-1])
            portfolio_timestamps.append(current_timestamp)
            continue

        # --- Prepare Observation ---
        # Slice data for historical features (ends at i-1)
        observation_slice_df = backtest_df.iloc[i - RL_LOOKBACK_WINDOW : i]

        if len(observation_slice_df) != RL_LOOKBACK_WINDOW:
            logging.warning(f"Skipping {current_timestamp}: Incorrect slice length {len(observation_slice_df)}")
            portfolio_value_history.append(portfolio_value_history[-1])
            portfolio_timestamps.append(current_timestamp)
            continue

        # Select and normalize features
        features_df = observation_slice_df[FEATURE_COLS].copy()
        if features_df.isnull().values.any():
            logging.warning(f"Skipping {current_timestamp}: NaNs found in feature slice before normalization.")
            portfolio_value_history.append(portfolio_value_history[-1])
            portfolio_timestamps.append(current_timestamp)
            continue

        try:
            normalized_features = (features_df - norm_stats['means']) / norm_stats['stds']
            historical_features_flat = normalized_features.values.flatten()
        except Exception as e:
            logging.error(f"Error normalizing features at {current_timestamp}: {e}", exc_info=True)
            # Decide how to handle: skip step or stop backtest? Skipping for now.
            portfolio_value_history.append(portfolio_value_history[-1])
            portfolio_timestamps.append(current_timestamp)
            continue

        # Portfolio state features (normalized approximation)
        current_holding_value = holdings_qty * current_price
        # Use fixed trade amount as reference for normalization, similar to potential training setup
        trade_amount_safe = TRADE_AMOUNT_FIXED if TRADE_AMOUNT_FIXED > 0 else 1
        estimated_cash_for_norm = max(0, trade_amount_safe - current_holding_value) # Simplified
        norm_balance = estimated_cash_for_norm / trade_amount_safe
        norm_holdings = current_holding_value / trade_amount_safe
        portfolio_state = np.array([norm_balance, norm_holdings], dtype=np.float32)

        # Combine into final observation
        observation = np.concatenate((historical_features_flat, portfolio_state)).astype(np.float32)

        # Shape check
        if observation.shape != expected_obs_shape:
             logging.error(f"CRITICAL @ {current_timestamp}: Observation shape mismatch! Expected {expected_obs_shape}, Got {observation.shape}. Stopping.")
             break

        # --- Predict Action ---
        try:
            action_raw, _ = model.predict(observation, deterministic=True)
            action_int = action_raw.item() if isinstance(action_raw, np.ndarray) else int(action_raw)
            signal = action_map.get(action_int, 'hold')
        except Exception as e:
             logging.error(f"Error during model prediction at {current_timestamp}: {e}", exc_info=True)
             signal = 'hold' # Default to hold on prediction error

        # --- Simulate Trade ---
        qty_to_trade = 0.0
        side = None
        trade_executed = False

        if signal == 'buy_all' or signal == 'buy_half':
            side = 'buy'
            amount_to_spend = TRADE_AMOUNT_FIXED if signal == 'buy_all' else TRADE_AMOUNT_FIXED / 2
            affordable_amount = min(amount_to_spend, cash) # Can't spend more than available cash
            if affordable_amount > 0:
                qty_to_trade = affordable_amount / current_price
            else:
                qty_to_trade = 0

            if qty_to_trade > 1e-8: # Check for minimum trade size
                holdings_qty += qty_to_trade
                cash -= qty_to_trade * current_price
                trades.append({'timestamp': current_timestamp, 'side': 'buy', 'qty': qty_to_trade, 'price': current_price, 'value': qty_to_trade * current_price})
                logging.debug(f"{current_timestamp}: BOUGHT {qty_to_trade:.6f} @ {current_price:.2f}, Cash: {cash:.2f}")
                trade_executed = True

        elif signal == 'sell_all' or signal == 'sell_half':
            side = 'sell'
            if holdings_qty > 1e-8: # Check if we have anything to sell
                qty_to_trade = holdings_qty if signal == 'sell_all' else holdings_qty / 2

                cash += qty_to_trade * current_price
                holdings_qty -= qty_to_trade
                trades.append({'timestamp': current_timestamp, 'side': 'sell', 'qty': qty_to_trade, 'price': current_price, 'value': qty_to_trade * current_price})
                logging.debug(f"{current_timestamp}: SOLD {qty_to_trade:.6f} @ {current_price:.2f}, Cash: {cash:.2f}")
                trade_executed = True
            else:
                 qty_to_trade = 0 # Cannot sell if holdings are zero

        # Update portfolio value for this step
        current_total_value = cash + (holdings_qty * current_price)
        portfolio_value_history.append(current_total_value)
        portfolio_timestamps.append(current_timestamp)

    logging.info("Simulation loop finished.")

    # 6. Calculate Metrics
    if not portfolio_value_history:
        logging.error("Portfolio history is empty, cannot calculate metrics.")
        return

    final_portfolio_value = portfolio_value_history[-1]
    total_return_pct = ((final_portfolio_value / INITIAL_CAPITAL) - 1) * 100 if INITIAL_CAPITAL > 0 else 0

    # Buy & Hold Benchmark
    buy_hold_start_price = backtest_df['close'].iloc[0]
    buy_hold_end_price = backtest_df['close'].iloc[-1]
    if buy_hold_start_price > 0:
        buy_hold_qty = INITIAL_CAPITAL / buy_hold_start_price
        buy_hold_final_value = buy_hold_qty * buy_hold_end_price
        buy_hold_return_pct = ((buy_hold_final_value / INITIAL_CAPITAL) - 1) * 100
    else:
        buy_hold_final_value = INITIAL_CAPITAL
        buy_hold_return_pct = 0
        logging.warning("Could not calculate Buy & Hold benchmark due to zero start price.")

    # 7. Output Results
    print("\n--- Backtest Results ---")
    print(f"Symbol: {SYMBOL}")
    print(f"Model: {MODEL_PATH}")
    print(f"Backtest Period Start: {backtest_df.index[0]}")
    print(f"Backtest Period End:   {backtest_df.index[-1]}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
    print(f"Total Return (Strategy): {total_return_pct:.2f}%")
    print(f"Total Trades: {len(trades)}")
    print("--- Benchmark ---")
    print(f"Final Value (Buy & Hold): ${buy_hold_final_value:,.2f}")
    print(f"Total Return (Buy & Hold): {buy_hold_return_pct:.2f}%")
    print("------------------------\n")

    # Create a DataFrame for portfolio value history
    portfolio_df = pd.DataFrame({'Timestamp': portfolio_timestamps, 'PortfolioValue': portfolio_value_history})
    portfolio_df.set_index('Timestamp', inplace=True)

    # Additional metrics
    returns = (portfolio_df['PortfolioValue'].pct_change().dropna() + 1).cumprod()
    sharpe = np.sqrt(365) * returns.mean() / returns.std()
    max_drawdown = (portfolio_df['PortfolioValue'].cummax() - portfolio_df['PortfolioValue']).max()
    
    print(f"\n=== Final Metrics ===")
    print(f"Total Return: {returns.iloc[-1] - 1:.2%}")
    print(f"Annualized Sharpe: {sharpe:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

    # Optional: Save results or plot
    # Consider plotting portfolio_df vs buy&hold benchmark

    # Evaluate model with multiple metrics and robustness checks
    model_metrics = evaluate_model(MODEL_PATH, SYMBOL, RL_LOOKBACK_WINDOW, num_episodes=10)
    print("\n--- Model Evaluation Metrics ---")
    for metric, value in model_metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    run_backtest()
