#!/usr/bin/env python3
"""
Web Server for Robinhood Crypto Bot Dashboard

This module provides a web interface for the Robinhood cryptocurrency trading bot,
displaying real-time trading data, portfolio information, and strategy performance.
"""

import os
import time
import logging
import json
import threading
import robin_stocks.robinhood as rh
from flask import Flask, jsonify, render_template, request
from alt_crypto_data import AltCryptoDataProvider
from dotenv import load_dotenv
import datetime
import subprocess

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("web_server")

# Initialize alternative data provider (if needed by server itself, otherwise remove)
# alt_data_provider = AltCryptoDataProvider()

# Initialize Flask app
app = Flask(__name__)

# Global variables to store bot state (related to subprocess control)
global_bot_process = None
script_dir = os.path.dirname(os.path.abspath(__file__))
crypto_bot_script_path = os.path.join(script_dir, 'crypto_bot.py')

def get_bot_status():
    """Checks the status of the global bot process."""
    global global_bot_process
    if global_bot_process is None:
        return "stopped"
    elif global_bot_process.poll() is None:
        # Process is still running if poll() returns None
        return "running"
    else:
        # Process has finished (or crashed)
        return_code = global_bot_process.returncode
        logger.info(f"Bot process finished with return code: {return_code}")
        global_bot_process = None # Reset the global variable
        return f"finished (code: {return_code})"

@app.route('/')
def dashboard():
    """Render the main dashboard page"""
    # Pass the bot status obtained from the subprocess check
    data = {
        "title": "CryptoNeon Dashboard",
        "bot_status": get_bot_status() 
    }
    return render_template('dashboard.html', **data)

@app.route('/portfolio')
def portfolio():
    """Render the portfolio page"""
    return render_template('portfolio.html')

@app.route('/trades')
def trades():
    """Render the trade history page"""
    return render_template('trades.html')

@app.route('/settings')
def settings():
    """Render the settings page"""
    return render_template('settings.html')

# --- API Endpoints for Data (These need redesign to fetch from bot process/logs) ---

@app.route('/api/status') # Example: Should get status from get_bot_status()
def api_status():
    # Placeholder - needs implementation to read from actual bot process
    status = get_bot_status()
    return jsonify({
        "bot_status": status,
        "last_check_time": None, # Placeholder
        "pid": global_bot_process.pid if global_bot_process else None
    })

@app.route('/api/portfolio')
def get_portfolio():
    """Get the current portfolio holdings"""
    # Placeholder - needs implementation to read from actual bot process
    logger.warning("/api/portfolio endpoint needs redesign to read from bot process.")
    return jsonify({"holdings": [], "total_value": 0, "message": "Endpoint needs redesign"})

@app.route('/api/crypto/prices')
def get_crypto_prices():
    """Get current prices for all cryptocurrencies"""
    # Placeholder - needs implementation
    logger.warning("/api/crypto/prices endpoint needs redesign.")
    return jsonify({"message": "Endpoint needs redesign"})

@app.route('/api/trading/signals')
def get_trading_signals():
    """Get recent trading signals"""
    # Placeholder - needs implementation
    logger.warning("/api/trading/signals endpoint needs redesign.")
    return jsonify([])

@app.route('/api/portfolio/history')
def get_portfolio_history():
    """Get portfolio value history"""
    # Placeholder - needs implementation
    logger.warning("/api/portfolio/history endpoint needs redesign.")
    return jsonify([])

@app.route('/api/strategy/performance')
def get_strategy_performance():
    """Get performance metrics for different strategies"""
    # Placeholder - needs implementation
    logger.warning("/api/strategy/performance endpoint needs redesign.")
    # Return placeholder data for now
    return jsonify({
        "strategies": [
            {"name": "MACD", "win_rate": 0, "profit_factor": 0, "avg_profit": 0, "max_drawdown": 0, "sharpe_ratio": 0},
            {"name": "RSI", "win_rate": 0, "profit_factor": 0, "avg_profit": 0, "max_drawdown": 0, "sharpe_ratio": 0},
            {"name": "Bollinger", "win_rate": 0, "profit_factor": 0, "avg_profit": 0, "max_drawdown": 0, "sharpe_ratio": 0},
            {"name": "Multi", "win_rate": 0, "profit_factor": 0, "avg_profit": 0, "max_drawdown": 0, "sharpe_ratio": 0}
        ],
        "message": "Endpoint needs redesign"
    })

# --- Bot Control (Subprocess Implementation) ---

@app.route('/start_bot', methods=['POST'])
def start_bot():
    """Starts the crypto_bot.py script as a background process."""
    global global_bot_process
    
    status = get_bot_status()
    if status == "running":
        logger.warning("Attempted to start bot, but it is already running.")
        return jsonify({"status": "error", "message": "Bot is already running."})

    try:
        logger.info(f"Starting crypto bot script: {crypto_bot_script_path}")
        # Use subprocess.Popen for non-blocking execution
        # Redirect stdout and stderr to PIPE if you want to capture them, or to DEVNULL to discard
        # Run from the script's directory to ensure relative paths in the bot work
        global_bot_process = subprocess.Popen(
            ['python', crypto_bot_script_path],
            cwd=script_dir,
            stdout=subprocess.PIPE, # Capture stdout
            stderr=subprocess.PIPE, # Capture stderr
            text=True # Decode stdout/stderr as text
        )
        logger.info(f"Bot process started with PID: {global_bot_process.pid}")
        # Give it a moment to potentially fail immediately
        time.sleep(1)
        if global_bot_process.poll() is not None:
             # It already finished/crashed
             stdout, stderr = global_bot_process.communicate()
             error_msg = f"Bot process failed immediately. Code: {global_bot_process.returncode}. Stderr: {stderr.strip()}"
             logger.error(error_msg)
             global_bot_process = None
             return jsonify({"status": "error", "message": error_msg})
             
        return jsonify({"status": "success", "message": "Bot started.", "pid": global_bot_process.pid})

    except Exception as e:
        logger.error(f"Failed to start bot process: {e}", exc_info=True)
        global_bot_process = None # Ensure it's reset on error
        return jsonify({"status": "error", "message": f"Failed to start bot: {e}"}), 500

@app.route('/stop_bot', methods=['POST'])
def stop_bot():
    """Stops the running crypto bot process."""
    global global_bot_process
    
    status = get_bot_status()
    if status != "running" or global_bot_process is None:
        logger.warning("Attempted to stop bot, but it is not running.")
        return jsonify({"status": "error", "message": "Bot is not running."})
        
    try:
        pid = global_bot_process.pid
        logger.info(f"Attempting to terminate bot process with PID: {pid}")
        global_bot_process.terminate() # Send SIGTERM
        
        # Wait a bit for graceful termination
        try:
            stdout, stderr = global_bot_process.communicate(timeout=5) # Wait up to 5 seconds
            logger.info(f"Bot process {pid} terminated gracefully. Return code: {global_bot_process.returncode}")
        except subprocess.TimeoutExpired:
            logger.warning(f"Bot process {pid} did not terminate gracefully after 5s. Sending SIGKILL.")
            global_bot_process.kill() # Force kill
            stdout, stderr = global_bot_process.communicate()
            logger.info(f"Bot process {pid} killed. Return code: {global_bot_process.returncode}")

        global_bot_process = None # Reset global variable
        return jsonify({"status": "success", "message": f"Bot process {pid} stopped."})

    except Exception as e:
        logger.error(f"Failed to stop bot process: {e}", exc_info=True)
        # We don't know the state here, maybe it's still running?
        # Might need more robust process management in a real app.
        return jsonify({"status": "error", "message": f"Failed to stop bot: {e}"}), 500

@app.route('/refresh_bot', methods=['POST']) # Changed from GET to POST to match JS
def refresh_bot():
    """Returns the current status of the bot."""
    status = get_bot_status()
    pid = global_bot_process.pid if global_bot_process else None
    logger.debug(f"Refreshing bot status: {status}")
    return jsonify({"status": "success", "bot_status": status, "pid": pid})

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=9091, debug=True)
