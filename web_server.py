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
import subprocess
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

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

# Initialize Flask app
app = Flask(__name__)

# Global variables to store bot state (related to subprocess control)
global_bot_process = None
script_dir = os.path.dirname(os.path.abspath(__file__))
# Point to the RL agent script instead of the old TA bot
live_agent_script_path = os.path.join(script_dir, 'run_live_agent.py')
# Define path to the state file written by the live agent
STATE_FILE_PATH = os.path.join(script_dir, 'live_agent_state.json') 

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

def read_agent_state():
    """Reads the latest state from the agent's state file."""
    default_state = {
        'timestamp': 0,
        'step': 0,
        'last_action': None,
        'current_price': 0,
        'portfolio_value': 0,
        'capital': 0,
        'holdings': 0,
        'info': {'status': 'file_not_found_or_invalid'},
        'symbol': None
    }
    if not os.path.exists(STATE_FILE_PATH):
        logger.warning(f"State file not found: {STATE_FILE_PATH}")
        return default_state
    try:
        with open(STATE_FILE_PATH, 'r') as f:
            state = json.load(f)
        return state
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from state file: {STATE_FILE_PATH}")
        return default_state
    except Exception as e:
        logger.error(f"Error reading state file {STATE_FILE_PATH}: {e}", exc_info=True)
        return default_state

@app.route('/')
def dashboard():
    """Render the main dashboard page"""
    process_status = get_bot_status()
    agent_state = read_agent_state()
    data = {
        "title": "Crypto RL Dashboard",
        "bot_status": process_status, 
        "agent_state": agent_state # Pass the full state to the template
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
    """Returns the current process status and latest agent state timestamp."""
    process_status = get_bot_status()
    agent_state = read_agent_state()
    pid = global_bot_process.pid if global_bot_process and process_status == 'running' else None
    
    return jsonify({
        "process_status": process_status,
        "pid": pid,
        "last_state_update_time": agent_state.get('timestamp', 0),
        "agent_step": agent_state.get('step', 0),
        "agent_info": agent_state.get('info', {})
    })

@app.route('/api/portfolio')
def get_portfolio():
    """Get the current portfolio holdings from the state file."""
    agent_state = read_agent_state()
    
    # Check if the state file was read successfully (basic check)
    if agent_state.get('info', {}).get('status') == 'file_not_found_or_invalid':
         logger.warning("/api/portfolio called but state file is missing or invalid.")
         return jsonify({"error": "Agent state file not available"}), 404
         
    portfolio_data = {
        "symbol": agent_state.get('symbol'),
        "holdings": agent_state.get('holdings', 0),
        "capital": agent_state.get('capital', 0),
        "portfolio_value": agent_state.get('portfolio_value', 0),
        "current_price": agent_state.get('current_price', 0),
        "last_updated": agent_state.get('timestamp', 0)
    }
    return jsonify(portfolio_data)

@app.route('/api/crypto/prices')
def get_crypto_prices():
    """Get current price from the state file."""
    agent_state = read_agent_state()
    if agent_state.get('info', {}).get('status') == 'file_not_found_or_invalid':
         return jsonify({"error": "Agent state file not available"}), 404
         
    return jsonify({
        agent_state.get('symbol', 'UNKNOWN'): agent_state.get('current_price', 0),
        "last_updated": agent_state.get('timestamp', 0)
        })

@app.route('/api/trading/signals')
def get_trading_signals():
    """Get recent trading signals (last action) from state file."""
    agent_state = read_agent_state()
    if agent_state.get('info', {}).get('status') == 'file_not_found_or_invalid':
         return jsonify({"error": "Agent state file not available"}), 404

    # Return the last action as a simple signal representation
    signals = [{
        'timestamp': agent_state.get('timestamp', 0),
        'symbol': agent_state.get('symbol', 'UNKNOWN'),
        'action': agent_state.get('last_action'), # 0: Hold, 1-3: Buy, 4-6: Sell
        'step': agent_state.get('step', 0)
    }]
    return jsonify(signals)

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
    """Starts the run_live_agent.py script as a background process."""
    global global_bot_process
    
    status = get_bot_status()
    if status == "running":
        logger.warning("Attempted to start live agent, but it is already running.")
        return jsonify({"status": "error", "message": "Live agent is already running."})

    try:
        logger.info(f"Starting live agent script: {live_agent_script_path}")
        # Use subprocess.Popen for non-blocking execution
        # Redirect stdout and stderr to PIPE if you want to capture them, or to DEVNULL to discard
        # Run from the script's directory to ensure relative paths work
        global_bot_process = subprocess.Popen(
            ['python', live_agent_script_path],
            cwd=script_dir,
            stdout=subprocess.PIPE, # Capture stdout
            stderr=subprocess.PIPE, # Capture stderr
            text=True # Decode stdout/stderr as text
        )
        logger.info(f"Live agent process started with PID: {global_bot_process.pid}")
        # Give it a moment to potentially fail immediately
        time.sleep(1)
        if global_bot_process.poll() is not None:
             # It already finished/crashed
             stdout, stderr = global_bot_process.communicate()
             error_msg = f"Live agent process failed immediately. Code: {global_bot_process.returncode}. Stderr: {stderr.strip()}"
             logger.error(error_msg)
             global_bot_process = None
             return jsonify({"status": "error", "message": error_msg})
             
        return jsonify({"status": "success", "message": "Live agent started.", "pid": global_bot_process.pid})

    except Exception as e:
        logger.error(f"Failed to start live agent process: {e}", exc_info=True)
        global_bot_process = None # Ensure it's reset on error
        return jsonify({"status": "error", "message": f"Failed to start live agent: {e}"}), 500

@app.route('/stop_bot', methods=['POST'])
def stop_bot():
    """Stops the running live agent process."""
    global global_bot_process
    
    status = get_bot_status()
    if status != "running" or global_bot_process is None:
        logger.warning("Attempted to stop live agent, but it is not running.")
        return jsonify({"status": "error", "message": "Live agent is not running."})
        
    try:
        pid = global_bot_process.pid
        logger.info(f"Attempting to terminate live agent process with PID: {pid}")
        global_bot_process.terminate() # Send SIGTERM
        
        # Wait a bit for graceful termination
        try:
            stdout, stderr = global_bot_process.communicate(timeout=5) # Wait up to 5 seconds
            logger.info(f"Live agent process {pid} terminated gracefully. Return code: {global_bot_process.returncode}")
        except subprocess.TimeoutExpired:
            logger.warning(f"Live agent process {pid} did not terminate gracefully after 5s. Sending SIGKILL.")
            global_bot_process.kill() # Force kill
            stdout, stderr = global_bot_process.communicate()
            logger.info(f"Live agent process {pid} killed. Return code: {global_bot_process.returncode}")

        global_bot_process = None # Reset global variable
        return jsonify({"status": "success", "message": f"Live agent process {pid} stopped."})

    except Exception as e:
        logger.error(f"Failed to stop live agent process: {e}", exc_info=True)
        # We don't know the state here, maybe it's still running?
        # Might need more robust process management in a real app.
        return jsonify({"status": "error", "message": f"Failed to stop live agent: {e}"}), 500

@app.route('/refresh_bot', methods=['POST']) # Changed from GET to POST to match JS
def refresh_bot():
    """Returns the current status of the live agent."""
    status = get_bot_status()
    pid = global_bot_process.pid if global_bot_process else None
    logger.debug(f"Refreshing live agent status: {status}")
    return jsonify({"status": "success", "bot_status": status, "pid": pid})

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=9091, debug=True)
