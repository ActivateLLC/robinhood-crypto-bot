#!/bin/bash

# Ensure script runs from project root
cd "$(dirname "$0")"

# Set Python environment
export PYTHONPATH=.

# Logging configuration
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Optimization mode selection
MODE="${1:-default}"

# Function to run default optimization
run_default_optimization() {
    echo "Running Default Optimization"
    python3 run_optimization.py
}

# Function to run continuous optimization
run_continuous_optimization() {
    echo "Starting Continuous Optimization Agent"
    python3 -m crew_agents.src.continuous_optimization_agent
}

# Function to run backtesting
run_backtesting() {
    echo "Running Comprehensive Backtesting"
    python3 -m crew_agents.src.backtest_optimization
}

# Main execution
case "$MODE" in
    "continuous")
        run_continuous_optimization
        ;;
    "backtest")
        run_backtesting
        ;;
    *)
        run_default_optimization
        ;;
esac
