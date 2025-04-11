#!/bin/bash

# Ensure virtual environment is activated
source .venv/bin/activate

# Set up logging directories
mkdir -p logs

# Redirect stdout and stderr to log files
python optuna_ppo_tuning.py > logs/ppo_tuning.log 2>&1 &
python optuna_alternative_tuning.py > logs/alternative_tuning.log 2>&1 &

# Wait for both processes to complete
wait

echo "Optuna tuning completed. Check logs for results."
