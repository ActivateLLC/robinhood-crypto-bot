import json
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def load_optuna_results(database_path):
    """Load Optuna trial results from SQLite database"""
    conn = sqlite3.connect(database_path)
    df = pd.read_sql_query("SELECT * FROM trials", conn)
    conn.close()
    return df

def compare_tuning_results():
    # Load results from both tuning runs
    ppo_trials = load_optuna_results('optuna_ppo_tuning.db')
    alt_trials = load_optuna_results('optuna_alternative_tuning.db')
    
    # Visualize distribution of rewards
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    ppo_trials['value'].hist(bins=20, alpha=0.5, label='PPO')
    plt.title('PPO Trials Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    alt_trials['value'].hist(bins=20, alpha=0.5, label='SAC/TD3')
    plt.title('Alternative Algorithms Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('optuna_rewards_comparison.png')
    
    # Detailed summary
    summary = {
        'PPO': {
            'best_reward': ppo_trials['value'].max(),
            'mean_reward': ppo_trials['value'].mean(),
            'median_reward': ppo_trials['value'].median()
        },
        'Alternative': {
            'best_reward': alt_trials['value'].max(),
            'mean_reward': alt_trials['value'].mean(),
            'median_reward': alt_trials['value'].median()
        }
    }
    
    # Save summary
    with open('optuna_tuning_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Tuning results analysis complete. Check optuna_rewards_comparison.png and optuna_tuning_summary.json")

if __name__ == '__main__':
    compare_tuning_results()
