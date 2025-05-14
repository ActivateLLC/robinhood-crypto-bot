# PPO Hyperparameter Tuning Strategy Using Optuna

## 1. Objective Function

The objective function for optimizing the PPO model in the context of cryptocurrency trading will focus on maximizing the **Sharpe Ratio** or **net profit** from backtests conducted in a custom trading environment, such as `CryptoTradingEnv`. A higher Sharpe Ratio indicates better risk-adjusted returns, making it a suitable choice for reinforcing profitable trading strategies.

### Example of Objective Function:
```python
def objective(trial):
    # Extract hyperparameters from the trial
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_int("n_steps", 1024, 2048, step=1024)
    batch_size = trial.suggest_int("batch_size", 64, 256, step=32)
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 0.5)

    # Create and evaluate PPO agent
    model = PPO('MlpPolicy', CryptoTradingEnv(), 
                learning_rate=learning_rate, 
                n_steps=n_steps, 
                batch_size=batch_size, 
                n_epochs=n_epochs,
                gamma=gamma, 
                gae_lambda=gae_lambda, 
                ent_coef=ent_coef, 
                vf_coef=vf_coef,
                verbose=0)

    # Train the PPO agent
    model.learn(total_timesteps=100000)
    
    # Evaluate the model and return the Sharpe Ratio or net profit
    sharpe_ratio = evaluate_agent(model)  # Function to calculate Sharpe Ratio
    return sharpe_ratio
```

## 2. Optuna Search Space Definitions for Key PPO Hyperparameters

Here are the Optuna search space definitions for the key PPO hyperparameters as derived from the research report:

```python
import optuna

def create_study():
    study = optuna.create_study(direction='maximize')  # Aim to maximize the Sharpe Ratio
    return study

# Example parameters to include in your hyperparameter tuning
def define_search_spaces(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_int("n_steps", 1024, 2048, step=1024)
    batch_size = trial.suggest_int("batch_size", 64, 256, step=32)
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 0.5)
    return (learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, ent_coef, vf_coef)
```

## 3. High-Level Steps for Integrating Optuna into a Stable-Baselines3 Training Loop

Here’s how to integrate Optuna with a stable-baselines3 PPO training loop:

1. **Setup the Optuna Study:**
   - Create a study using `optuna.create_study()` and specify the direction as 'maximize' for the Sharpe Ratio.

2. **Define the Objective Function:**
   - Implement the `objective()` function as detailed earlier to extract and apply hyperparameters.

3. **Run the Optimization Trials:**
   - Use `study.optimize(objective, n_trials=100)` for running multiple optimization trials.

4. **Evaluate Results and Best Hyperparameters:**
   - Use methods like `study.best_trial` to evaluate the best hyperparameters after trials complete.

5. **Train the Final Model:**
   - Once the optimal hyperparameters are found, set them in the PPO model instance and train it on the entire dataset for the desired number of timesteps.

### Example Implementation Workflow:
```python
if __name__ == "__main__":
    study = create_study()
    study.optimize(objective, n_trials=100)
    
    print("Best Trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Train the final model
    final_model = PPO('MlpPolicy', CryptoTradingEnv(),
                       learning_rate=trial.params['learning_rate'],
                       n_steps=trial.params['n_steps'],
                       batch_size=trial.params['batch_size'],
                       n_epochs=trial.params['n_epochs'],
                       gamma=trial.params['gamma'],
                       gae_lambda=trial.params['gae_lambda'],
                       ent_coef=trial.params['ent_coef'],
                       vf_coef=trial.params['vf_coef'],
                       verbose=1)
    
    final_model.learn(total_timesteps=1000000)
```

By implementing this structured plan, you will effectively optimize the PPO hyperparameters for your cryptocurrency trading environment while leveraging the powerful capabilities of the Optuna library.