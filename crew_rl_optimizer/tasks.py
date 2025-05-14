from crewai import Task

class PPOHyperparameterTasks:
    def research_ppo_hyperparameters(self, agent):
        return Task(
            description=(
                'Investigate the Proximal Policy Optimization (PPO) algorithm. Identify its critical '
                'hyperparameters for tuning in the context of cryptocurrency trading. For each '
                'hyperparameter (e.g., learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, '
                'ent_coef, vf_coef), describe its role, typical ranges for financial time series, and any '
                'known sensitivities or interdependencies. Focus on practical insights for a PPO model like '
                'the one used in stable-baselines3, applied to BTC-USD trading.'
            ),
            expected_output=(
                'A structured markdown report detailing each key PPO hyperparameter, its function, '
                'impact on training, common ranges for financial/crypto applications, and any specific '
                'considerations for stable-baselines3 implementations.'
            ),
            agent=agent,
            async_execution=False
        )

    def design_optuna_strategy(self, agent, context_task=None):
        # context_task is the research_ppo_hyperparameters task, its output will be available in context
        return Task(
            description=(
                'Based on the PPO hyperparameter research report, design a hyperparameter optimization '
                'strategy using the Optuna library. Define the conceptual objective function for optimizing '
                'a PPO model for crypto trading (e.g., maximizing Sharpe ratio or net profit from a backtest, '
                'considering the CryptoTradingEnv environment). For each key PPO hyperparameter identified in the '
                'research, specify its Optuna search space (e.g., trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)). '
                'Outline the steps needed to integrate this Optuna study with an RL training script that uses '
                'stable-baselines3 and a custom environment similar to CryptoTradingEnv.'
            ),
            expected_output=(
                'A clear, actionable markdown plan for implementing PPO hyperparameter tuning with Optuna. '
                'This plan should include: \n'
                '1. A brief on the chosen objective function. \n'
                '2. Example Optuna search space definitions for key PPO hyperparameters. \n'
                '3. High-level steps for integrating Optuna into a stable-baselines3 training loop, including where to call `trial.suggest_*` methods and how to run multiple trials.'
            ),
            agent=agent,
            context=[context_task] if context_task else [], # Pass the research task as context
            async_execution=False
        )

# Example instantiation (optional)
# if __name__ == '__main__':
#     from agents import PPOExpertResearcherAgent, RLTuningStrategistAgent
#     research_agent = PPOExpertResearcherAgent()
#     strategy_agent = RLTuningStrategistAgent()
#     tasks = PPOHyperparameterTasks()
#     research_task = tasks.research_ppo_hyperparameters(research_agent)
#     strategy_task = tasks.design_optuna_strategy(strategy_agent, research_task)
#     print("Tasks created successfully.")
