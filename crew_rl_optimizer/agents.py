from crewai import Agent
from crewai_tools import SerperDevTool # Using Serper as an example for web search

# Initialize the search tool (SerperDevTool requires SERPER_API_KEY environment variable)
# You might need to install it: pip install crewai-tools
# Ensure SERPER_API_KEY is set in your .env or environment
search_tool = SerperDevTool()

class PPOExpertResearcherAgent(Agent):
    def __init__(self):
        super().__init__(
            role='PPO Hyperparameter Research Expert',
            goal='To research and identify key PPO hyperparameters, their impact on training (especially for time-series/financial data), and common effective ranges.',
            backstory=(
                'An expert in Reinforcement Learning with a deep understanding of the PPO algorithm '
                'and its nuances in financial applications. You are skilled at synthesizing information '
                'from various sources into concise and actionable insights.'
            ),
            tools=[search_tool],
            verbose=True,
            allow_delegation=False
        )

class RLTuningStrategistAgent(Agent):
    def __init__(self):
        super().__init__(
            role='RL Hyperparameter Tuning Strategist',
            goal='To take researched PPO hyperparameters and formulate a practical hyperparameter tuning strategy, ideally using the Optuna library.',
            backstory=(
                'A specialist in machine learning model optimization, proficient in designing and implementing '
                'hyperparameter search strategies. You excel at translating theoretical research into '
                'practical, implementable plans for model improvement.'
            ),
            tools=[search_tool], # May also use a file reading tool if the report is passed as a file
            verbose=True,
            allow_delegation=False # Or True if it needs to delegate sub-tasks
        )

# Example instantiation (optional, for testing or direct use)
# if __name__ == '__main__':
#     researcher = PPOExpertResearcherAgent()
#     strategist = RLTuningStrategistAgent()
#     print("Agents created successfully.")
