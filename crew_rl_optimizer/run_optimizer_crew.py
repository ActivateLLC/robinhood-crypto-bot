import os
from crewai import Crew, Process
from dotenv import load_dotenv

# Load environment variables from .env file in the parent directory
# This assumes your .env file is in the root of robinhood-crypto-bot
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Now import agents and tasks
from agents import PPOExpertResearcherAgent, RLTuningStrategistAgent
from tasks import PPOHyperparameterTasks

def main():
    print("Initializing RL Hyperparameter Optimization Crew...")

    # Check if SERPER_API_KEY is loaded (optional, for user feedback)
    if not os.getenv('SERPER_API_KEY'):
        print("WARNING: SERPER_API_KEY not found in environment. Web search capabilities may be limited.")
        print("Please ensure it's set in your .env file at the root of the project.")

    # Instantiate Agents
    research_agent = PPOExpertResearcherAgent()
    strategy_agent = RLTuningStrategistAgent()
    print("Agents created.")

    # Instantiate Tasks
    tasks = PPOHyperparameterTasks()
    research_task = tasks.research_ppo_hyperparameters(research_agent)
    strategy_task = tasks.design_optuna_strategy(strategy_agent, context_task=research_task) # Pass research_task as context
    print("Tasks created.")

    # Form the Crew
    optimizer_crew = Crew(
        agents=[research_agent, strategy_agent],
        tasks=[research_task, strategy_task],
        process=Process.sequential,  # Tasks will be executed one after the other
        verbose=True  # 0 for no output, 1 for brief, 2 for detailed output
    )
    print("Crew formed.")

    # Kick off the work
    print("\nStarting Crew execution...")
    try:
        result = optimizer_crew.kickoff()

        print("\n--------------------------------------------------")
        print("Crew Execution Finished.")
        print("--------------------------------------------------")
        print("Final Result:")
        print(result) # This will be the output of the last task (strategy_task)

        # You might want to save the result to a file
        output_filename = "ppo_hyperparameter_optimization_report.md"
        with open(output_filename, "w") as f:
            f.write(str(result))
        print(f"\nReport saved to {output_filename}")

    except Exception as e:
        print(f"\nAn error occurred during Crew execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
