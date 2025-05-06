import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

from crewai import Agent, Task, Crew, Process
from crewai_tools import FileReadTool
from dotenv import load_dotenv

# Adjust import based on actual project structure if needed
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config

load_dotenv() # Load environment variables like OPENAI_API_KEY

# Setup basic logging for the agent script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LOG_FILE_PATH = "../logs/live_experience.log" # Assuming the script is run from the agents directory

class ExperienceProcessorAgent:
    """Agent responsible for reading, processing, and potentially acting on logged trading experiences."""

    def __init__(self):
        self.config = load_config() # Load bot configuration
        self.experience_log_file = os.path.abspath(self.config.EXPERIENCE_LOG_FILE)
        logger.info(f"Experience Processor initialized. Monitoring log file: {self.experience_log_file}")

        # --- Define Tools --- #
        # Tool to read the entire log file. We'll need logic to find *new* entries.
        self.file_reader_tool = FileReadTool(file_path=self.experience_log_file)
        
        # --- Define Agents --- #
        self.log_reader_agent = Agent(
            role='Experience Log Analyst',
            goal=f'Read and extract all trading experiences from {os.path.basename(self.experience_log_file)} as structured JSON objects.',
            backstory=(
                "An analytical agent specialized in parsing structured log data, particularly trading records. "
                "It accurately extracts all JSON entries from the log file for further analysis."
            ),
            tools=[self.file_reader_tool],
            verbose=True,
            allow_delegation=False
            # Configure LLM if not using default (e.g., set OPENAI_API_KEY in .env)
        )

        # --- TODO: Add more agents as needed (e.g., ModelUpdaterAgent) --- #

    def process_all_experiences(self) -> List[Dict[str, Any]]:
        """Reads and processes ALL entries from the experience log."""
        logger.info("Processing all experiences from log file...")
        
        # --- Define Task --- #
        read_log_task = Task(
            description=(
                f"Read the entire content of the experience log file located at '{self.experience_log_file}'. "
                f"Each line in the file is a separate JSON object representing a trading experience. "
                f"Parse each line as a JSON object and return a list containing all these JSON objects."
            ),
            expected_output=(
                "A Python list containing all the parsed JSON objects (dictionaries) from the log file. "
                "If the file is empty or cannot be parsed, return an empty list."
            ),
            agent=self.log_reader_agent,
            tools=[self.file_reader_tool] # Explicitly pass tool to task
        )

        # --- Create and Run Crew --- #
        experience_crew = Crew(
            agents=[self.log_reader_agent],
            tasks=[read_log_task],
            verbose=True # Set verbosity level (True/False)
        )

        # Execute the task
        try:
            result = experience_crew.kickoff() # result is likely CrewOutput
            logger.info(f"Crew kickoff finished.")

            parsed_experiences = []
            processed_output_str = None

            # --- Extract the relevant output string from CrewOutput --- 
            if hasattr(result, 'tasks_output') and result.tasks_output:
                # Get the output of the first (and only) task
                task_output = result.tasks_output[0]
                # Access the raw string output of the task first
                if hasattr(task_output, 'raw_output') and isinstance(task_output.raw_output, str):
                    processed_output_str = task_output.raw_output
                elif isinstance(task_output, str): # Check if the task output itself is the string
                     processed_output_str = task_output 
                else:
                     # If raw_output isn't a string, try converting the whole task_output object
                     logger.info("TaskOutput.raw_output not usable. Trying str(task_output) as fallback.")
                     processed_output_str = str(task_output) 
                     # Add a check here? Maybe log if str(task_output) looks like a dict repr?
                     if not processed_output_str or '__dict__' in processed_output_str: 
                         logger.warning(f"str(task_output) didn't seem helpful. Trying result.raw / str(result). Output was: {processed_output_str[:100]}...")
                         processed_output_str = str(result.raw) if hasattr(result, 'raw') else str(result)

            elif hasattr(result, 'raw') and isinstance(result.raw, str):
                 processed_output_str = result.raw # Check the raw output of the CrewOutput itself

            # --- Parse the extracted string --- 
            if isinstance(processed_output_str, str):
                # Split potentially multi-line JSON log string
                log_lines = processed_output_str.strip().split('\n')
                for line in log_lines:
                    try:
                        line_stripped = line.strip()
                        if line_stripped: # Avoid empty lines
                            # Handle potential escaping issues if LLM added extra quotes
                            if line_stripped.startswith('"') and line_stripped.endswith('"'):
                                try:
                                    # Try parsing as escaped JSON string first
                                    line_stripped = json.loads(line_stripped)
                                except json.JSONDecodeError:
                                     # If that fails, assume it was just extra quotes and remove them
                                     line_stripped = line_stripped[1:-1].replace('\\"', '"') # Handle potential escaped quotes within
                            
                            # Ensure we are parsing a string now
                            if isinstance(line_stripped, str):
                                parsed_experiences.append(json.loads(line_stripped))
                            elif isinstance(line_stripped, dict):
                                # If json.loads above already gave a dict
                                parsed_experiences.append(line_stripped)

                    except json.JSONDecodeError as json_err:
                        logger.warning(f"Skipping line due to JSON decode error: {json_err} - Line: '{line[:100]}...'" )
                    except Exception as line_err:
                        logger.warning(f"Skipping line due to unexpected error: {line_err} - Line: '{line[:100]}...'" )
            else:
                 logger.warning(f"Could not extract a processable string from CrewAI output.")

            logger.info(f"Successfully parsed {len(parsed_experiences)} experiences from the log.")
            return parsed_experiences
        
        except Exception as e:
            logger.error(f"Error during CrewAI kickoff or processing: {e}", exc_info=True)
            return [] # Return empty list on error

def count_trades_for_date(log_file_path, target_date_str):
    trade_count = 0
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    if 'timestamp' in log_entry and 'action' in log_entry:
                        # Extract the date part from the timestamp
                        timestamp_dt = datetime.fromisoformat(log_entry['timestamp'])
                        log_date_str = timestamp_dt.strftime('%Y-%m-%d')
                        
                        if log_date_str == target_date_str:
                            action = log_entry['action'].upper()
                            if action == "BUY" or action == "SELL":
                                trade_count += 1
                except json.JSONDecodeError:
                    print(f"Skipping malformed JSON line: {line.strip()}")
                except Exception as e:
                    print(f"Error processing log entry: {log_entry} - {e}")
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return -1 # Indicate error
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1 # Indicate error
    return trade_count

if __name__ == "__main__":
    print("Running Experience Processor Agent standalone test...")
    processor = ExperienceProcessorAgent()
    all_data = processor.process_all_experiences()
    print(f"\n--- Found {len(all_data)} total experiences ---")
    if all_data:
        # Print the first and last entry for brevity
        print("First entry:", all_data[0])
        if len(all_data) > 1:
            print("Last entry:", all_data[-1])
    else:
        print("No experiences found in the log or an error occurred.")

    target_date = "2025-05-05"
    trades = count_trades_for_date(LOG_FILE_PATH, target_date)
    
    if trades != -1:
        print(f"Number of trades (BUY/SELL) on {target_date}: {trades}")
        if trades > 10:
            print(f"The number of trades on {target_date} ({trades}) is greater than 10.")
        else:
            print(f"The number of trades on {target_date} ({trades}) is not greater than 10.")
