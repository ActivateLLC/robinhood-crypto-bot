import os
from dotenv import load_dotenv

# Load environment variables from .env file in the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env') # Go up two levels from src
if not os.path.exists(dotenv_path):
    # Fallback for when script might be run from root or crew_agents directly
    dotenv_path_alt = os.path.join(os.path.dirname(__file__), '..', '.env') 
    if os.path.exists(dotenv_path_alt):
        dotenv_path = dotenv_path_alt
    else: # If still not found, try project root assuming script is in project_root/crew_agents/src
        # This is a bit of a guess, ideally the script is run from project root
        dotenv_path_current_level = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(dotenv_path_current_level):
            dotenv_path = dotenv_path_current_level
        else: # Final fallback to assuming script is run from the project root
             dotenv_path = os.path.join(os.getcwd(), '.env')


load_dotenv(dotenv_path=dotenv_path, override=True)
# Add a print statement to confirm which .env is being used
print(f"Attempting to load .env from: {os.path.abspath(dotenv_path)}")
if os.getenv('OPENAI_API_KEY'):
    print("OPENAI_API_KEY found in environment after dotenv load.")
else:
    print("WARNING: OPENAI_API_KEY NOT found after dotenv load. Please check .env path and content.")

import pandas as pd
import numpy as np
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

class CryptoTradingAgents:
    def __init__(self, symbol='BTC-USD', data_provider=None):
        """
        Initialize Crypto Trading Agents with specialized roles
        
        Args:
            symbol (str): Cryptocurrency trading pair
            data_provider (object): Data provider for market information
        """
        # Initialize language model
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.7,
            max_tokens=4096
        )
        
        # Store symbol and data provider
        self.symbol = symbol
        self.data_provider = data_provider
        
        # Create specialized agents
        self.market_researcher = self._create_market_researcher()
        self.technical_analyst = self._create_technical_analyst()
        self.risk_manager = self._create_risk_manager()
        self.strategy_generator = self._create_strategy_generator()
        
    def _create_market_researcher(self):
        """Create an agent for comprehensive market research"""
        return Agent(
            role='Crypto Market Researcher',
            goal='Analyze global market trends, news, and macroeconomic factors '
                 'affecting cryptocurrency markets',
            backstory='An experienced financial analyst with deep understanding '
                      'of global crypto market dynamics and emerging trends',
            verbose=True,
            llm=self.llm
        )
    
    def _create_technical_analyst(self):
        """Create an agent for technical analysis and indicator interpretation"""
        return Agent(
            role='Crypto Technical Analyst',
            goal='Perform in-depth technical analysis using advanced indicators '
                 'and identify potential trading opportunities',
            backstory='A seasoned trader specializing in advanced technical '
                      'analysis techniques for cryptocurrency markets',
            verbose=True,
            llm=self.llm
        )
    
    def _create_risk_manager(self):
        """Create an agent focused on risk assessment and management"""
        return Agent(
            role='Crypto Risk Manager',
            goal='Evaluate and mitigate potential risks in trading strategies, '
                 'ensuring robust risk management',
            backstory='A meticulous risk assessment expert with extensive '
                      'experience in financial risk modeling',
            verbose=True,
            llm=self.llm
        )
    
    def _create_strategy_generator(self):
        """Create an agent for generating and refining trading strategies"""
        return Agent(
            role='Crypto Strategy Generator',
            goal='Develop and optimize trading strategies based on market research, '
                 'technical analysis, and risk assessment',
            backstory='An innovative strategist who combines quantitative analysis '
                      'with adaptive machine learning techniques',
            verbose=True,
            llm=self.llm
        )
    
    def analyze_market(self, historical_data):
        """
        Perform comprehensive market analysis
        
        Args:
            historical_data (pd.DataFrame): Historical price and indicator data
        
        Returns:
            dict: Analysis results from different agents (strings), including 'recommended_action'
        """
        # Create market analysis crew
        market_crew = self.create_market_analysis_crew(historical_data)
        
        # Kickoff the crew and get results
        crew_kickoff_result = market_crew.kickoff() # This is a CrewOutput object

        # --- Start Debug Prints --- (Consider using logging for production)
        print(f"\n--- DEBUGGING CryptoTradingAgents.analyze_market ---")
        print(f"DEBUG: crew_kickoff_result raw object: {crew_kickoff_result}")
        if crew_kickoff_result:
            print(f"DEBUG: type(crew_kickoff_result): {type(crew_kickoff_result)}")
            # Try to access common attributes of CrewOutput if it's an object
            if hasattr(crew_kickoff_result, 'raw'):
                print(f"DEBUG: crew_kickoff_result.raw: {crew_kickoff_result.raw[:500]}...") # Print first 500 chars
            if hasattr(crew_kickoff_result, 'tasks_output'):
                print(f"DEBUG: crew_kickoff_result has tasks_output. Count: {len(crew_kickoff_result.tasks_output) if crew_kickoff_result.tasks_output else 0}")
                if crew_kickoff_result.tasks_output:
                    # Assuming the last task's output is the most relevant for the final strategy document
                    last_task_output = crew_kickoff_result.tasks_output[-1]
                    if hasattr(last_task_output, 'raw_output'):
                         print(f"DEBUG: Last task raw_output: {last_task_output.raw_output[:500]}...")
                    elif isinstance(last_task_output, str):
                        print(f"DEBUG: Last task output (str): {last_task_output[:500]}...")
        print(f"--- END DEBUGGING CryptoTradingAgents.analyze_market ---")
        # --- End Debug Prints ---

        results = {}
        final_decision_text = ""

        # Extract the raw text output from the last task, which should contain the strategy document
        if crew_kickoff_result:
            if hasattr(crew_kickoff_result, 'tasks_output') and crew_kickoff_result.tasks_output:
                # The last task in the sequential crew is the strategy_generator
                last_task_output_obj = crew_kickoff_result.tasks_output[-1]
                if hasattr(last_task_output_obj, 'exported_output') and last_task_output_obj.exported_output:
                     final_decision_text = last_task_output_obj.exported_output
                elif hasattr(last_task_output_obj, 'raw_output') and last_task_output_obj.raw_output: # Fallback to raw_output
                     final_decision_text = last_task_output_obj.raw_output
                elif isinstance(last_task_output_obj, str): # If the task output itself is a string
                    final_decision_text = last_task_output_obj
            elif hasattr(crew_kickoff_result, 'raw') and crew_kickoff_result.raw: # Fallback for older crewai versions or different structures
                final_decision_text = crew_kickoff_result.raw
            elif isinstance(crew_kickoff_result, str):
                final_decision_text = crew_kickoff_result

        # Store the full strategy document if needed, and other agent outputs if parsed
        # For now, we are primarily interested in the final_decision_text from the strategy generator
        results['strategy_document'] = final_decision_text 

        # Simple parsing for "Recommended immediate action: [ACTION]"
        parsed_action = 'hold' # Default action
        action_line_prefix = "Recommended immediate action:"
        
        found_action_line = False
        for line in final_decision_text.lower().splitlines():
            if action_line_prefix.lower() in line:
                action_part = line.split(action_line_prefix.lower())[-1].strip() # Get text after the prefix
                # Remove any potential markdown like ** or trailing characters like . or !
                action_part = action_part.replace('*','').replace('[','').replace(']','').split('.')[0].split('!')[0].strip()
                
                if 'buy' in action_part:
                    parsed_action = 'buy'
                    found_action_line = True
                    break
                elif 'sell' in action_part:
                    parsed_action = 'sell'
                    found_action_line = True
                    break
                elif 'hold' in action_part: # Explicit hold might be stated
                    parsed_action = 'hold'
                    found_action_line = True
                    break
        
        results['recommended_action'] = parsed_action
        # Use logging, but keep print for direct script runs during testing
        # logging.info(f"CrewAI recommended action parsed: {parsed_action}. Found in text: {found_action_line}") 
        print(f"DEBUG: Parsed recommended_action: {parsed_action}. Found in text: {found_action_line}")
        print(f"DEBUG: Full decision text for parsing was (first 500 chars): {final_decision_text[:500]}...")

        # Placeholder for other agent outputs if they were individually retrievable and structured
        # results['market_research'] = "..."
        # results['technical_analysis'] = "..."
        # results['risk_assessment'] = "..."

        return results

    def create_market_analysis_crew(self, historical_data):
        """
        Creates and configures the market analysis crew.
        
        Args:
            historical_data (pd.DataFrame): Historical price data.
            
        Returns:
            Crew: Configured CrewAI crew for market analysis.
        """
        # Create instances of agents to be used in this crew
        # This ensures fresh state if agents were modified or had memory in other crews
        market_researcher = self._create_market_researcher()
        technical_analyst = self._create_technical_analyst()
        risk_manager = self._create_risk_manager()
        strategy_generator = self._create_strategy_generator()

        # Define Tasks for each agent
        # Task for Market Researcher
        market_research_task = Task(
            description=(
                f"Analyze current market conditions for {self.symbol}. Focus on news, sentiment, macroeconomic factors, "
                f"and any specific events relevant to {self.symbol}. Provide a concise summary."
            ),
            expected_output=(
                f"A summary of current market conditions, sentiment analysis, and key influencing factors for {self.symbol}. "
                f"recent trends, and potential upcoming catalysts. Highlight any significant bullish or bearish signals."
            ),
            agent=market_researcher
        )

        # Task for Technical Analyst
        technical_analysis_task = Task(
            description=(
                f"Perform detailed technical analysis for {self.symbol} using the provided historical data. "
                f"Calculate and interpret key indicators like Moving Averages (SMA, EMA), RSI, MACD, and Bollinger Bands. "
                f"Identify support/resistance levels, chart patterns, and potential trading signals (buy/sell/hold)."
            ),
            expected_output=(
                f"A detailed technical analysis report for {self.symbol}. Include interpretations of key indicators, identified patterns, "
                f"support/resistance levels, and any clear buy/sell/hold signals derived from the analysis. "
                f"The historical data is: \n{historical_data.to_string()}"
            ),
            agent=technical_analyst,
            context=[market_research_task]  # Depends on market research
        )

        # Task for Risk Manager
        risk_management_task = Task(
            description=(
                f"Assess the risks associated with trading {self.symbol} based on the market research and technical analysis. "
                f"Identify potential downside scenarios, volatility concerns, and liquidity risks. "
                f"Suggest risk mitigation strategies (e.g., stop-loss levels, position sizing)."
            ),
            expected_output=(
                f"A risk assessment report for {self.symbol}. Detail potential risks, their likelihood, and impact. "
                f"Propose specific risk mitigation techniques, including recommended stop-loss levels and position sizing rules."
            ),
            agent=risk_manager,
            context=[market_research_task, technical_analysis_task]  # Depends on both
        )

        # Task for Strategy Generator
        strategy_generation_task = Task(
            description=(
                f"Generate a refined trading strategy for {self.symbol}. Integrate the findings from market research, "
                f"technical analysis, and risk assessment. Define clear entry and exit criteria, target profit levels, "
                f"and an overall strategic approach. Based on all this, conclude with an immediate recommended trading action for the current moment."
            ),
            expected_output=(
                f"A comprehensive trading strategy document for {self.symbol}. Should include: "
                f"1. Summary of integrated analysis. 2. Defined entry criteria (conditions for buy/sell). "
                f"3. Defined exit criteria (take-profit and stop-loss levels). 4. Recommended position sizing. "
                f"5. Overall strategic approach and confidence level. "
                f"6. **Recommended immediate action: [BUY/SELL/HOLD]** based on the current synthesis of information."
            ),
            agent=strategy_generator,
            context=[market_research_task, technical_analysis_task, risk_management_task]  # Depends on all previous
        )

        # Assemble the crew
        return Crew(
            agents=[market_researcher, technical_analyst, risk_manager, strategy_generator],
            tasks=[market_research_task, technical_analysis_task, risk_management_task, strategy_generation_task],
            verbose=True,
            # process=Process.sequential # Ensure sequential execution if not default
            # memory=True # To enable memory for context carry-over if needed across multiple kickoffs
        )

# Example of how to run this if it were the main script (for testing purposes)
if __name__ == "__main__":
    crypto_agents = CryptoTradingAgents(symbol='BTC-USD')
    historical_data = pd.DataFrame({
        'Date': ['2022-01-01', '2022-01-02', '2022-01-03'],
        'Open': [100, 120, 110],
        'High': [120, 130, 125],
        'Low': [90, 110, 105],
        'Close': [110, 125, 115]
    })
    analysis = crypto_agents.analyze_market(historical_data)
    print(analysis)
