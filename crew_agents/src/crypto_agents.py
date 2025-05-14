import os
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
            dict: Analysis results from different agents (strings)
        """
        # Create market analysis crew
        market_crew = self.create_market_analysis_crew(historical_data)
        
        # Kickoff the crew and get results
        crew_kickoff_result = market_crew.kickoff() # This is a CrewOutput object

        # --- Start Debug Prints ---
        print(f"\n--- DEBUGGING CryptoTradingAgents.analyze_market ---")
        print(f"DEBUG: crew_kickoff_result raw object: {crew_kickoff_result}")
        if crew_kickoff_result:
            print(f"DEBUG: type(crew_kickoff_result): {type(crew_kickoff_result)}")
            # Try to access common attributes of CrewOutput if they exist
            print(f"DEBUG: crew_kickoff_result.description: {getattr(crew_kickoff_result, 'description', 'N/A')}")
            print(f"DEBUG: crew_kickoff_result.raw (if any): {getattr(crew_kickoff_result, 'raw', 'N/A')}")
            print(f"DEBUG: crew_kickoff_result.tasks_output: {getattr(crew_kickoff_result, 'tasks_output', 'N/A')}")
            
            tasks_outputs_attr = getattr(crew_kickoff_result, 'tasks_output', None)
            if tasks_outputs_attr:
                print(f"DEBUG: Number of tasks_output items: {len(tasks_outputs_attr)}")
                for i, task_out in enumerate(tasks_outputs_attr):
                    print(f"  DEBUG: Task {i} output object: {task_out}")
                    print(f"  DEBUG: Task {i} type: {type(task_out)}")
                    print(f"  DEBUG: Task {i} description: {getattr(task_out, 'description', 'N/A')}")
                    print(f"  DEBUG: Task {i} raw_output: {getattr(task_out, 'raw_output', 'N/A')}")
                    # Also check for 'exported_output' or 'output' if raw_output is consistently N/A
                    print(f"  DEBUG: Task {i} exported_output: {getattr(task_out, 'exported_output', 'N/A')}")
                    print(f"  DEBUG: Task {i} agent_id: {getattr(task_out, 'agent_id', 'N/A')}") # Might be agent_id or agent
            else:
                print(f"DEBUG: crew_kickoff_result.tasks_output is None or empty.")
        else:
            print(f"DEBUG: crew_kickoff_result is None.")
        print(f"--- END DEBUGGING CryptoTradingAgents.analyze_market ---\n")
        # --- End Debug Prints ---

        analysis = {
            'market_research': "N/A - Market Research Output Missing",
            'technical_analysis': "N/A - Technical Analysis Output Missing",
            'risk_assessment': "N/A - Risk Assessment Output Missing",
            'trading_strategy': "N/A - Trading Strategy Output Missing"
        }

        tasks_outputs = crew_kickoff_result.tasks_output if crew_kickoff_result and hasattr(crew_kickoff_result, 'tasks_output') else []

        key_map = ['market_research', 'technical_analysis', 'risk_assessment', 'trading_strategy']

        for i, task_key in enumerate(key_map):
            if i < len(tasks_outputs) and tasks_outputs[i]:
                task_out = tasks_outputs[i]
                extracted_text = None

                # Try exported_output
                if hasattr(task_out, 'exported_output') and isinstance(task_out.exported_output, str):
                    candidate = task_out.exported_output.strip()
                    if candidate:
                        extracted_text = candidate
                
                # Try raw_output if exported_output wasn't usable
                if extracted_text is None and hasattr(task_out, 'raw_output') and isinstance(task_out.raw_output, str):
                    candidate = task_out.raw_output.strip()
                    if candidate:
                        extracted_text = candidate

                # Try str(task_out) as a last resort if others failed
                if extracted_text is None:
                    try:
                        candidate = str(task_out).strip()
                        # Avoid generic object representations like '<crewai.tasks.task_output.TaskOutput object at 0x...>' or similar
                        if candidate and not (candidate.startswith('<') and 'object at 0x' in candidate and candidate.endswith('>')):
                           # Further check to ensure it's not just the default repr if it's very short or contains the class name prominently
                            if len(candidate) > 50 or not task_out.__class__.__name__ in candidate[:len(task_out.__class__.__name__)+10]: 
                                extracted_text = candidate
                    except Exception:
                        pass # Ignore errors from str(task_out)

                if extracted_text:
                    analysis[task_key] = extracted_text
            
        return analysis

    def create_market_analysis_crew(self, historical_data):
        """
        Creates and configures the market analysis crew.
        
        Args:
            historical_data (pd.DataFrame): Historical price data.
            
        Returns:
            Crew: Configured CrewAI crew for market analysis.
        """
        # Instantiate agents
        market_researcher = self.market_researcher
        technical_analyst = self.technical_analyst
        risk_manager = self.risk_manager
        strategy_generator = self.strategy_generator

        # Define tasks for each agent
        # Task for Market Researcher
        market_research_task = Task(
            description=(
                f"Analyze current market conditions for {self.symbol}. "
                f"Consider macroeconomic factors, news, sentiment, and overall crypto market trends. "
                f"Focus on data from the last {30} days."
            ),
            expected_output=(
                f"A comprehensive report on market conditions for {self.symbol}, including key influencing factors, "
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
                f"and an overall strategic approach (e.g., trend-following, range-trading)."
            ),
            expected_output=(
                f"A comprehensive trading strategy document for {self.symbol}. Should include: "
                f"1. Summary of integrated analysis. 2. Defined entry criteria (conditions for buy/sell). "
                f"3. Defined exit criteria (take-profit and stop-loss levels). 4. Recommended position sizing. "
                f"5. Overall strategic approach and confidence level."
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
