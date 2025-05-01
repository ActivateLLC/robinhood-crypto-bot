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
    
    def create_market_analysis_crew(self, historical_data):
        """
        Create a crew for comprehensive market analysis
        
        Args:
            historical_data (pd.DataFrame): Historical price and indicator data
        
        Returns:
            Crew: CrewAI crew for market analysis
        """
        # Market Research Task
        market_research_task = Task(
            description=f"Analyze current market conditions for {self.symbol}. "
                        "Identify key trends, sentiment, and potential market-moving events.",
            agent=self.market_researcher,
            expected_output="Comprehensive market analysis report with key insights"
        )
        
        # Technical Analysis Task
        technical_analysis_task = Task(
            description=f"Perform detailed technical analysis for {self.symbol}. "
                        "Interpret key indicators and identify potential trading signals.",
            agent=self.technical_analyst,
            context=[market_research_task],
            expected_output="Technical analysis report with trading signals and key levels"
        )
        
        # Risk Assessment Task
        risk_assessment_task = Task(
            description=f"Assess potential risks for trading {self.symbol}. "
                        "Evaluate market volatility, liquidity, and potential downside scenarios.",
            agent=self.risk_manager,
            context=[market_research_task, technical_analysis_task],
            expected_output="Comprehensive risk assessment with mitigation strategies"
        )
        
        # Strategy Generation Task
        strategy_generation_task = Task(
            description=f"Generate a refined trading strategy for {self.symbol}. "
                        "Integrate market research, technical analysis, and risk assessment.",
            agent=self.strategy_generator,
            context=[market_research_task, technical_analysis_task, risk_assessment_task],
            expected_output="Detailed trading strategy with entry/exit criteria and risk management"
        )
        
        # Create and execute crew
        crew = Crew(
            agents=[
                self.market_researcher, 
                self.technical_analyst, 
                self.risk_manager, 
                self.strategy_generator
            ],
            tasks=[
                market_research_task, 
                technical_analysis_task, 
                risk_assessment_task, 
                strategy_generation_task
            ],
            verbose=2
        )
        
        return crew
    
    def analyze_market(self, historical_data):
        """
        Perform comprehensive market analysis
        
        Args:
            historical_data (pd.DataFrame): Historical price and indicator data
        
        Returns:
            dict: Analysis results from different agents
        """
        # Create market analysis crew
        market_crew = self.create_market_analysis_crew(historical_data)
        
        # Kickoff the crew and get results
        result = market_crew.kickoff()
        
        return {
            'market_research': result,
            'technical_analysis': market_crew.tasks[1].output,
            'risk_assessment': market_crew.tasks[2].output,
            'trading_strategy': market_crew.tasks[3].output
        }
