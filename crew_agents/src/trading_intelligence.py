import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv
import traceback

# --- Load environment variables early ---
print("DEBUG: Loading .env in trading_intelligence.py")
load_dotenv(override=True) # Load environment variables from .env file
# --- End environment variable loading ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dynamically add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'crew_agents', 'src'))

# Conditional imports with fallback mechanisms
def safe_import(module_name, fallback_class=None):
    try:
        return __import__(module_name)
    except ImportError:
        print(f"Warning: Could not import {module_name}")
        return fallback_class() if fallback_class else None

# Mock classes for fallback
class MockChatOpenAI:
    def __init__(self, *args, **kwargs):
        print("Warning: Using mock ChatOpenAI")
    
    def __call__(self, *args, **kwargs):
        return self

class MockAgent:
    def __init__(self, *args, **kwargs):
        print("Warning: Using mock Agent")
    
    def __call__(self, *args, **kwargs):
        return self

class MockTask:
    def __init__(self, *args, **kwargs):
        print("Warning: Using mock Task")
    
    def __call__(self, *args, **kwargs):
        return self

class MockCrew:
    def __init__(self, *args, **kwargs):
        print("Warning: Using mock Crew")
    
    def kickoff(self):
        return "Mock Crew Kickoff Result"

# Attempt to import with fallback
print("DEBUG: Attempting to import crewai...")
try:
    from crewai import Agent, Task, Crew
    print("DEBUG: Successfully imported crewai.")
except ImportError as e:
    # --- Print full traceback for import error ---
    print(f"DEBUG: Failed to import crewai. Error: {e}")
    print("DEBUG: Traceback:")
    traceback.print_exc()
    # --- End traceback print ---
    logger.warning("crewai package not found or import failed. Using mock implementations for Agent, Task, and Crew.")
    Agent = MockAgent
    Task = MockTask
    Crew = MockCrew

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = MockChatOpenAI

# Import local modules
from alt_crypto_data import AltCryptoDataProvider

class TradingIntelligenceEngine:
    def __init__(self, symbol: str = 'BTC-USD', data_provider: Optional[AltCryptoDataProvider] = None):
        """
        Initialize Trading Intelligence Engine
        
        Args:
            symbol (str): Cryptocurrency trading pair
            data_provider (AltCryptoDataProvider): Data provider for market data
        """
        self.symbol = symbol
        self.data_provider = data_provider or AltCryptoDataProvider()

        # Initialize CrewAI components with error handling
        try:
            # --- Debug print for API key before init ---
            openai_api_key = os.getenv('OPENAI_API_KEY')
            print(f"DEBUG: OPENAI_API_KEY inside TradingIntelligenceEngine.__init__: {openai_api_key is not None}")
            # --- End debug print ---
            print("DEBUG: Initializing ChatOpenAI...")
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.7,
                max_tokens=1000
            )
            logger.info("Successfully initialized ChatOpenAI LLM.")
            print("Successfully initialized ChatOpenAI LLM.") # Added print statement for confirmation

        except Exception as e:
            print(f"LLM Initialization Error: {e}")
            print(f"Detailed Exception: {repr(e)}")
            # Print traceback for detailed debugging
            print("--- LLM Init Traceback Start ---")
            traceback.print_exc() # Print the full traceback
            print("--- LLM Init Traceback End ---")
            logger.warning("Failed to initialize ChatOpenAI. Falling back to MockChatOpenAI.")
            print("Warning: Using mock ChatOpenAI")
            self.llm = MockChatOpenAI()

    def generate_comprehensive_trading_insights(self, historical_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Generate comprehensive trading insights using CrewAI agents
        
        Args:
            historical_data (pd.DataFrame): Historical price data
        
        Returns:
            Dict[str, Any]: Comprehensive market insights
        """
        try:
            # If no historical data provided, fetch it
            if historical_data is None:
                historical_data = self.data_provider.fetch_price_history(
                    symbol=self.symbol, 
                    days=365, 
                    interval='1h'
                )
            
            # Market Research Agent
            market_researcher = Agent(
                role='Crypto Market Researcher',
                goal='Analyze market trends and provide comprehensive insights',
                backstory='Expert in cryptocurrency market analysis with deep understanding of technical and fundamental factors',
                llm=self.llm,
                allow_delegation=False,  # Explicitly disable delegation
                verbose=True # Add verbose for more detailed output from this agent
            )

            # Technical Analysis Agent
            technical_analyst = Agent(
                role='Technical Analysis Specialist',
                goal='Perform advanced technical analysis on cryptocurrency markets',
                backstory='Skilled in interpreting complex technical indicators and market patterns',
                llm=self.llm,
                allow_delegation=False, # Also set for this agent for consistency in testing
                verbose=True # Add verbose for more detailed output from this agent
            )

            # Market Research Task
            market_research_task = Task(
                description=f'Analyze market trends for {self.symbol}, considering historical data, volatility, and potential market movements',
                agent=market_researcher,
                expected_output='Comprehensive market trend report with key insights and potential scenarios'
            )

            # Technical Analysis Task
            technical_analysis_task = Task(
                description=f'Perform detailed technical analysis on {self.symbol}, identifying key support/resistance levels, momentum, and potential trading signals',
                agent=technical_analyst,
                expected_output='Detailed technical analysis report with actionable trading insights'
            )

            # Create Crew
            market_intelligence_crew = Crew(
                agents=[market_researcher, technical_analyst],
                tasks=[market_research_task, technical_analysis_task],
                verbose=1
            )

            # Generate Insights
            insights = market_intelligence_crew.kickoff()

            # Combine insights with data provider results
            return {
                'symbol': self.symbol,
                'raw_insights': insights,
                'market_trends': self._extract_market_trends(insights),
                'technical_signals': self._extract_technical_signals(insights),
                'historical_data_summary': {
                    'total_periods': len(historical_data),
                    'price_range': {
                        'min': historical_data['close'].min(),
                        'max': historical_data['close'].max(),
                        'current': historical_data['close'].iloc[-1]
                    },
                    'volatility': historical_data['close'].std()
                }
            }
        
        except Exception as e:
            print(f"Insights Generation Error: {e}")
            return {
                'symbol': self.symbol,
                'error': str(e)
            }

    def _extract_market_trends(self, insights: str) -> Dict[str, Any]:
        """
        Extract market trends from insights text
        
        Args:
            insights (str): Raw insights text
        
        Returns:
            Dict[str, Any]: Extracted market trends
        """
        # Simplified trend extraction logic
        return {
            'bullish_indicators': 'Positive market sentiment' in insights,
            'bearish_indicators': 'Negative market sentiment' in insights,
            'volatility_expected': 'High volatility' in insights
        }

    def _extract_technical_signals(self, insights: str) -> Dict[str, Any]:
        """
        Extract technical signals from insights text
        
        Args:
            insights (str): Raw insights text
        
        Returns:
            Dict[str, Any]: Extracted technical signals
        """
        # Simplified signal extraction logic
        return {
            'buy_signal': 'Buy Signal' in insights,
            'sell_signal': 'Sell Signal' in insights,
            'hold_signal': 'Hold Signal' in insights
        }

    def extract_reward_function_design(self, insights: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract reward function design parameters from market insights
        
        Args:
            insights (Dict[str, Any]): Market insights
        
        Returns:
            Dict[str, float]: Reward function design parameters
        """
        # Implement reward function design based on market insights
        historical_data = insights.get('historical_data_summary', {})
        
        # Dynamic reward function design
        stability_factor = 0.5
        volatility = historical_data.get('volatility', 0)
        
        # Trading efficiency bonus based on market conditions
        trading_efficiency_bonus = 0.2
        
        return {
            'stability_factor': stability_factor,
            'volatility': volatility,
            'trading_efficiency_bonus': trading_efficiency_bonus
        }

    def generate_trading_strategy_recommendations(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading strategy recommendations
        
        Args:
            insights (Dict[str, Any]): Market insights
        
        Returns:
            Dict[str, Any]: Trading strategy recommendations
        """
        # Implement strategy generation logic
        technical_signals = insights.get('technical_signals', {})
        
        # Determine recommended action
        if technical_signals.get('buy_signal'):
            recommended_action = 'Buy'
        elif technical_signals.get('sell_signal'):
            recommended_action = 'Sell'
        else:
            recommended_action = 'Hold'
        
        return {
            'recommended_action': recommended_action,
            'entry_conditions': insights.get('raw_insights', 'No specific entry conditions')
        }

def main():
    """
    Main function to demonstrate TradingIntelligenceEngine
    """
    engine = TradingIntelligenceEngine(symbol='BTC-USD')
    insights = engine.generate_comprehensive_trading_insights()
    
    # Print results with formatting
    print("\n--- Market Insights for BTC-USD ---")
    # Use lowercase 'close' and handle potential missing keys gracefully
    price_range = insights.get('historical_data_summary', {}).get('price_range', {})
    print(f"Latest Price: ${price_range.get('current', 'N/A'):,.2f}")
    print(f"Price Range (Min): ${price_range.get('min', 'N/A'):,.2f}")
    print(f"Price Range (Max): ${price_range.get('max', 'N/A'):,.2f}")
    
    print("\n--- Historical Data Summary ---")
    hist_summary = insights.get('historical_data_summary', {})
    print(f"Total Periods: {hist_summary.get('total_periods', 'N/A')}")
    print(f"Volatility (Std Dev): {hist_summary.get('volatility', 'N/A'):,.4f}")

    # Print raw insights if available
    if 'raw_insights' in insights:
        print("\n--- Raw CrewAI Insights ---")
        print(insights['raw_insights'])
    elif 'error' in insights:
        print(f"\n--- Error --- ")
        print(insights['error'])

    # Example of using other methods (commented out by default)
    # reward_params = engine.design_reward_function(insights)
    # print("\n--- Reward Function Parameters ---")
    # print(reward_params)

    # strategy_rec = engine.generate_trading_strategy(insights)
    # print("\n--- Trading Strategy Recommendation ---")
    # print(strategy_rec)

if __name__ == "__main__":
    main()
