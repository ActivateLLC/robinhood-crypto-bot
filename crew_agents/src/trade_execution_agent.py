import os
import sys
import logging
from dotenv import load_dotenv
from crewai import Agent, Task
from langchain_openai import ChatOpenAI
from typing import Optional

# Add project root to Python path to allow importing broker logic
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import RobinhoodBroker
from brokers.robinhood_broker import RobinhoodBroker

# Load environment variables from .env file
load_dotenv()

class TradeExecutionAgent:
    def __init__(self):
        # Setup logger for this agent
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.3,
            max_tokens=2048
        )
        
        self.agent = Agent(
            role='Trade Execution Specialist',
            goal='Precisely execute trading decisions based on provided strategies, manage orders, and interact with brokerage APIs.',
            backstory=(
                'A meticulous and highly reliable execution agent, specializing in translating trading strategies '
                'into actionable orders on cryptocurrency exchanges. Ensures accuracy, monitors execution, '
                'and handles order management with precision.'
            ),
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        # Initialize broker connection
        self.broker = self._initialize_broker()

    def _initialize_broker(self) -> Optional[RobinhoodBroker]:
        """
        Initializes and connects the RobinhoodBroker.
        """
        self.logger.info("Initializing RobinhoodBroker...")
        try:
            api_key = os.getenv("RH_API_KEY")
            private_key_b64 = os.getenv("RH_PRIVATE_KEY")

            if not api_key or not private_key_b64:
                self.logger.error("Robinhood API Key (RH_API_KEY) or Private Key (RH_PRIVATE_KEY) not found in environment variables.")
                return None

            broker_config = {
                "api_key": api_key,
                "base64_private_key": private_key_b64
            }
            
            broker = RobinhoodBroker(config=broker_config, logger=self.logger)
            self.logger.info("RobinhoodBroker instantiated. Attempting to connect...")
            
            if broker.connect():
                self.logger.info("Successfully connected to Robinhood.")
                return broker
            else:
                self.logger.error("Failed to connect to Robinhood.")
                return None
        except Exception as e:
            self.logger.exception(f"Error initializing RobinhoodBroker: {e}")
            return None

    def execute_trade_task(self, strategy_document: str, symbol: str = "BTC-USD"):
        self.logger.info(f"🤖 TradeExecutionAgent received task for {symbol} with strategy:\n{strategy_document[:500]}...") # Log first 500 chars

        # Fetch and log portfolio data
        try:
            account_info = self.broker.get_account()
            if account_info and isinstance(account_info, list) and len(account_info) > 0:
                # Assuming the first account in the list is the relevant one
                acc_data = account_info[0]
                buying_power = acc_data.get('buying_power', {}).get('amount', 'N/A')
                cash_held_for_orders = acc_data.get('cash_held_for_orders', {}).get('amount', 'N/A')
                withdrawable_cash = acc_data.get('withdrawable_cash', {}).get('amount', 'N/A')
                self.logger.info(f"💰 Account Info: Buying Power: ${buying_power}, Cash Held for Orders: ${cash_held_for_orders}, Withdrawable Cash: ${withdrawable_cash}")
            elif account_info:
                 self.logger.info(f"💰 Account Info (raw): {account_info}") # Log raw if structure is unexpected
            else:
                self.logger.warning("⚠️ Could not retrieve detailed account information.")
        except Exception as e:
            self.logger.error(f"Error fetching account information: {e}")

        try:
            # Extract the base currency from the symbol (e.g., BTC from BTC-USD)
            asset_code_to_fetch = symbol.split('-')[0] if '-' in symbol else symbol
            holdings_info = self.broker.get_holdings(asset_code_to_fetch)
            if holdings_info and isinstance(holdings_info, list) and len(holdings_info) > 0:
                # Assuming the API returns a list and the first item is our asset if found
                asset_holding = holdings_info[0] # This might need adjustment based on actual API response for a single queried asset
                quantity = asset_holding.get('quantity', {}).get('amount', 'N/A')
                avg_cost = asset_holding.get('average_cost', {}).get('amount', 'N/A') # If available
                market_value = asset_holding.get('market_value', {}).get('amount', 'N/A') # If available
                self.logger.info(f"📊 {asset_code_to_fetch} Holdings: Quantity: {quantity}, Avg Cost: ${avg_cost}, Market Value: ${market_value}")
            elif holdings_info: # It might return an empty list if no holdings for that asset
                 self.logger.info(f"📊 {asset_code_to_fetch} Holdings (raw): {holdings_info}")
            else:
                self.logger.info(f"ℹ️ No specific holdings found for {asset_code_to_fetch} or API returned no data.")
        except Exception as e:
            self.logger.error(f"Error fetching {asset_code_to_fetch} holdings: {e}")

        return Task(
            description=(
                f"Review the following trading strategy document carefully:\n\n"
                f"-- STRATEGY DOCUMENT START --\n{strategy_document}\n-- STRATEGY DOCUMENT END --\n\n"
                f"Your tasks are to:\n"
                f"1. Parse this strategy to identify concrete, actionable trading orders (e.g., BUY/SELL, symbol, quantity, order type, price limits if any)."
                f"2. If clear, actionable orders are found, prepare to execute them using the brokerage interface (self.broker)."
                f"3. For each intended order, confirm the action (e.g., 'Attempting to BUY 0.1 BTC-USD at market using self.broker.place_market_order(...)')."
                f"4. (Simulated) Execute the trade. For now, log the trade details. In a real scenario, you would call the appropriate self.broker method."
                f"5. Report the outcome of each attempted trade (e.g., 'Simulated BUY order for 0.1 BTC-USD placed successfully')."
                f"If the strategy is unclear, or no actionable trades can be derived, state that clearly."
            ),
            agent=self.agent,
            expected_output=(
                "A summary of actions taken: For each trade, include the intended action (BUY/SELL, symbol, quantity, type), "
                "confirmation of attempt, and the (simulated) execution status. If no trades were actionable, provide a clear statement."
            )
        )

# Example usage (for testing this agent standalone - will be integrated into orchestrator later)
if __name__ == '__main__':
    execution_agent_handler = TradeExecutionAgent()
    
    if execution_agent_handler.broker:
        execution_agent_handler.logger.info("TradeExecutionAgent initialized with a connected broker.")
    else:
        execution_agent_handler.logger.error("TradeExecutionAgent failed to initialize broker. Trading functions will not work.")

    sample_strategy = ("""
    **Trading Strategy for BTC-USD - May 14, 2025**

    **Market Overview:**
    The market shows consolidation around $60,000. Key support at $58,000, resistance at $62,000.
    Sentiment is cautiously optimistic due to upcoming ETF news.

    **Technical Signals:**
    - RSI (14): 55 (Neutral)
    - MACD: Bullish crossover on 4H chart.
    - Moving Averages: Price is above 50-period MA, below 200-period MA on daily.

    **Recommended Actions:**
    1. **Entry Long:** If BTC-USD breaks above $62,500 with strong volume, consider a long position.
       Target: $65,000. Stop-loss: $61,500. Allocate 25% of trading capital.
    2. **Short Opportunity:** If BTC-USD breaks below $57,500, consider a short position.
       Target: $55,000. Stop-loss: $58,500. Allocate 15% of trading capital.
    3. **Hold:** If price remains between $58,000 and $62,000, maintain current positions and observe.

    **Risk Management:**
    - Do not risk more than 2% of total portfolio on any single trade.
    - Adjust stop-losses if volatility increases significantly.
    """)

    trade_task = execution_agent_handler.execute_trade_task(sample_strategy)
    
    # To run this task, you'd typically add it to a Crew and kickoff.
    # For now, just printing the task description:
    print("--- Trade Execution Task Description ---")
    print(trade_task.description)
    print("--- Expected Output --- ")
    print(trade_task.expected_output)
    # In a real scenario: 
    # from crewai import Crew
    # crew = Crew(agents=[execution_agent_handler.agent], tasks=[trade_task], verbose=True)
    # result = crew.kickoff()
    # print("\n--- Execution Result ---")
    # print(result)
