import os
import sys
import logging
from decimal import Decimal
from dotenv import load_dotenv
from pydantic import ConfigDict
from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
from langchain_openai import ChatOpenAI
from typing import Optional, Type

# Add project root to Python path to allow importing broker logic
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import RobinhoodBroker
from brokers.robinhood_broker import RobinhoodBroker

# Load environment variables from .env file
load_dotenv()

# --- Tool Definitions ---
class PlaceMarketOrderTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Place Market Order"
    description: str = "Places a real market order (buy or sell) for a given crypto symbol and quantity. Input: symbol, side (buy/sell), quantity (string)."
    broker: Optional[RobinhoodBroker] = None # Will be set by TradeExecutionAgent

    def _run(self, symbol: str, side: str, quantity: str) -> str:
        if not self.broker:
            return "Error: Broker not initialized for PlaceMarketOrderTool."
        try:
            order_quantity = Decimal(quantity)
            self.broker.logger.info(f"[Tool] Attempting to place market order: {side} {order_quantity} {symbol}")
            result = self.broker.place_market_order(symbol=symbol, side=side, quantity=order_quantity)
            return f"Market order placement result: {result}"
        except ValueError as ve:
            return f"Error: Invalid quantity '{quantity}'. Must be a valid number. Details: {ve}"
        except Exception as e:
            self.broker.logger.exception(f"[Tool] Exception in PlaceMarketOrderTool: {e}")
            return f"Error placing market order: {e}"

class PlaceLimitOrderTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Place Limit Order"
    description: str = "Places a real limit order (buy or sell) for a given crypto symbol, quantity, and limit price. Input: symbol, side (buy/sell), quantity (string), limit_price (string)."
    broker: Optional[RobinhoodBroker] = None # Will be set by TradeExecutionAgent

    def _run(self, symbol: str, side: str, quantity: str, limit_price: str) -> str:
        if not self.broker:
            return "Error: Broker not initialized for PlaceLimitOrderTool."
        try:
            order_quantity = Decimal(quantity)
            price_limit = Decimal(limit_price)
            self.broker.logger.info(f"[Tool] Attempting to place limit order: {side} {order_quantity} {symbol} @ {price_limit}")
            result = self.broker.place_limit_order(symbol=symbol, side=side, quantity=order_quantity, limit_price=price_limit)
            return f"Limit order placement result: {result}"
        except ValueError as ve:
            return f"Error: Invalid quantity '{quantity}' or limit_price '{limit_price}'. Must be valid numbers. Details: {ve}"
        except Exception as e:
            self.broker.logger.exception(f"[Tool] Exception in PlaceLimitOrderTool: {e}")
            return f"Error placing limit order: {e}"

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
        
        # Initialize broker connection FIRST, as tools need it
        self.broker = self._initialize_broker()

        # Initialize tools if broker is available
        self.place_market_order_tool = None
        self.place_limit_order_tool = None
        if self.broker:
            self.place_market_order_tool = PlaceMarketOrderTool(broker=self.broker)
            self.place_limit_order_tool = PlaceLimitOrderTool(broker=self.broker)
            self.logger.info("Order placement tools initialized.")
        else:
            self.logger.error("Broker not initialized. Order placement tools cannot be created.")

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
            allow_delegation=False,
            # Tools will be dynamically added in execute_trade_task if available
        )
        
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

        # Prepare tools list
        available_tools = []
        if self.place_market_order_tool:
            available_tools.append(self.place_market_order_tool)
        if self.place_limit_order_tool:
            available_tools.append(self.place_limit_order_tool)

        if not self.broker or not available_tools:
            self.logger.error("Broker or essential trading tools are not available. Cannot execute trades.")
            # Optionally, create a specialist agent that can only report this fact
            error_specialist_agent = Agent(
                role='Trade Execution Monitor',
                goal='Report inability to execute trades due to system issues.',
                backstory='An agent that alerts when trading systems are offline or misconfigured.',
                verbose=True, llm=self.llm, allow_delegation=False
            )
            error_task_description = (
                f"The trading system for {symbol} is currently unable to place orders because the brokerage connection "
                f"or trading tools failed to initialize. Please inform the user that no trades can be executed at this time."
            )
            error_task = Task(
                description=error_task_description,
                expected_output="A clear message stating that trades cannot be executed and why.",
                agent=error_specialist_agent
            )
            error_crew = Crew(agents=[error_specialist_agent], tasks=[error_task], verbose=True)
            result = error_crew.kickoff()
            self.logger.info(f"Trade execution specialist (error mode) for {symbol} finished. Raw output: {result.raw}")
            return result.raw

        # Fetch and log portfolio data
        try:
            account_info = self.broker.get_account_info() # Changed from get_account()
            if account_info and isinstance(account_info, list) and len(account_info) > 0:
                acc_data = account_info[0]
                self.logger.info(f"DEBUG: Full acc_data structure: {acc_data}") 
                buying_power = acc_data.get('buying_power', 'N/A') # Direct access
                # Assuming portfolio_value might also be a direct key if present, or needs calculation
                total_equity = acc_data.get('portfolio_value', 'N/A') # Direct access or placeholder
                self.logger.info(f"💰 Portfolio: Buying Power: ${buying_power}, Total Equity: ${total_equity}")
            elif account_info:
                self.logger.info(f"💰 Account Info (raw): {account_info}")
            else:
                self.logger.warning("⚠️ Could not retrieve detailed account information.")
        except Exception as e:
            self.logger.error(f"Error fetching account information: {e}")

        try:
            asset_code_to_fetch = symbol.split('-')[0] if '-' in symbol else symbol
            holdings_info = self.broker.get_holdings(asset_code=asset_code_to_fetch)
            if holdings_info and isinstance(holdings_info, list) and len(holdings_info) > 0:
                asset_holding_wrapper = holdings_info[0] # This is the dict with 'results'
                self.logger.info(f"DEBUG: Full asset_holding_wrapper structure for {asset_code_to_fetch}: {asset_holding_wrapper}")
                
                quantity = 'N/A'
                avg_cost = 'N/A'
                market_value = 'N/A'

                if isinstance(asset_holding_wrapper.get('results'), list) and len(asset_holding_wrapper['results']) > 0:
                    actual_holding = asset_holding_wrapper['results'][0]
                    self.logger.info(f"DEBUG: Actual holding data for {asset_code_to_fetch}: {actual_holding}")
                    quantity = actual_holding.get('total_quantity', 'N/A') # Or 'quantity_available_for_trading'
                    # avg_cost and market_value are not in this API response, will remain N/A
                else:
                    self.logger.warning(f"'results' key missing or empty in asset_holding_wrapper for {asset_code_to_fetch}.")

                self.logger.info(f"📊 {asset_code_to_fetch} Holdings: Quantity: {quantity}, Avg Cost: ${avg_cost}, Market Value: ${market_value}")
            elif holdings_info:
                self.logger.warning(f"Holdings for {asset_code_to_fetch} received but not in expected list format or empty: {holdings_info}")
        except Exception as e:
            self.logger.error(f"Error fetching {asset_code_to_fetch} holdings: {e}")

        # Create the specialist agent for this task, now with tools
        specialist_agent = Agent(
            role='Trade Execution Specialist',
            goal='Precisely execute trading decisions based on provided strategies using available brokerage tools, manage orders, and report outcomes.',
            backstory=(
                'A meticulous and highly reliable execution agent, specializing in translating trading strategies '
                'into actionable orders on cryptocurrency exchanges. Ensures accuracy, monitors execution, '
                'and handles order management with precision using provided tools.'
            ),
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
            tools=available_tools # Pass the instantiated tools
        )

        execution_task_description = (
            f"You are the Trade Execution Specialist for {symbol}. "
            f"Your primary brokerage interface is `self.broker` (which you will interact with via provided tools). "
            f"Review the following trading strategy document carefully:\n\n"
            f"-- STRATEGY DOCUMENT START --\n{strategy_document}\n-- STRATEGY DOCUMENT END --\n\n"
            f"Your tasks are to:\n"
            f"1. Parse this strategy to identify concrete, actionable trading orders (e.g., BUY/SELL, symbol, quantity, order type, price limits if any). Symbol is always {symbol}.\n"
            f"2. If clear, actionable orders are found, YOU MUST USE THE PROVIDED TOOLS ('Place Market Order' or 'Place Limit Order') to execute them. Do NOT simulate.\n"
            f"3. For each intended order, confirm the action by stating what tool you are using and with what parameters (e.g., 'Using Place Market Order tool: symbol={symbol}, side=buy, quantity=0.1').\n"
            f"4. Execute the trade using the appropriate tool.\n"
            f"5. Report the OUTCOME of each trade as returned by the tool (e.g., 'Market order placement result: {{{{...order details...}}}}').\n"
            f"If the strategy is unclear, no actionable trades can be derived, or if market conditions for an order are not met (e.g. price not at limit), state that clearly and do not attempt to place an order.\n"
            f"Ensure quantities and prices are passed as strings to the tools."
        )
        
        task = Task(
            description=execution_task_description,
            expected_output="A summary of actions taken, including trade confirmations or a statement if no actions could be derived.",
            agent=specialist_agent,
            # context = any relevant context from portfolio if needed, or fetched live data
        )

        # Create and kickoff the execution crew
        execution_crew = Crew(
            agents=[specialist_agent],
            tasks=[task],
            verbose=True, # Set to True for detailed crew output
            # process=Process.sequential # If you have multiple tasks for the specialist
        )

        self.logger.info(f"Kicking off trade execution specialist for {symbol}...")
        try:
            result = execution_crew.kickoff(inputs={'trading_strategy_document': strategy_document})
            self.logger.info(f"Trade execution specialist for {symbol} finished. Raw output: {result.raw}")
            return result.raw # Return the raw string output
        except Exception as e:
            self.logger.error(f"Error during trade execution specialist kickoff for {symbol}: {e}", exc_info=True)
            return f"Error during trade execution: {e}"

# Example usage (for testing this agent standalone - will be integrated into orchestrator later)
if __name__ == "__main__":
    load_dotenv()
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sample_strategy_document = """
    **Trading Strategy for BTC-USD - Standalone Test Mode**

    **Recommended Actions:**
    No specific trading actions defined for standalone execution.
    This agent is designed to receive strategies from an orchestrator.
    """

    trade_agent = TradeExecutionAgent()
    
    if trade_agent.broker:
        trade_agent.logger.info("TradeExecutionAgent initialized with a connected broker.")
    else:
        trade_agent.logger.error("TradeExecutionAgent failed to initialize broker. Trading functions will not work.")

    trade_task_result = trade_agent.execute_trade_task(sample_strategy_document)
    
    # To run this task, you'd typically add it to a Crew and kickoff.
    # For now, just printing the task description:
    print("--- Trade Execution Task Result ---")
    print(trade_task_result)
