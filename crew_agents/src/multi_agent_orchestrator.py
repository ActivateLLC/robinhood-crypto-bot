import os
import sys
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import traceback
import pandas as pd
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import all agents and communication hub
from crew_agents.src.market_intelligence_agent import MarketIntelligenceAgent
from crew_agents.src.crew_optimization_manager import CrewOptimizationAgent
from crew_agents.src.continuous_optimization_agent import ContinuousOptimizationAgent
from crew_agents.src.unified_optimization_agent import UnifiedOptimizationAgent
from crew_agents.src.btc_popcat_research_agent import BTCPopCatResearchAgent
from crew_agents.src.agent_communication_hub import AgentCommunicationHub
from crew_agents.src.trade_execution_agent import TradeExecutionAgent
from crew_agents.src.crypto_agents import CryptoTradingAgents
from crewai import Crew

# Configure comprehensive logging
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'multi_agent_orchestrator.log')

# Use RotatingFileHandler for log rotation
from logging.handlers import RotatingFileHandler

# Configure logging with more detailed settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10 MB per log file
            backupCount=5,  # Keep 5 backup log files
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)

# Set up a global logger with additional configuration
logger = logging.getLogger('MultiAgentOrchestrator')
logger.setLevel(logging.INFO)

# Add exception hook for unhandled exceptions
def global_exception_handler(exc_type, exc_value, exc_traceback):
    """
    Global exception handler to log unhandled exceptions
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.error(
        "Uncaught exception", 
        exc_info=(exc_type, exc_value, exc_traceback)
    )

sys.excepthook = global_exception_handler

class MultiAgentOrchestrator:
    def __init__(self):
        # These instances are for other potential threaded jobs within the orchestrator itself.
        # They will NOT be passed to the launch_crypto_agents process.
        self.crypto_agents_local_instance = CryptoTradingAgents() 
        self.communication_hub = AgentCommunicationHub()
        self.trade_executor_local_instance = TradeExecutionAgent()   
        logger.info("MultiAgentOrchestrator initialized its own local agent instances (if any are used by its direct methods).")

    def _market_intelligence_job(self):
        """
        Market Intelligence Agent's primary job
        Continuously gather and analyze market data
        """
        logger.info("🌐 Market Intelligence Agent Activated")
        while True:
            try:
                # Comprehensive market trend analysis
                market_insights = MarketIntelligenceAgent().analyze_market_trends()
                
                # Detect arbitrage opportunities
                for symbol in ['BTC-USD', 'ETH-USD', 'BNB-USD']:
                    arbitrage_ops = MarketIntelligenceAgent()._detect_arbitrage(symbol)
                    if arbitrage_ops:
                        logger.info(f"🔍 Arbitrage Opportunities for {symbol}: {arbitrage_ops}")
                
                time.sleep(3600)  # Hourly analysis
            
            except Exception as e:
                logger.error(f"Market Intelligence Job Error: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
    
    def _optimization_job(self):
        """
        Crew Optimization Agent's primary job
        Continuous strategy optimization and portfolio management
        """
        logger.info("🚀 Crew Optimization Agent Activated")
        while True:
            try:
                # Run optimization pipeline for each symbol
                for symbol in ['BTC-USD', 'ETH-USD', 'BNB-USD']:
                    optimization_results = CrewOptimizationAgent(symbol=symbol, initial_capital=100000).run_optimization_pipeline(symbol)
                    logger.info(f"📊 Optimization Results for {symbol}: {optimization_results}")
                
                time.sleep(86400)  # Daily optimization
            
            except Exception as e:
                logger.error(f"Optimization Job Error: {e}")
                time.sleep(3600)  # Wait 1 hour before retry
    
    def _continuous_trading_job(self):
        """
        Continuous Optimization Agent's primary job
        Real-time trading strategy adjustments
        """
        logger.info("⚡ Continuous Optimization Agent Activated")
        while True:
            try:
                # Run continuous optimization
                ContinuousOptimizationAgent(symbols=['BTC-USD', 'ETH-USD', 'BNB-USD'], optimization_interval=3600, initial_capital=100000).run_continuous_optimization()
                
                time.sleep(1800)  # Every 30 minutes
            
            except Exception as e:
                logger.error(f"Continuous Trading Job Error: {e}")
                time.sleep(600)  # Wait 10 minutes before retry
    
    def _unified_optimization_job(self):
        """
        Unified Optimization Agent's primary job
        Unified optimization pipeline
        """
        logger.info("📈 Unified Optimization Agent Activated")
        while True:
            try:
                # Run unified optimization pipeline
                optimization_results = UnifiedOptimizationAgent(symbols=['BTC-USD', 'ETH-USD', 'BNB-USD']).run_optimization_pipeline()
                
                # Log optimization results
                for symbol, results in optimization_results.items():
                    logger.info(f"🚀 Optimization Results for {symbol}: {results}")
                
                time.sleep(86400)  # Daily optimization
            
            except Exception as e:
                logger.error(f"Unified Optimization Job Error: {e}")
                time.sleep(3600)  # Wait 1 hour before retry
    
    def _popcat_research_job(self):
        """
        Popcat Research Agent's primary job
        Social sentiment and trend analysis
        """
        logger.info("🐱 Popcat Research Agent Activated")
        while True:
            try:
                # Analyze popcat trends
                popcat_insights = BTCPopCatResearchAgent(primary_symbol='BTC-USD', social_sentiment_sources=['twitter', 'reddit', 'telegram']).analyze_popcat_trends()
                logger.info(f"🌈 Popcat Trend Insights: {popcat_insights}")
                
                time.sleep(43200)  # Every 12 hours
            
            except Exception as e:
                logger.error(f"Popcat Research Job Error: {e}")
                time.sleep(1800)  # Wait 30 minutes before retry
    
    def _periodic_insight_aggregation(self):
        """
        Periodically aggregate and broadcast collaborative insights
        """
        logger.info("🤝 Collaborative Insights Aggregation Started")
        while True:
            try:
                # Generate and broadcast collaborative insights
                collaborative_insights = AgentCommunicationHub().generate_collaborative_insights()
                
                AgentCommunicationHub().broadcast_message(
                    sender='multi_agent_orchestrator',
                    message_type='collaborative_insights',
                    content=collaborative_insights
                )
                
                logger.info(f"🌟 Collaborative Insights: {collaborative_insights}")
                
                time.sleep(3600)  # Hourly aggregation
            
            except Exception as e:
                logger.error(f"Insight Aggregation Error: {e}")
                time.sleep(600)  # Wait 10 minutes before retry
    
    def start(self):
        """
        Start all agent jobs with their specific responsibilities
        """
        logger.info("🌟 Multi-Agent System Initializing")
        
        # Create and start threads for each agent's job
        agent_threads = [
            threading.Thread(target=self._market_intelligence_job, daemon=True),
            threading.Thread(target=self._optimization_job, daemon=True),
            threading.Thread(target=self._continuous_trading_job, daemon=True),
            threading.Thread(target=self._unified_optimization_job, daemon=True),
            threading.Thread(target=self._popcat_research_job, daemon=True),
            threading.Thread(target=self._periodic_insight_aggregation, daemon=True)
        ]
        
        for thread in agent_threads:
            thread.start()
        
        logger.info("🚀 All Agent Jobs Launched")
    
    def stop(self):
        """
        Gracefully stop all agent jobs
        """
        logger.info("🛑 Stopping Multi-Agent System")
        
        # for thread in agent_threads:
        #     thread.join(timeout=10)
        
        logger.info("✅ All Agent Jobs Stopped")

def launch_crypto_agents(): # Removed orchestrator_instance parameter
    """
    Launch crypto trading agents as a separate process.
    This function handles the full flow from analysis to execution.
    """
    try:
        logger.info(f"🚀 Launching Self-Contained Crypto Analysis & Execution Flow (Process ID: {os.getpid()})")
        
        # Create agent instances locally within this process
        crypto_agents = CryptoTradingAgents()
        trade_executor = TradeExecutionAgent()

        # Create dummy historical data (can be replaced with actual data fetching)
        data = {
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
            'open': [20000, 20100, 20050, 20200, 20150],
            'high': [20200, 20150, 20250, 20300, 20250],
            'low': [19900, 20000, 20000, 20100, 20050],
            'close': [20100, 20050, 20200, 20150, 20200],
            'volume': [1000, 1200, 1100, 1300, 1050]
        }
        historical_data = pd.DataFrame(data)
        historical_data.set_index('timestamp', inplace=True)
        
        # Step 1: Perform Market Analysis using local CryptoTradingAgents
        logger.info(f"Feeding historical data to local CryptoTradingAgents for analysis (Symbol: {crypto_agents.symbol}):\n{historical_data.head()}")
        analysis_result = crypto_agents.analyze_market(historical_data)
        
        logger.info("--- RAW MARKET ANALYSIS RESULT (within launch_crypto_agents) ---")
        logger.info(json.dumps(analysis_result, indent=2))
        print("--- START OF MARKET ANALYSIS RESULT (from launch_crypto_agents) ---")
        print(f"Symbol: {crypto_agents.symbol}")
        print(analysis_result)
        print("--- END OF MARKET ANALYSIS RESULT (from launch_crypto_agents) ---")

        # Step 2: Execute Trading Strategy using local TradeExecutionAgent
        logger.info("Proceeding to Trade Execution Agent (within launch_crypto_agents)...")
        trading_strategy_text = analysis_result.get('trading_strategy')

        if not trade_executor or not trade_executor.broker:
            logger.error("Local TradeExecutionAgent or its broker is not initialized. Skipping trade execution.")
            # This error will appear if RH_API_KEY/RH_PRIVATE_KEY are missing, which is fine.
        elif trading_strategy_text:
            logger.info("Trading strategy found. Creating execution task with local TradeExecutionAgent.")
            execution_task = trade_executor.execute_trade_task(trading_strategy_text)
            
            execution_crew = Crew(
                agents=[trade_executor.agent],
                tasks=[execution_task],
                verbose=True 
            )
            
            logger.info("Kicking off trade execution crew (within launch_crypto_agents)...")
            execution_result = execution_crew.kickoff()
            
            logger.info("--- TRADE EXECUTION RESULT (within launch_crypto_agents) ---")
            logger.info(execution_result)
        else:
            logger.warning("No trading strategy string found in analysis_result. Skipping trade execution (within launch_crypto_agents).")

        logger.info(f"✅ Self-Contained Crypto Analysis & Execution Flow completed (Process ID: {os.getpid()}).")
        
    except Exception as e:
        logger.exception(f"Error in launch_crypto_agents process (PID: {os.getpid()}): {e}")

def launch_popcat_research_agent():
    """
    Launch BTC PopCat Research Agent process
    """
    try:
        from crew_agents.src.btc_popcat_research_agent import main as popcat_research_main
        logger.info("🌐 Launching BTC PopCat Research Agent")
        popcat_research_main()
    except Exception as e:
        logger.error(f"Failed to launch BTC PopCat Research Agent: {e}")
        logger.error(traceback.format_exc())

def launch_all_agents_multiprocess():
    """
    Launch all agents using multiprocessing
    """
    agents = [
        launch_crypto_agents, # No longer needs orchestrator_instance
        # launch_popcat_research_agent, 
        # launch_unified_optimization_agent, 
    ]
    
    processes = []
    for agent_launcher_func in agents:
        p = multiprocessing.Process(target=agent_launcher_func)
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()

def main():
    """
    Launch and manage multi-agent crypto trading system
    """
    try:
        # orchestrator.start()
        launch_all_agents_multiprocess()
        
        # Keep main thread alive
        while True:
            time.sleep(3600)  # Check every hour
    
    except KeyboardInterrupt:
        logger.info("Received interrupt. Stopping agents...")
        # orchestrator.stop()
    
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        # orchestrator.stop()

if __name__ == "__main__":
    main()
