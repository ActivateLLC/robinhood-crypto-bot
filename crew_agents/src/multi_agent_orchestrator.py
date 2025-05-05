import os
import sys
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import traceback

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import all agents and communication hub
from crew_agents.src.market_intelligence_agent import MarketIntelligenceAgent
from crew_agents.src.crew_optimization_manager import CrewOptimizationAgent
from crew_agents.src.continuous_optimization_agent import ContinuousOptimizationAgent
from crew_agents.src.unified_optimization_agent import UnifiedOptimizationAgent
from crew_agents.src.btc_popcat_research_agent import BTCPopCatResearchAgent
from crew_agents.src.agent_communication_hub import communication_hub

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
    def __init__(
        self, 
        symbols=['BTC-USD', 'ETH-USD', 'BNB-USD']
    ):
        """
        Initialize and coordinate multiple crypto trading agents
        
        Args:
            symbols (List[str]): Cryptocurrency symbols to track
        """
        self.symbols = symbols
        
        # Initialize communication hub with project goals
        communication_hub.update_project_context({
            'primary_objectives': [
                'Optimize cryptocurrency trading strategies',
                'Develop adaptive reinforcement learning models',
                'Minimize risk and maximize returns'
            ],
            'target_cryptocurrencies': symbols,
            'risk_tolerance': 0.2,
            'investment_horizon': 'Long-term',
            'rebalancing_strategy': 'Quarterly'
        })
        
        # Initialize agents with their specific configurations
        self.market_intelligence_agent = MarketIntelligenceAgent(
            symbols=symbols, 
            exchanges=['binance', 'coinbase', 'kraken']
        )
        
        self.crew_optimization_agent = CrewOptimizationAgent(
            symbol='BTC-USD', 
            initial_capital=100000
        )
        
        self.continuous_optimization_agent = ContinuousOptimizationAgent(
            symbols=symbols,
            optimization_interval=3600,
            initial_capital=100000
        )
        
        self.unified_optimization_agent = UnifiedOptimizationAgent(
            symbols=symbols
        )
        
        self.btc_popcat_research_agent = BTCPopCatResearchAgent(
            primary_symbol='BTC-USD',
            social_sentiment_sources=['twitter', 'reddit', 'telegram']
        )
        
        # Tracking flags
        self.is_running = False
        self.agent_threads = []
    
    def _market_intelligence_job(self):
        """
        Market Intelligence Agent's primary job
        Continuously gather and analyze market data
        """
        logger.info("🌐 Market Intelligence Agent Activated")
        while self.is_running:
            try:
                # Comprehensive market trend analysis
                market_insights = self.market_intelligence_agent.analyze_market_trends()
                
                # Detect arbitrage opportunities
                for symbol in self.symbols:
                    arbitrage_ops = self.market_intelligence_agent._detect_arbitrage(symbol)
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
        while self.is_running:
            try:
                # Run optimization pipeline for each symbol
                for symbol in self.symbols:
                    optimization_results = self.crew_optimization_agent.run_optimization_pipeline(symbol)
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
        while self.is_running:
            try:
                # Run continuous optimization
                self.continuous_optimization_agent.run_continuous_optimization()
                
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
        while self.is_running:
            try:
                # Run unified optimization pipeline
                optimization_results = self.unified_optimization_agent.run_optimization_pipeline()
                
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
        while self.is_running:
            try:
                # Analyze popcat trends
                popcat_insights = self.btc_popcat_research_agent.analyze_popcat_trends()
                logger.info(f"🌈 Popcat Trend Insights: {popcat_insights}")
                
                time.sleep(43200)  # Every 12 hours
            
            except Exception as e:
                logger.error(f"Popcat Research Job Error: {e}")
                time.sleep(1800)  # Wait 30 minutes before retry
    
    def _data_provider_job(self):
        """
        Alt Crypto Data Provider's primary job
        Continuous data collection and preprocessing
        """
        logger.info("📡 Alt Crypto Data Provider Activated")
        while self.is_running:
            try:
                # Fetch and preprocess crypto data
                for symbol in self.symbols:
                    crypto_data = self.alt_crypto_data_provider.fetch_crypto_data(symbol)
                    logger.info(f"📊 Data Fetched for {symbol}")
                
                time.sleep(7200)  # Every 2 hours
            
            except Exception as e:
                logger.error(f"Data Provider Job Error: {e}")
                time.sleep(1200)  # Wait 20 minutes before retry
    
    def _periodic_insight_aggregation(self):
        """
        Periodically aggregate and broadcast collaborative insights
        """
        logger.info("🤝 Collaborative Insights Aggregation Started")
        while self.is_running:
            try:
                # Generate and broadcast collaborative insights
                collaborative_insights = communication_hub.generate_collaborative_insights()
                
                communication_hub.broadcast_message(
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
        self.is_running = True
        
        # Create and start threads for each agent's job
        self.agent_threads = [
            threading.Thread(target=self._market_intelligence_job, daemon=True),
            threading.Thread(target=self._optimization_job, daemon=True),
            threading.Thread(target=self._continuous_trading_job, daemon=True),
            threading.Thread(target=self._unified_optimization_job, daemon=True),
            threading.Thread(target=self._popcat_research_job, daemon=True),
            threading.Thread(target=self._periodic_insight_aggregation, daemon=True)
        ]
        
        for thread in self.agent_threads:
            thread.start()
        
        logger.info("🚀 All Agent Jobs Launched")
    
    def stop(self):
        """
        Gracefully stop all agent jobs
        """
        logger.info("🛑 Stopping Multi-Agent System")
        self.is_running = False
        
        for thread in self.agent_threads:
            thread.join(timeout=10)
        
        logger.info("✅ All Agent Jobs Stopped")

def launch_crypto_agents():
    """
    Launch crypto trading agents as a separate process
    """
    try:
        from crew_agents.src.crypto_agents import CryptoTradingAgents
        logger.info("🚀 Launching Crypto Trading Agents")
        crypto_agents = CryptoTradingAgents()
        # Add any specific initialization or method calls here
    except Exception as e:
        logger.error(f"Failed to launch Crypto Trading Agents: {e}")
        logger.error(traceback.format_exc())

def launch_crew_optimization():
    """
    Launch crew optimization process
    """
    try:
        from crew_agents.src.crew_optimization_manager import main as crew_optimization_main
        logger.info("🔧 Launching Crew Optimization")
        crew_optimization_main()
    except Exception as e:
        logger.error(f"Failed to launch Crew Optimization: {e}")
        logger.error(traceback.format_exc())

def launch_btc_roi_optimizer():
    """
    Launch BTC ROI Optimizer process
    """
    try:
        from crew_agents.src.btc_roi_optimizer import main as roi_optimizer_main
        logger.info("📊 Launching BTC ROI Optimizer")
        roi_optimizer_main()
    except Exception as e:
        logger.error(f"Failed to launch BTC ROI Optimizer: {e}")
        logger.error(traceback.format_exc())

def launch_btc_popcat_research():
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
        launch_crypto_agents,
        launch_crew_optimization,
        launch_btc_roi_optimizer,
        launch_btc_popcat_research
    ]
    
    processes = []
    for agent_launcher in agents:
        p = multiprocessing.Process(target=agent_launcher)
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()

def main():
    """
    Launch and manage multi-agent crypto trading system
    """
    orchestrator = MultiAgentOrchestrator()
    
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
