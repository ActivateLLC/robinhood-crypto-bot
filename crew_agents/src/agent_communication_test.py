import sys
import os
import logging

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from crew_agents.src.crew_optimization_manager import CrewOptimizationAgent
from crew_agents.src.continuous_optimization_agent import ContinuousOptimizationAgent
from crew_agents.src.market_intelligence_agent import MarketIntelligenceAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_agent_communication():
    """
    Demonstrate inter-agent communication and collaboration
    """
    logger.info("🚀 Initializing Agent Communication Test")
    
    # Initialize agents
    crew_agent = CrewOptimizationAgent(symbol='BTC-USD')
    continuous_agent = ContinuousOptimizationAgent()
    market_intel_agent = MarketIntelligenceAgent(symbols=['BTC-USD', 'ETH-USD'])
    
    # Simulate a communication scenario
    logger.info("📡 Simulating Market Intelligence Insights")
    market_insights = market_intel_agent.get_comprehensive_market_insights()
    
    # Broadcast market insights
    continuous_agent.communication_hub.broadcast_message(
        sender='market_intelligence_agent',
        message={
            'type': 'market_insight',
            'insights': market_insights
        }
    )
    
    # Run optimization pipeline
    logger.info("🔍 Running Crew Optimization Pipeline")
    optimization_results = crew_agent.run_optimization_pipeline()
    
    # Generate collaborative insights
    logger.info("🤝 Generating Collaborative Insights")
    collaborative_insights = crew_agent.communication_hub.generate_collaborative_insights()
    
    # Log results
    logger.info("\n🌟 Agent Communication Test Results 🌟")
    logger.info(f"Market Insights: {market_insights}")
    logger.info(f"Optimization Results: {optimization_results}")
    logger.info(f"Collaborative Insights: {collaborative_insights}")

def main():
    test_agent_communication()

if __name__ == "__main__":
    main()
