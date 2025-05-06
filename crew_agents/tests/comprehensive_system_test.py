import sys
import os
import logging
import unittest
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from alt_crypto_data import AltCryptoDataProvider
from crew_agents.src.crypto_trading_env import CryptoTradingEnvironment
from crew_agents.src.error_monitoring_agent import AutoRemediationAgent, SystemReliabilityAuditor
from crew_agents.src.infinite_returns_agent import InfiniteReturnsAgent
from crew_agents.src.rl_model_tuner import RLModelTuner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveSystemTest(unittest.TestCase):
    def setUp(self):
        self.start_time = datetime.now()
        logger.info("Starting comprehensive system test...")

    def test_1_rl_environment(self):
        """Test Reinforcement Learning Trading Environment"""
        logger.info("Testing Crypto Trading Environment...")
        data_provider = AltCryptoDataProvider()
        env = CryptoTradingEnvironment(data_provider=data_provider)
        
        # Test environment reset
        initial_state = env.reset()
        self.assertIsNotNone(initial_state, "Environment reset failed")
        
        # Test step function
        action = env.action_space.sample()  # Random action
        next_state, reward, terminated, truncated, info = env.step(action)
        
        self.assertIsNotNone(next_state, "Environment step failed")
        self.assertTrue(isinstance(reward, float), "Reward must be a float")
        self.assertTrue(isinstance(terminated, bool), "Terminated flag must be a boolean")
        self.assertTrue(isinstance(truncated, bool), "Truncated flag must be a boolean")
        
        logger.info("Crypto Trading Environment test completed successfully")

    def test_2_error_handling(self):
        """Test Error Handling and Monitoring System"""
        logger.info("Testing Error Handling System...")
        remediation_agent = AutoRemediationAgent()
        
        # Simulate error scenarios
        test_errors = [
            {'error_type': 'API_TIMEOUT', 'message': 'Connection to trading API timed out'},
            {'error_type': 'INSUFFICIENT_FUNDS', 'message': 'Trading account has insufficient balance'}
        ]
        
        for error in test_errors:
            result = remediation_agent.issue_manager.delegate_issue_solving(error)
            self.assertIsNotNone(result, f"Failed to handle {error['error_type']} error")
        
        logger.info("Error Handling System test completed successfully")

    def test_3_reward_methodology(self):
        """Test Reward Calculation Methods"""
        logger.info("Testing Reward Methodology...")
        agent = InfiniteReturnsAgent()
        
        # Test scenarios with different market conditions
        test_scenarios = [
            {'realized_pnl': 1000, 'transaction_costs': 50, 'current_drawdown': 0.1},
            {'realized_pnl': -500, 'transaction_costs': 20, 'current_drawdown': 0.3}
        ]
        
        for scenario in test_scenarios:
            reward = agent.calculate_hybrid_reward(**scenario)
            self.assertTrue(isinstance(reward, float), "Reward must be a float")
        
        logger.info("Reward Methodology test completed successfully")

    def test_4_system_reliability_audit(self):
        """Test System Reliability Audit"""
        logger.info("Conducting System Reliability Audit...")
        auditor = SystemReliabilityAuditor()
        
        audit_results = auditor.conduct_comprehensive_audit()
        
        self.assertIsNotNone(audit_results, "Audit failed to generate results")
        self.assertTrue('agent_insights' in audit_results, "Audit results missing agent insights")
        
        logger.info("System Reliability Audit completed successfully")

    def tearDown(self):
        duration = datetime.now() - self.start_time
        logger.info(f"Comprehensive system test completed in {duration}")

if __name__ == '__main__':
    unittest.main()
