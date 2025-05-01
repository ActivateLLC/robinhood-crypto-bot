import os
import sys
import json
import numpy as np
import pandas as pd

# Dynamically add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'crew_agents', 'src'))

try:
    from trading_intelligence import TradingIntelligenceEngine
    from alt_crypto_data import AltCryptoDataProvider
    from rl_environment import CryptoTradingEnv
except ImportError as e:
    print(f"Import Error: {e}")
    print("Sys Path:", sys.path)
    raise

class RLIntelligenceBridge:
    def __init__(self, symbol='BTC-USD', data_period='1y', data_interval='1h'):
        """
        Bridge between CrewAI intelligence and Reinforcement Learning environment
        
        Args:
            symbol (str): Cryptocurrency trading pair
            data_period (str): Historical data period
            data_interval (str): Data granularity
        """
        try:
            # Initialize data provider
            self.data_provider = AltCryptoDataProvider()
            
            # Fetch historical data
            self.historical_data = self.data_provider.fetch_price_history(
                symbol=symbol, 
                days=365,  # 1 year of data
                interval=data_interval
            )
            
            # Create CrewAI trading intelligence engine
            self.trading_intelligence = TradingIntelligenceEngine(
                symbol=symbol, 
                data_provider=self.data_provider
            )
            
            # Store symbol and configuration
            self.symbol = symbol
            self.data_period = data_period
            self.data_interval = data_interval
        
        except Exception as e:
            print(f"Initialization Error: {e}")
            raise
    
    def generate_enhanced_environment(self):
        """
        Generate an enhanced RL environment with CrewAI-powered insights
        
        Returns:
            CryptoTradingEnv: Enhanced trading environment
        """
        try:
            # Generate comprehensive trading insights
            insights = self.trading_intelligence.generate_comprehensive_trading_insights(
                self.historical_data
            )
            
            # Extract reward function design
            reward_design = self.trading_intelligence.extract_reward_function_design(insights)
            
            # Generate trading strategy recommendations
            strategies = self.trading_intelligence.generate_trading_strategy_recommendations(insights)
            
            # Create enhanced environment
            enhanced_env = CryptoTradingEnv(
                symbol=self.symbol,
                historical_data=self.historical_data,
                custom_reward_params=reward_design
            )
            
            # Inject CrewAI insights into environment
            enhanced_env.crew_insights = {
                'market_insights': insights,
                'reward_design': reward_design,
                'trading_strategies': strategies
            }
            
            return enhanced_env
        
        except Exception as e:
            print(f"Environment Generation Error: {e}")
            raise
    
    def export_intelligence_report(self, env):
        """
        Export comprehensive intelligence report
        
        Args:
            env (CryptoTradingEnv): Enhanced trading environment
        
        Returns:
            str: Path to exported report
        """
        try:
            # Prepare intelligence report
            intelligence_report = {
                'symbol': self.symbol,
                'data_period': self.data_period,
                'data_interval': self.data_interval,
                'crew_insights': env.crew_insights,
                'environment_metadata': {
                    'observation_space': str(env.observation_space),
                    'action_space': str(env.action_space),
                    'max_episode_steps': env.max_episode_steps
                }
            }
            
            # Define output path
            output_dir = os.path.join(project_root, 'crew_agents', 'outputs')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{self.symbol}_intelligence_report.json')
            
            # Export report
            with open(output_path, 'w') as f:
                json.dump(intelligence_report, f, indent=2, default=str)
            
            return output_path
        
        except Exception as e:
            print(f"Report Export Error: {e}")
            raise

def main():
    """
    Main execution for RL Intelligence Bridge
    """
    try:
        # Initialize bridge
        intelligence_bridge = RLIntelligenceBridge(symbol='BTC-USD')
        
        # Generate enhanced environment
        enhanced_env = intelligence_bridge.generate_enhanced_environment()
        
        # Export intelligence report
        report_path = intelligence_bridge.export_intelligence_report(enhanced_env)
        
        print(f"Intelligence report exported to: {report_path}")
        print("Enhanced RL environment created successfully!")
    
    except Exception as e:
        print(f"Main Execution Error: {e}")
        raise

if __name__ == "__main__":
    main()
