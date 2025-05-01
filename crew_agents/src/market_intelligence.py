import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from crew_agents.src.crypto_agents import CryptoTradingAgents
from alt_crypto_data import AltCryptoDataProvider

class MarketIntelligenceEngine:
    def __init__(self, symbol='BTC-USD', data_period='1y', data_interval='1h'):
        """
        Initialize Market Intelligence Engine
        
        Args:
            symbol (str): Cryptocurrency trading pair
            data_period (str): Historical data period
            data_interval (str): Data granularity
        """
        # Initialize data provider
        self.data_provider = AltCryptoDataProvider()
        
        # Fetch historical data
        self.historical_data = self.data_provider.fetch_price_history(
            symbol=symbol, 
            days=365,  # 1 year of data
            interval=data_interval
        )
        
        # Create CrewAI agents
        self.trading_agents = CryptoTradingAgents(
            symbol=symbol, 
            data_provider=self.data_provider
        )
    
    def generate_market_insights(self):
        """
        Generate comprehensive market insights using multi-agent system
        
        Returns:
            dict: Detailed market analysis and trading insights
        """
        # Perform market analysis
        market_analysis = self.trading_agents.analyze_market(self.historical_data)
        
        # Additional data enrichment
        market_analysis['historical_data_summary'] = {
            'total_periods': len(self.historical_data),
            'date_range': {
                'start': self.historical_data.index[0],
                'end': self.historical_data.index[-1]
            },
            'price_stats': {
                'mean': self.historical_data['close'].mean(),
                'median': self.historical_data['close'].median(),
                'std_dev': self.historical_data['close'].std(),
                'min': self.historical_data['close'].min(),
                'max': self.historical_data['close'].max()
            }
        }
        
        return market_analysis
    
    def export_insights(self, insights, output_path=None):
        """
        Export market insights to a file
        
        Args:
            insights (dict): Market analysis insights
            output_path (str, optional): Path to export insights
        
        Returns:
            str: Path to exported insights file
        """
        if output_path is None:
            output_path = os.path.join(
                project_root, 
                'crew_agents', 
                'outputs', 
                f'market_insights_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export insights
        with open(output_path, 'w') as f:
            import json
            json.dump(insights, f, indent=2, default=str)
        
        return output_path

def main():
    """
    Main execution for Market Intelligence Engine
    """
    # Initialize engine
    market_engine = MarketIntelligenceEngine()
    
    # Generate market insights
    insights = market_engine.generate_market_insights()
    
    # Export insights
    output_file = market_engine.export_insights(insights)
    print(f"Market insights exported to: {output_file}")

if __name__ == "__main__":
    main()
