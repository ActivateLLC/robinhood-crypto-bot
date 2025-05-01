import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import ccxt
import requests
from typing import Dict, Any, List

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

class MarketIntelligenceAgent:
    """
    Advanced Market Intelligence Agent for Cryptocurrency Trading
    Provides comprehensive market analysis and risk assessment
    """
    def __init__(
        self, 
        symbols: List[str] = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        exchanges: List[str] = ['binance', 'kraken', 'coinbase']
    ):
        """
        Initialize Market Intelligence Agent
        
        Args:
            symbols (List[str]): Cryptocurrency trading pairs
            exchanges (List[str]): Exchanges to analyze
        """
        self.symbols = symbols
        self.exchanges = exchanges
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize exchange clients
        self.exchange_clients = self._initialize_exchanges()
    
    def _initialize_exchanges(self) -> Dict[str, Any]:
        """
        Initialize exchange clients with error handling
        
        Returns:
            Dict of exchange clients
        """
        clients = {}
        for exchange_id in self.exchanges:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                clients[exchange_id] = exchange_class({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot'
                    }
                })
            except Exception as e:
                self.logger.error(f"Failed to initialize {exchange_id} exchange: {e}")
        return clients
    
    def analyze_market_trends(self) -> Dict[str, Any]:
        """
        Comprehensive market trend analysis
        
        Returns:
            Dict of market insights
        """
        market_insights = {
            'timestamp': time.time(),
            'symbols': {},
            'global_sentiment': None,
            'risk_indicators': {}
        }
        
        for symbol in self.symbols:
            symbol_insights = self._analyze_symbol_trends(symbol)
            market_insights['symbols'][symbol] = symbol_insights
        
        # Aggregate global sentiment
        market_insights['global_sentiment'] = self._calculate_global_sentiment(market_insights)
        
        # Calculate cross-exchange risk indicators
        market_insights['risk_indicators'] = self._calculate_market_risk(market_insights)
        
        return market_insights
    
    def _analyze_symbol_trends(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze trends for a specific cryptocurrency
        
        Args:
            symbol (str): Trading pair to analyze
        
        Returns:
            Dict of symbol-specific insights
        """
        symbol_insights = {
            'price_trends': {},
            'volume_analysis': {},
            'volatility': {},
            'arbitrage_opportunities': []
        }
        
        for exchange_id, client in self.exchange_clients.items():
            try:
                # Fetch OHLCV data
                ohlcv = client.fetch_ohlcv(symbol, '1h', limit=24)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Calculate technical indicators
                df['returns'] = df['close'].pct_change()
                df['volatility'] = df['returns'].rolling(window=12).std()
                
                # Store insights
                symbol_insights['price_trends'][exchange_id] = {
                    'current_price': df['close'].iloc[-1],
                    'price_change_24h': ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
                }
                
                symbol_insights['volume_analysis'][exchange_id] = {
                    'total_volume': df['volume'].sum(),
                    'avg_volume': df['volume'].mean()
                }
                
                symbol_insights['volatility'][exchange_id] = df['volatility'].iloc[-1]
            
            except Exception as e:
                self.logger.error(f"Failed to analyze {symbol} on {exchange_id}: {e}")
        
        # Detect arbitrage opportunities
        symbol_insights['arbitrage_opportunities'] = self._detect_arbitrage(symbol)
        
        return symbol_insights
    
    def _calculate_global_sentiment(self, market_insights: Dict[str, Any]) -> str:
        """
        Calculate overall market sentiment
        
        Args:
            market_insights (Dict): Comprehensive market insights
        
        Returns:
            str: Market sentiment (Bullish/Bearish/Neutral)
        """
        try:
            # Fetch global crypto sentiment from alternative sources
            response = requests.get('https://api.alternative.me/fng/')
            if response.status_code == 200:
                fear_greed_index = response.json()['data'][0]['value']
                
                if fear_greed_index > 75:
                    return 'Extreme Greed'
                elif fear_greed_index > 50:
                    return 'Bullish'
                elif fear_greed_index > 25:
                    return 'Neutral'
                else:
                    return 'Bearish'
        except Exception as e:
            self.logger.error(f"Failed to fetch global sentiment: {e}")
        
        # Fallback sentiment calculation
        price_changes = [
            symbol_data['price_trends'][exchange]['price_change_24h']
            for symbol_data in market_insights['symbols'].values()
            for exchange in symbol_data['price_trends']
        ]
        
        avg_change = np.mean(price_changes)
        
        if avg_change > 5:
            return 'Bullish'
        elif avg_change < -5:
            return 'Bearish'
        else:
            return 'Neutral'
    
    def _calculate_market_risk(self, market_insights: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comprehensive market risk indicators
        
        Args:
            market_insights (Dict): Comprehensive market insights
        
        Returns:
            Dict of risk indicators
        """
        risk_indicators = {
            'market_volatility': 0,
            'correlation_risk': 0,
            'liquidity_risk': 0
        }
        
        # Calculate market volatility
        volatilities = [
            vol
            for symbol_data in market_insights['symbols'].values()
            for vol_dict in symbol_data['volatility'].values()
            for vol in [vol_dict]
        ]
        risk_indicators['market_volatility'] = np.mean(volatilities)
        
        # Calculate correlation risk
        returns = [
            symbol_data['price_trends'][exchange]['price_change_24h']
            for symbol_data in market_insights['symbols'].values()
            for exchange in symbol_data['price_trends']
        ]
        correlation_matrix = np.corrcoef(returns)
        risk_indicators['correlation_risk'] = np.mean(np.abs(correlation_matrix))
        
        # Calculate liquidity risk
        volumes = [
            vol
            for symbol_data in market_insights['symbols'].values()
            for vol_dict in symbol_data['volume_analysis'].values()
            for vol in [vol_dict['total_volume']]
        ]
        risk_indicators['liquidity_risk'] = 1 - (np.std(volumes) / np.mean(volumes))
        
        return risk_indicators
    
    def _detect_arbitrage(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Detect potential arbitrage opportunities
        
        Args:
            symbol (str): Trading pair to analyze
        
        Returns:
            List of arbitrage opportunities
        """
        arbitrage_opportunities = []
        
        # Compare prices across exchanges
        prices = {}
        for exchange_id, client in self.exchange_clients.items():
            try:
                ticker = client.fetch_ticker(symbol)
                prices[exchange_id] = ticker['last']
            except Exception as e:
                self.logger.error(f"Failed to fetch {symbol} price on {exchange_id}: {e}")
        
        # Find price differences
        exchanges = list(prices.keys())
        for i in range(len(exchanges)):
            for j in range(i+1, len(exchanges)):
                exchange1, exchange2 = exchanges[i], exchanges[j]
                price_diff_pct = abs(prices[exchange1] - prices[exchange2]) / min(prices[exchange1], prices[exchange2]) * 100
                
                if price_diff_pct > 1:  # More than 1% difference
                    arbitrage_opportunities.append({
                        'symbol': symbol,
                        'exchange1': exchange1,
                        'exchange2': exchange2,
                        'price1': prices[exchange1],
                        'price2': prices[exchange2],
                        'price_diff_pct': price_diff_pct
                    })
        
        return arbitrage_opportunities
    
    def get_comprehensive_market_insights(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive market insights for all symbols
        
        Returns:
            Dict of market insights for all tracked symbols
        """
        try:
            # Analyze market trends
            market_insights = self.analyze_market_trends()
            
            # Prepare insights for communication
            insights = {
                'timestamp': market_insights['timestamp'],
                'global_sentiment': market_insights['global_sentiment'],
                'risk_indicators': market_insights['risk_indicators'],
                'symbols': {}
            }
            
            # Process insights for each symbol
            for symbol, data in market_insights['symbols'].items():
                insights['symbols'][symbol] = {
                    'price_trends': data['price_trends'],
                    'volume_analysis': data['volume_analysis'],
                    'volatility': data['volatility'],
                    'arbitrage_opportunities': data['arbitrage_opportunities']
                }
            
            return insights
        
        except Exception as e:
            self.logger.error(f"Failed to get comprehensive market insights: {e}")
            return {
                'error': str(e),
                'timestamp': time.time()
            }

def main():
    """
    Demonstration of Market Intelligence Agent
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    market_agent = MarketIntelligenceAgent()
    market_insights = market_agent.analyze_market_trends()
    
    # Pretty print market insights
    import json
    print(json.dumps(market_insights, indent=2))

if __name__ == "__main__":
    main()
