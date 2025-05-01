import os
import sys
import json
import logging
from typing import Dict, Any, List

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

class AgentCommunicationHub:
    """
    Centralized communication hub for coordinating and synchronizing agent activities
    """
    def __init__(self):
        """
        Initialize the communication hub with logging and message tracking
        """
        self.logger = logging.getLogger('AgentCommunicationHub')
        self.logger.setLevel(logging.INFO)
        
        # Message queues for different types of communications
        self.message_queues = {
            'project_goals': [],
            'market_insights': [],
            'optimization_results': [],
            'research_findings': [],
            'risk_assessments': []
        }
        
        # Global project context
        self.project_context = {
            'primary_objectives': [
                'Optimize cryptocurrency trading strategies',
                'Minimize risk and maximize returns',
                'Develop adaptive reinforcement learning models'
            ],
            'target_cryptocurrencies': ['BTC-USD', 'ETH-USD', 'BNB-USD'],
            'risk_tolerance': 0.2,
            'investment_horizon': 'Long-term',
            'rebalancing_strategy': 'Quarterly'
        }
    
    def broadcast_message(self, sender: str, message_type: str, content: Dict[str, Any]):
        """
        Broadcast a message to all relevant agents
        
        Args:
            sender (str): Name of the sending agent
            message_type (str): Type of message being sent
            content (Dict): Message content
        """
        full_message = {
            'timestamp': os.path.getmtime(__file__),
            'sender': sender,
            'type': message_type,
            'content': content
        }
        
        # Log the message
        self.logger.info(f"🌐 Message Broadcast: {sender} - {message_type}")
        
        # Store in appropriate message queue
        if message_type in self.message_queues:
            self.message_queues[message_type].append(full_message)
        
        # Truncate message queues to prevent memory overflow
        for queue in self.message_queues.values():
            if len(queue) > 100:
                queue.pop(0)
    
    def get_latest_messages(self, message_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve latest messages of a specific type
        
        Args:
            message_type (str): Type of messages to retrieve
            limit (int): Number of messages to return
        
        Returns:
            List of recent messages
        """
        if message_type not in self.message_queues:
            self.logger.warning(f"Unknown message type: {message_type}")
            return []
        
        return self.message_queues[message_type][-limit:]
    
    def update_project_context(self, updates: Dict[str, Any]):
        """
        Update the global project context
        
        Args:
            updates (Dict): Context updates to apply
        """
        self.project_context.update(updates)
        
        # Broadcast project context update
        self.broadcast_message(
            sender='communication_hub',
            message_type='project_goals',
            content={
                'updated_context': updates,
                'full_context': self.project_context
            }
        )
        
        self.logger.info(f"🔄 Project Context Updated: {updates}")
    
    def generate_collaborative_insights(self) -> Dict[str, Any]:
        """
        Generate collaborative insights by aggregating messages
        
        Returns:
            Dict of aggregated insights
        """
        insights = {
            'market_sentiment': self._analyze_market_sentiment(),
            'risk_aggregation': self._aggregate_risk_assessments(),
            'optimization_trends': self._identify_optimization_patterns()
        }
        
        return insights
    
    def _analyze_market_sentiment(self) -> str:
        """
        Analyze market sentiment from recent market insights
        
        Returns:
            Aggregated market sentiment
        """
        market_insights = self.get_latest_messages('market_insights')
        
        if not market_insights:
            return 'Neutral'
        
        # Simple sentiment analysis based on recent insights
        sentiments = [
            insight['content'].get('sentiment', 'Neutral')
            for insight in market_insights
        ]
        
        # Most common sentiment
        return max(set(sentiments), key=sentiments.count)
    
    def _aggregate_risk_assessments(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregate risk assessments from different agents
        
        Returns:
            Aggregated risk indicators
        """
        risk_assessments = self.get_latest_messages('risk_assessments')
        
        aggregated_risks = {
            'volatility': {'mean': 0, 'max': 0, 'min': 0},
            'max_drawdown': {'mean': 0, 'max': 0, 'min': 0},
            'correlation_risk': {'mean': 0, 'max': 0, 'min': 0}
        }
        
        if not risk_assessments:
            return aggregated_risks
        
        # Aggregate risk metrics
        for metric in aggregated_risks.keys():
            values = [
                assessment['content'].get(metric, 0)
                for assessment in risk_assessments
            ]
            
            aggregated_risks[metric] = {
                'mean': sum(values) / len(values),
                'max': max(values),
                'min': min(values)
            }
        
        return aggregated_risks
    
    def _identify_optimization_patterns(self) -> Dict[str, Dict]:
        """
        Identify optimization trends and patterns
        
        Returns:
            Dict of optimization insights
        """
        optimization_results = self.get_latest_messages('optimization_results')
        
        return {
            'most_successful_strategies': {},
            'emerging_patterns': {}
        }

# Global communication hub instance
communication_hub = AgentCommunicationHub()
