import os
import sys
import json
import time
import asyncio
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
import threading
from typing import Dict, List, Any

# Advanced machine learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'crew_agents', 'src'))

# Import local modules
from rl_environment import CryptoTradingEnv
from agent_communication_hub import communication_hub

class BTCPopCatResearchAgent:
    def __init__(
        self, 
        symbols=['BTC-USD', 'POPCAT-USD'],
        initial_capital=100000.0,
        research_interval_hours=1,
        primary_symbol: str = 'BTC-USD', 
        social_sentiment_sources: List[str] = ['twitter', 'reddit', 'telegram'],
        communication_hub=None
    ):
        """
        Initialize research agent for BTC and PopCat
        
        Args:
            symbols (list): Cryptocurrencies to research
            initial_capital (float): Initial trading capital
            research_interval_hours (int): Interval between research cycles
            primary_symbol (str): Primary cryptocurrency symbol
            social_sentiment_sources (List[str]): Sources for social sentiment tracking
            communication_hub (object): Communication hub for inter-agent messaging
        """
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.research_interval = research_interval_hours * 3600  # Convert to seconds
        self.primary_symbol = primary_symbol
        self.social_sentiment_sources = social_sentiment_sources
        self.communication_hub = communication_hub or communication_hub
        
        # Configure logging
        self.logger = logging.getLogger('BTCPopCatResearchAgent')
        self.logger.setLevel(logging.INFO)
        
        # Sentiment tracking
        self.sentiment_history = {
            source: [] for source in social_sentiment_sources
        }
        
        # Research configuration
        self.research_config = {
            'sentiment_window': 24,  # hours
            'sentiment_threshold': 0.5,
            'trend_detection_interval': 6  # hours
        }
        
        # Initialize data providers and models
        self.data_provider = None
        self.market_data = {}
        self.sentiment_data = {}
        self.trading_models = {}
        
        # Research and optimization agents
        self.market_research_agent = self._create_market_research_agent()
        self.trading_strategy_agent = self._create_trading_strategy_agent()
        self.model_optimization_agent = self._create_model_optimization_agent()
    
    def _create_market_research_agent(self):
        """
        Create market research agent for cryptocurrency analysis
        """
        return {
            'role': 'Crypto Market Researcher',
            'goal': 'Analyze market trends, sentiment, and potential trading opportunities',
            'backstory': 'Expert in cryptocurrency market analysis with deep understanding of technical and fundamental factors'
        }
    
    def _create_trading_strategy_agent(self):
        """
        Create trading strategy optimization agent
        """
        return {
            'role': 'Trading Strategy Optimizer',
            'goal': 'Design and refine trading strategies to maximize returns and minimize risk',
            'backstory': 'Advanced quantitative analyst specializing in algorithmic trading and risk management'
        }
    
    def _create_model_optimization_agent(self):
        """
        Create model optimization agent
        """
        return {
            'role': 'Model Optimization Specialist',
            'goal': 'Continuously improve and fine-tune machine learning models',
            'backstory': 'Expert in reinforcement learning and model performance optimization'
        }
    
    def _fetch_social_sentiment(self, source: str) -> Dict[str, float]:
        """
        Fetch social sentiment from a specific source
        
        Args:
            source (str): Social media platform
        
        Returns:
            Dict with sentiment metrics
        """
        # Placeholder for actual social sentiment API integration
        # In a real implementation, this would call Twitter, Reddit, etc. APIs
        mock_sentiment = {
            'twitter': {
                'overall_sentiment': 0.6,
                'volume': 1500,
                'sentiment_score': 0.7
            },
            'reddit': {
                'overall_sentiment': 0.4,
                'volume': 800,
                'sentiment_score': 0.5
            },
            'telegram': {
                'overall_sentiment': 0.55,
                'volume': 600,
                'sentiment_score': 0.6
            }
        }
        
        return mock_sentiment.get(source, {})
    
    def _analyze_social_trends(self) -> Dict[str, Any]:
        """
        Analyze social trends across different platforms
        
        Returns:
            Dict with trend analysis results
        """
        platform_sentiments = {}
        
        for source in self.social_sentiment_sources:
            sentiment_data = self._fetch_social_sentiment(source)
            platform_sentiments[source] = sentiment_data
            
            # Update sentiment history
            self.sentiment_history[source].append(sentiment_data)
            
            # Truncate history to maintain window size
            if len(self.sentiment_history[source]) > self.research_config['sentiment_window']:
                self.sentiment_history[source].pop(0)
        
        # Aggregate insights
        insights = {
            'overall_sentiment': sum(
                data.get('sentiment_score', 0) 
                for data in platform_sentiments.values()
            ) / len(platform_sentiments),
            'total_volume': sum(
                data.get('volume', 0) 
                for data in platform_sentiments.values()
            ),
            'platform_details': platform_sentiments
        }
        
        return insights
    
    def fetch_market_data(self):
        """
        Fetch and process market data for all symbols
        """
        for symbol in self.symbols:
            try:
                # Fetch historical price data
                historical_data = yf.download(symbol, period='1y', interval='1h')
                
                # Compute advanced features
                historical_data['log_returns'] = np.log(historical_data['Close'] / historical_data['Close'].shift(1))
                historical_data['volatility'] = historical_data['log_returns'].rolling(window=30).std()
                historical_data['trend_strength'] = historical_data['Close'].rolling(window=50).mean() / historical_data['Close'].rolling(window=200).mean()
                
                # Remove NaN values
                historical_data.dropna(inplace=True)
                
                self.market_data[symbol] = historical_data
                
                # Get market sentiment
                self.sentiment_data[symbol] = None
            
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
    
    def create_trading_environment(self, symbol):
        """
        Create a sophisticated trading environment for a given symbol
        """
        market_data = self.market_data[symbol]
        
        env = CryptoTradingEnv(
            market_data=market_data,
            initial_capital=self.initial_capital,
            risk_free_rate=0.02,  # US Treasury Bill rate
            transaction_cost_percentage=0.001,
            reward_scaling_factor=10.0
        )
        
        return DummyVecEnv([lambda: env])
    
    def train_reinforcement_learning_model(self, symbol):
        """
        Train advanced reinforcement learning model
        """
        env = self.create_trading_environment(symbol)
        
        model = PPO(
            "MlpPolicy", 
            env, 
            learning_rate=1e-3,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1
        )
        
        model.learn(total_timesteps=50000)
        
        return model
    
    def perform_market_research(self):
        """
        Conduct comprehensive market research
        """
        market_research_task = {
            'description': 'Analyze market trends and sentiment for BTC and PopCat, identifying potential trading opportunities',
            'agent': self.market_research_agent,
            'expected_output': 'Detailed market trend report with key insights and potential scenarios'
        }
        
        research_crew = {
            'agents': [self.market_research_agent],
            'tasks': [market_research_task],
            'verbose': 1
        }
        
        research_insights = research_crew
        return research_insights
    
    def optimize_trading_strategy(self, research_insights):
        """
        Optimize trading strategy based on market research
        """
        strategy_optimization_task = {
            'description': 'Design and refine trading strategy based on latest market research and performance metrics',
            'agent': self.trading_strategy_agent,
            'context': [research_insights],
            'expected_output': 'Optimized trading strategy with detailed implementation guidelines'
        }
        
        optimization_crew = {
            'agents': [self.trading_strategy_agent],
            'tasks': [strategy_optimization_task],
            'verbose': 1
        }
        
        strategy_insights = optimization_crew
        return strategy_insights
    
    def run_research(self):
        """
        Continuously run social sentiment research
        """
        self.logger.info(f"🐱 Popcat Research Agent Started for {self.primary_symbol}")
        
        while True:
            try:
                # Analyze social trends
                trend_insights = self._analyze_social_trends()
                
                # Broadcast insights via communication hub
                self.communication_hub.broadcast_message(
                    sender='btc_popcat_research_agent',
                    message_type='research_findings',
                    content={
                        'symbol': self.primary_symbol,
                        'insights': trend_insights,
                        'timestamp': time.time()
                    }
                )
                
                # Log insights
                self.logger.info(f"🌐 Social Sentiment Insights: {trend_insights}")
                
                # Sleep for configured trend detection interval
                time.sleep(
                    self.research_config['trend_detection_interval'] * 3600
                )
            
            except Exception as e:
                self.logger.error(f"Research Error: {e}")
                time.sleep(1200)  # Wait 20 minutes before retry
    
    def start(self):
        """
        Start the research agent in a separate thread
        """
        research_thread = threading.Thread(target=self.run_research, daemon=True)
        research_thread.start()
        return research_thread
    
    def run_continuous_research_cycle(self):
        """
        Execute continuous research and optimization cycle
        """
        # Fetch latest market data
        self.fetch_market_data()
        
        # Perform market research
        research_insights = self.perform_market_research()
        
        # Optimize trading strategy
        strategy_insights = self.optimize_trading_strategy(research_insights)
        
        # Train/update models for each symbol
        for symbol in self.symbols:
            try:
                # Train or update reinforcement learning model
                self.trading_models[symbol] = self.train_reinforcement_learning_model(symbol)
            except Exception as e:
                print(f"Error training model for {symbol}: {e}")
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'research_insights': research_insights,
            'strategy_insights': strategy_insights,
            'market_data': {symbol: data.to_dict() for symbol, data in self.market_data.items()},
            'sentiment_data': self.sentiment_data
        }
        
        results_dir = os.path.join(project_root, 'crew_agents', 'results', 'btc_popcat_research')
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f'research_cycle_{int(time.time())}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    async def continuous_research_loop(self):
        """
        Asynchronous continuous research loop
        """
        while True:
            print("Starting research and optimization cycle...")
            try:
                results = self.run_continuous_research_cycle()
                print("Research cycle completed successfully.")
                
                # Print key insights
                print("\n--- Research Insights ---")
                print(results.get('research_insights', 'No insights available'))
                
                print("\n--- Strategy Insights ---")
                print(results.get('strategy_insights', 'No strategy insights available'))
            
            except Exception as e:
                print(f"Error in research cycle: {e}")
            
            # Wait for next research cycle
            await asyncio.sleep(self.research_interval)

def main():
    """
    Main execution function
    """
    # Initialize research agent
    research_agent = BTCPopCatResearchAgent(
        symbols=['BTC-USD', 'POPCAT-USD'],
        initial_capital=100000.0,
        research_interval_hours=1
    )
    
    # Run continuous research loop
    asyncio.run(research_agent.continuous_research_loop())

if __name__ == "__main__":
    main()
