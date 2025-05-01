#!/usr/bin/env python3
import sys
import os
import logging
import json
import numpy as np
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import optimization manager
from crew_agents.src.crew_optimization_manager import CrewOptimizationAgent

def custom_json_serializer(obj):
    """
    Advanced JSON serializer for complex objects
    """
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, 'to_json'):
        return obj.to_json()
    elif isinstance(obj, (list, dict)):
        return obj
    elif isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return str(obj)

def setup_logging():
    """Configure comprehensive logging"""
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'optimization_run_{timestamp}.log')
    
    # Configure logging with more detailed format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def analyze_performance(results):
    """
    Perform comprehensive performance analysis
    
    Args:
        results (dict): Optimization results for multiple cryptocurrencies
    
    Returns:
        dict: Detailed performance metrics
    """
    performance_summary = {
        'total_cryptocurrencies': len(results),
        'overall_performance': {
            'total_roi': 0,
            'average_sharpe_ratio': 0,
            'max_roi': float('-inf'),
            'min_roi': float('inf')
        },
        'detailed_crypto_performance': {}
    }
    
    for symbol, data in results.items():
        # Safely extract portfolio strategy and risk profile
        portfolio_strategy = data.get('portfolio_strategy', {}) if isinstance(data, dict) else {}
        risk_profile = data.get('risk_profile', {}) if isinstance(data, dict) else {}
        
        # Individual cryptocurrency performance
        crypto_performance = {
            'roi': portfolio_strategy.get('cumulative_roi', 0) if isinstance(portfolio_strategy, dict) else 0,
            'sharpe_ratio': portfolio_strategy.get('sharpe_ratio', 0) if isinstance(portfolio_strategy, dict) else 0,
            'allocation_strategy': portfolio_strategy.get('allocation_strategy', {}) if isinstance(portfolio_strategy, dict) else {},
            'risk_metrics': {
                'volatility': risk_profile.get('volatility', 0) if isinstance(risk_profile, dict) else 0,
                'max_drawdown': risk_profile.get('max_drawdown', 0) if isinstance(risk_profile, dict) else 0
            }
        }
        
        performance_summary['detailed_crypto_performance'][symbol] = crypto_performance
        
        # Update overall performance metrics
        roi = crypto_performance['roi']
        performance_summary['overall_performance']['total_roi'] += roi
        performance_summary['overall_performance']['max_roi'] = max(
            performance_summary['overall_performance']['max_roi'], roi
        )
        performance_summary['overall_performance']['min_roi'] = min(
            performance_summary['overall_performance']['min_roi'], roi
        )
    
    # Calculate average Sharpe ratio
    performance_summary['overall_performance']['average_sharpe_ratio'] = np.mean([
        data.get('portfolio_strategy', {}).get('sharpe_ratio', 0) 
        for data in results.values() if isinstance(data, dict)
    ])
    
    return performance_summary

def run_optimization():
    """
    Run comprehensive cryptocurrency trading optimization
    """
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Cryptocurrencies to optimize
        cryptocurrencies = ['BTC-USD', 'ETH-USD', 'BNB-USD']
        
        # Store results for each cryptocurrency
        all_results = {}
        
        for symbol in cryptocurrencies:
            logger.info(f"Starting optimization for {symbol}")
            
            # Initialize crew optimization agent
            crew_agent = CrewOptimizationAgent(
                symbol=symbol, 
                initial_capital=100000  # $100,000 starting capital
            )
            
            # Run optimization pipeline
            optimization_results = crew_agent.run_optimization_pipeline()
            
            # Convert complex objects to basic types
            serializable_results = {
                key: custom_json_serializer(value) 
                for key, value in optimization_results.items()
            }
            
            # Store results
            all_results[symbol] = serializable_results
            
            # Log key performance metrics
            portfolio_strategy = serializable_results.get('portfolio_strategy', {})
            if isinstance(portfolio_strategy, dict):
                logger.info(f"Optimization Results for {symbol}:")
                logger.info(f"Cumulative ROI: {portfolio_strategy.get('cumulative_roi', 'N/A')}")
                logger.info(f"Sharpe Ratio: {portfolio_strategy.get('sharpe_ratio', 'N/A')}")
        
        # Perform comprehensive performance analysis
        performance_summary = analyze_performance(all_results)
        
        # Log performance summary
        logger.info("\n--- COMPREHENSIVE OPTIMIZATION SUMMARY ---")
        logger.info(f"Total Cryptocurrencies Analyzed: {performance_summary['total_cryptocurrencies']}")
        logger.info("Overall Performance:")
        logger.info(f"  Total ROI: {performance_summary['overall_performance']['total_roi']:.2%}")
        logger.info(f"  Average Sharpe Ratio: {performance_summary['overall_performance']['average_sharpe_ratio']:.2f}")
        logger.info(f"  Max ROI: {performance_summary['overall_performance']['max_roi']:.2%}")
        logger.info(f"  Min ROI: {performance_summary['overall_performance']['min_roi']:.2%}")
        
        logger.info("\nDetailed Cryptocurrency Performance:")
        for symbol, perf in performance_summary['detailed_crypto_performance'].items():
            logger.info(f"\n{symbol} Performance:")
            logger.info(f"  ROI: {perf['roi']:.2%}")
            logger.info(f"  Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
            logger.info(f"  Allocation Strategy: {perf['allocation_strategy']}")
            logger.info(f"  Risk Metrics:")
            logger.info(f"    Volatility: {perf['risk_metrics']['volatility']:.2%}")
            logger.info(f"    Max Drawdown: {perf['risk_metrics']['max_drawdown']:.2%}")
        
        # Export results
        results_file = os.path.join(project_root, 'logs', 'optimization_results.json')
        performance_file = os.path.join(project_root, 'logs', 'performance_summary.json')
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4, default=custom_json_serializer)
        
        with open(performance_file, 'w') as f:
            json.dump(performance_summary, f, indent=4, default=custom_json_serializer)
        
        logger.info(f"\nDetailed results saved to {results_file}")
        logger.info(f"Performance summary saved to {performance_file}")
        
        return all_results
    
    except Exception as e:
        logger.error(f"Optimization process failed: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    results = run_optimization()
