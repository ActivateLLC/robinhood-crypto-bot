import os
import sys
import time
import logging
import threading
import subprocess
from typing import List, Dict, Optional

# Ensure project root is in Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', 'agent_launcher.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import custom modules
from crew_agents.src.rl_model_tuner import RLModelTuner
from crew_agents.src.error_monitoring_agent import start_auto_remediation_agent

class ROIOptimizationWorkflow:
    """
    Advanced ROI Optimization Workflow for Reinforcement Learning Models
    """
    def __init__(self, symbols: List[str] = ['BTC-USD', 'ETH-USD', 'BNB-USD']):
        self.symbols = symbols
        self.roi_history = {symbol: [] for symbol in symbols}
        self.optimization_params = {
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [32, 64, 128],
            'gamma': [0.9, 0.95, 0.99],
            'exploration_rate': [0.1, 0.2, 0.3]
        }
    
    def adaptive_roi_tuning(self, current_roi: float, symbol: str):
        """
        Dynamically adjust RL model hyperparameters based on ROI
        
        Args:
            current_roi (float): Current Return on Investment
            symbol (str): Cryptocurrency symbol
        """
        self.roi_history[symbol].append(current_roi)
        
        # Adaptive parameter adjustment
        if len(self.roi_history[symbol]) > 5:
            roi_trend = self.roi_history[symbol][-5:]
            roi_volatility = max(roi_trend) - min(roi_trend)
            
            if roi_volatility < 0.05:  # Low volatility
                # Increase exploration to find better strategies
                return {
                    'exploration_rate': min(self.optimization_params['exploration_rate']) * 1.5,
                    'learning_rate': min(self.optimization_params['learning_rate']) * 0.8
                }
            elif current_roi < 0:  # Negative ROI
                # More aggressive learning and exploration
                return {
                    'learning_rate': max(self.optimization_params['learning_rate']),
                    'batch_size': max(self.optimization_params['batch_size']),
                    'exploration_rate': max(self.optimization_params['exploration_rate'])
                }
        
        return {}

class AgentLauncher:
    """
    Advanced Agent Launching and Coordination System
    """
    def __init__(self):
        # Initialize RLModelTuner
        self.rl_model_tuner = RLModelTuner()
        
        # Initialize ROIOptimizationWorkflow
        self.roi_workflow = ROIOptimizationWorkflow()
        
        # Agent configuration
        self.agents = {
            'market_intelligence': {
                'module': 'crew_agents.src.market_intelligence_agent',
                'log_file': 'market_intelligence.log',
                'required': True
            },
            'continuous_optimization': {
                'module': 'crew_agents.src.continuous_optimization_agent',
                'log_file': 'continuous_optimization.log',
                'required': True
            },
            'logging_management': {
                'module': 'crew_agents.src.logging_management_agent',
                'log_file': 'logging_management.log',
                'required': True
            },
            'rl_model_tuner': {
                'module': 'crew_agents.src.rl_model_tuner',
                'log_file': 'rl_model_tuner.log',
                'required': True
            },
            'crypto_sentiment': {
                'module': 'crypto_sentiment',
                'log_file': 'crypto_sentiment.log',
                'required': False
            },
            'web_research_agent': {
                'module': 'error_monitoring_agent.WebResearchAgent',
                'log_file': 'web_research.log',
                'required': True
            }
        }
        
        # Ensure logs directory exists
        os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
    
    def _launch_agent(self, agent_name: str, agent_config: Dict) -> Optional[subprocess.Popen]:
        """
        Launch a single agent as a subprocess
        
        Args:
            agent_name (str): Name of the agent
            agent_config (Dict): Configuration for the agent
        
        Returns:
            subprocess.Popen or None: Process for the launched agent
        """
        log_file_path = os.path.join(project_root, 'logs', agent_config['log_file'])
        
        # Construct command to run agent
        command = [
            sys.executable, 
            '-m', 
            agent_config['module']
        ]
        
        # Open log file for writing
        log_file = open(log_file_path, 'w')
        
        try:
            process = subprocess.Popen(
                command, 
                stdout=log_file, 
                stderr=subprocess.STDOUT,
                cwd=project_root
            )
            
            logger.info(f"Launched {agent_name} agent (PID: {process.pid})")
            return process
        except Exception as e:
            logger.error(f"Failed to launch {agent_name} agent: {e}")
            log_file.close()
            return None
    
    def launch_all_agents(self):
        """
        Launch all configured agents simultaneously using threads
        """
        def threaded_launch(agent_name, agent_config):
            """
            Thread-safe agent launcher
            """
            self._launch_agent(agent_name, agent_config)
        
        # Create and start a thread for each agent
        threads = []
        for agent_name, agent_config in self.agents.items():
            if agent_config.get('required', False):
                thread = threading.Thread(
                    target=threaded_launch, 
                    args=(agent_name, agent_config)
                )
                thread.start()
                threads.append(thread)
                time.sleep(1)  # Small delay to prevent potential race conditions
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        logger.info("All required agents launched simultaneously!")
    
    def run_model_optimization(self):
        """
        Run continuous RL model optimization
        """
        while True:
            try:
                # Run optimization for each symbol
                for symbol in self.roi_workflow.symbols:
                    logger.info(f"Optimizing model for {symbol}")
                    
                    # Optimize hyperparameters
                    best_params = self.rl_model_tuner.optimize_hyperparameters(symbol)
                    
                    # Train model with best parameters
                    performance = self.rl_model_tuner.train_model(
                        symbol, 
                        total_timesteps=100000, 
                        custom_hyperparams={
                            'learning_rates': {symbol: best_params['learning_rate']},
                            'batch_sizes': {symbol: best_params['batch_size']},
                            'gamma_values': {symbol: best_params['gamma']}
                        }
                    )
                    
                    # Log performance
                    logger.info(f"{symbol} Model Performance: {performance}")
                
                # Wait before next optimization cycle
                time.sleep(24 * 3600)  # 24-hour interval
            
            except Exception as e:
                logger.error(f"Model optimization error: {e}")
                time.sleep(3600)  # Wait 1 hour before retry
    
    def monitor_and_optimize_roi(self):
        """
        Continuously monitor and optimize ROI for RL models
        """
        while True:
            try:
                # Simulate ROI tracking (replace with actual ROI retrieval)
                for symbol in self.roi_workflow.symbols:
                    # Placeholder: Get current ROI from optimization results
                    current_roi = self._get_current_roi(symbol)
                    
                    # Adaptive ROI tuning
                    optimization_params = self.roi_workflow.adaptive_roi_tuning(
                        current_roi, 
                        symbol
                    )
                    
                    # Apply optimized parameters to RL models
                    if optimization_params:
                        self._update_rl_model_params(symbol, optimization_params)
                
                # Wait before next optimization cycle
                time.sleep(3600)  # 1-hour interval
            
            except Exception as e:
                logger.error(f"ROI optimization error: {e}")
                time.sleep(600)  # Wait 10 minutes before retry
    
    def _get_current_roi(self, symbol: str) -> float:
        """
        Retrieve current ROI for a given symbol
        
        Args:
            symbol (str): Cryptocurrency symbol
        
        Returns:
            float: Current Return on Investment
        """
        # TODO: Implement actual ROI retrieval from optimization results
        # This is a placeholder implementation
        import random
        return random.uniform(-0.1, 0.2)
    
    def _update_rl_model_params(self, symbol: str, params: Dict):
        """
        Update RL model hyperparameters
        
        Args:
            symbol (str): Cryptocurrency symbol
            params (Dict): Optimization parameters
        """
        logger.info(f"Updating RL model parameters for {symbol}: {params}")
        # TODO: Implement actual parameter update mechanism
        # This might involve modifying model configuration files or 
        # sending signals to running RL training processes

def check_process_health(process, timeout=30):
    """
    Monitor process health and log potential issues
    
    Args:
        process (subprocess.Popen): Running process
        timeout (int): Timeout in seconds before checking for potential issues
    """
    start_time = time.time()
    
    while process.poll() is None:
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time > timeout:
            # Check for potential error conditions
            try:
                # Capture last 50 lines of error logs
                error_log_path = 'logs/error_monitoring.log'
                if os.path.exists(error_log_path):
                    with open(error_log_path, 'r') as f:
                        error_lines = f.readlines()[-50:]
                        
                    # Log potential critical errors
                    critical_patterns = [
                        'ERROR', 
                        'CRITICAL', 
                        'Failed', 
                        'Unable to', 
                        'Connection refused',
                        'Timeout',
                        'No module named'
                    ]
                    
                    potential_errors = [
                        line for line in error_lines 
                        if any(pattern.lower() in line.lower() for pattern in critical_patterns)
                    ]
                    
                    if potential_errors:
                        logger.warning("Potential errors detected during agent launch:")
                        for error in potential_errors:
                            logger.warning(error.strip())
                
                # Additional system resource checks
                import psutil
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                
                if cpu_usage > 90 or memory_usage > 90:
                    logger.critical(f"High resource utilization detected: CPU {cpu_usage}%, Memory {memory_usage}%")
            
            except Exception as e:
                logger.error(f"Error during health check: {e}")
            
            break
        
        time.sleep(1)

def launch_agents():
    """
    Launch agents with timeout and error monitoring
    """
    try:
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Start auto-remediation agent before launching other agents
        start_auto_remediation_agent()
        
        # Add WebResearchAgent to available agents
        agent_configurations = {
            'market_intelligence': {
                'module': 'crew_agents.src.market_intelligence_agent',
                'log_file': 'market_intelligence.log',
                'required': True
            },
            'continuous_optimization': {
                'module': 'crew_agents.src.continuous_optimization_agent',
                'log_file': 'continuous_optimization.log',
                'required': True
            },
            'logging_management': {
                'module': 'crew_agents.src.logging_management_agent',
                'log_file': 'logging_management.log',
                'required': True
            },
            'rl_model_tuner': {
                'module': 'crew_agents.src.rl_model_tuner',
                'log_file': 'rl_model_tuner.log',
                'required': True
            },
            'crypto_sentiment': {
                'module': 'crypto_sentiment',
                'log_file': 'crypto_sentiment.log',
                'required': False
            },
            'web_research_agent': {
                'module': 'error_monitoring_agent.WebResearchAgent',
                'log_file': 'web_research.log',
                'required': True
            }
        }
        
        # Launch agents
        launcher = AgentLauncher()
        launcher.launch_all_agents()
        
        # Start model optimization in a separate thread
        model_opt_thread = threading.Thread(target=launcher.run_model_optimization)
        model_opt_thread.start()
        
        # Start ROI optimization in a separate thread
        roi_opt_thread = threading.Thread(target=launcher.monitor_and_optimize_roi)
        roi_opt_thread.start()
        
        # Wait for all threads to complete
        model_opt_thread.join()
        roi_opt_thread.join()
        
    except Exception as launch_error:
        # Log critical launch failure
        logging.critical(f"Agent launch failed: {launch_error}")
        
        # Attempt to send critical alert
        try:
            from crew_agents.src.error_monitoring_agent import AutoRemediationAgent
            remediation_agent = AutoRemediationAgent()
            remediation_agent._send_critical_alert(
                title='Trading Bot Launch Failure',
                message=str(launch_error),
                severity='CRITICAL'
            )
        except Exception as alert_error:
            logging.error(f"Failed to send critical alert: {alert_error}")
        
        # Potential system-level recovery
        sys.exit(1)

def start_auto_remediation_agent():
    """
    Start auto-remediation agent in a separate thread
    """
    try:
        # Get project root dynamically, similar to how it's done at the top of the file
        project_root_dir = os.path.dirname(os.path.abspath(__file__))
        target_log_file = os.path.join(project_root_dir, 'logs', 'agent_launcher.log')

        from crew_agents.src.error_monitoring_agent import AutoRemediationAgent
        
        # Create and start the auto-remediation agent with the correct log path
        auto_remediation_agent = AutoRemediationAgent(log_path=target_log_file)
        auto_remediation_thread = threading.Thread(
            target=auto_remediation_agent.monitor_and_resolve_errors, 
            daemon=True
        )
        auto_remediation_thread.start()
        
        return auto_remediation_thread
    
    except Exception as agent_start_error:
        logging.critical(f"Failed to start auto-remediation agent: {agent_start_error}")
        return None

if __name__ == "__main__":
    exchanges = [
        'coinbase',   # Primary exchange
        'kraken',     # Alternative exchange
        'gateio'      # Additional exchange
    ]
    launch_agents()
