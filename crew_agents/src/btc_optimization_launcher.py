import sys
import os
import logging
from multiprocessing import Process

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/activate/Dev/robinhood-crypto-bot/logs/btc_optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('BTCOptimizationLauncher')

def launch_roi_optimizer():
    try:
        from crew_agents.src.btc_roi_optimizer import main as roi_optimizer_main
        logger.info("Launching BTC ROI Optimizer")
        roi_optimizer_main()
    except Exception as e:
        logger.error(f"ROI Optimizer failed: {e}")

def launch_crew_optimization():
    try:
        from crew_agents.src.crew_optimization_manager import main as crew_optimization_main
        logger.info("Launching Crew Optimization")
        crew_optimization_main()
    except Exception as e:
        logger.error(f"Crew Optimization failed: {e}")

def launch_all_optimizers():
    optimizers = [
        launch_roi_optimizer,
        launch_crew_optimization
    ]
    
    processes = []
    for optimizer in optimizers:
        try:
            p = Process(target=optimizer)
            p.start()
            processes.append(p)
        except Exception as e:
            logger.error(f"Failed to launch optimizer {optimizer.__name__}: {e}")
    
    # Wait for all processes to complete
    for p in processes:
        p.join()

if __name__ == '__main__':
    launch_all_optimizers()
