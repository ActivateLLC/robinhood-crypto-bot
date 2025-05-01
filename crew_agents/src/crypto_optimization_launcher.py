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
        logging.FileHandler('/Users/activate/Dev/robinhood-crypto-bot/logs/crypto_optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('CryptoOptimizationLauncher')

def launch_btc_roi_optimizer():
    try:
        from crew_agents.src.btc_roi_optimizer import main as btc_roi_optimizer_main
        logger.info("Launching BTC ROI Optimizer")
        os.environ['TRADING_SYMBOL'] = 'BTC-USD'
        btc_roi_optimizer_main()
    except Exception as e:
        logger.error(f"BTC ROI Optimizer failed: {e}")

def launch_popcat_roi_optimizer():
    try:
        from crew_agents.src.btc_roi_optimizer import main as popcat_roi_optimizer_main
        logger.info("Launching PopCat ROI Optimizer")
        os.environ['TRADING_SYMBOL'] = 'POPCAT-USD'
        popcat_roi_optimizer_main()
    except Exception as e:
        logger.error(f"PopCat ROI Optimizer failed: {e}")

def launch_btc_crew_optimization():
    try:
        from crew_agents.src.crew_optimization_manager import main as btc_crew_optimization_main
        logger.info("Launching BTC Crew Optimization")
        os.environ['TRADING_SYMBOL'] = 'BTC-USD'
        btc_crew_optimization_main()
    except Exception as e:
        logger.error(f"BTC Crew Optimization failed: {e}")

def launch_popcat_crew_optimization():
    try:
        from crew_agents.src.crew_optimization_manager import main as popcat_crew_optimization_main
        logger.info("Launching PopCat Crew Optimization")
        os.environ['TRADING_SYMBOL'] = 'POPCAT-USD'
        popcat_crew_optimization_main()
    except Exception as e:
        logger.error(f"PopCat Crew Optimization failed: {e}")

def launch_all_optimizers():
    optimizers = [
        launch_btc_roi_optimizer,
        launch_popcat_roi_optimizer,
        launch_btc_crew_optimization,
        launch_popcat_crew_optimization
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
