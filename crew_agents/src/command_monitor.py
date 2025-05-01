import time
import logging
import subprocess
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/activate/Dev/robinhood-crypto-bot/logs/command_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def monitor_command(command, cwd=None, check_interval=60, max_duration=None):
    """
    Monitor a long-running command with periodic status checks
    
    Args:
        command (list): Command to execute
        cwd (str, optional): Working directory
        check_interval (int): Seconds between status checks
        max_duration (int, optional): Maximum total runtime in seconds
    """
    logger = logging.getLogger('CommandMonitor')
    
    # Start the process
    try:
        process = subprocess.Popen(
            command, 
            cwd=cwd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        logger.info(f"Started command: {' '.join(command)}")
        logger.info(f"Working directory: {cwd or os.getcwd()}")
    except Exception as e:
        logger.error(f"Failed to start command: {e}")
        return False
    
    start_time = time.time()
    last_check = start_time
    
    try:
        while process.poll() is None:
            current_time = time.time()
            
            # Check if we've exceeded max duration
            if max_duration and (current_time - start_time) > max_duration:
                logger.warning(f"Command exceeded max duration of {max_duration} seconds")
                process.terminate()
                break
            
            # Periodic status check
            if current_time - last_check >= check_interval:
                # Read any available output
                stdout = process.stdout.read() if process.stdout else ''
                stderr = process.stderr.read() if process.stderr else ''
                
                if stdout:
                    logger.info(f"STDOUT: {stdout}")
                if stderr:
                    logger.warning(f"STDERR: {stderr}")
                
                last_check = current_time
            
            time.sleep(5)  # Small sleep to prevent tight looping
        
        # Final output capture
        stdout, stderr = process.communicate()
        if stdout:
            logger.info(f"Final STDOUT: {stdout}")
        if stderr:
            logger.warning(f"Final STDERR: {stderr}")
        
        # Log exit status
        if process.returncode == 0:
            logger.info("Command completed successfully")
        else:
            logger.error(f"Command failed with return code {process.returncode}")
        
        return process.returncode == 0
    
    except Exception as e:
        logger.error(f"Monitoring error: {e}")
        return False

if __name__ == '__main__':
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python command_monitor.py <command> [args...]")
        sys.exit(1)
    
    command = sys.argv[1:]
    monitor_command(
        command, 
        check_interval=60,  # Check every minute
        max_duration=3600   # Max 1 hour runtime
    )
