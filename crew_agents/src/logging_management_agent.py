import os
import sys
import time
import json
import logging
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

class LoggingManagementAgent:
    """
    Comprehensive logging and error management agent
    Monitors system health, captures errors, and provides diagnostic insights
    """
    def __init__(
        self, 
        log_directory: str = None, 
        max_log_age_days: int = 30,
        error_threshold: int = 5
    ):
        """
        Initialize Logging Management Agent
        
        Args:
            log_directory (str): Directory to manage logs
            max_log_age_days (int): Maximum days to retain log files
            error_threshold (int): Number of errors before triggering alert
        """
        self.log_directory = log_directory or os.path.join(project_root, 'logs')
        self.max_log_age_days = max_log_age_days
        self.error_threshold = error_threshold
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Error tracking
        self.error_history: List[Dict[str, Any]] = []
        self.system_health_metrics: Dict[str, Any] = {
            'total_errors': 0,
            'last_error_timestamp': None,
            'error_types': {}
        }
    
    def capture_error(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None, 
        severity: str = 'ERROR'
    ):
        """
        Capture and log detailed error information
        
        Args:
            error (Exception): The error that occurred
            context (Dict): Additional context about the error
            severity (str): Error severity level
        """
        error_details = {
            'timestamp': datetime.now().isoformat(),
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {},
            'severity': severity
        }
        
        # Log to file
        error_log_path = os.path.join(
            self.log_directory, 
            f'error_log_{datetime.now().strftime("%Y%m%d")}.json'
        )
        
        try:
            # Append to error log file
            with open(error_log_path, 'a+') as f:
                json.dump(error_details, f)
                f.write('\n')
            
            # Update system health metrics
            self.system_health_metrics['total_errors'] += 1
            self.system_health_metrics['last_error_timestamp'] = error_details['timestamp']
            error_type = error_details['type']
            self.system_health_metrics['error_types'][error_type] = \
                self.system_health_metrics['error_types'].get(error_type, 0) + 1
            
            # Store in memory for quick reference
            self.error_history.append(error_details)
            
            # Trigger alert if error threshold is exceeded
            if self.system_health_metrics['total_errors'] >= self.error_threshold:
                self._trigger_error_alert()
        
        except Exception as log_error:
            print(f"Failed to log error: {log_error}")
            print(f"Original Error: {error_details}")
    
    def _trigger_error_alert(self):
        """
        Send alerts when error threshold is exceeded
        Can be extended to send emails, Slack messages, etc.
        """
        alert_message = f"""
        SYSTEM HEALTH ALERT: 
        Total Errors Exceeded Threshold
        
        Error Summary:
        - Total Errors: {self.system_health_metrics['total_errors']}
        - Last Error: {self.system_health_metrics['last_error_timestamp']}
        - Error Types: {json.dumps(self.system_health_metrics['error_types'], indent=2)}
        """
        
        # Log critical alert
        self.logger.critical(alert_message)
        
        # TODO: Implement external notification system
        # Examples:
        # - Send email
        # - Send Slack/Discord message
        # - Trigger PagerDuty incident
    
    def clean_old_logs(self):
        """
        Remove log files older than max_log_age_days
        """
        current_time = datetime.now()
        
        for filename in os.listdir(self.log_directory):
            file_path = os.path.join(self.log_directory, filename)
            
            # Check if it's a file and not a directory
            if os.path.isfile(file_path):
                # Get file modification time
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Calculate age
                file_age = current_time - mod_time
                
                # Delete if older than max_log_age_days
                if file_age > timedelta(days=self.max_log_age_days):
                    try:
                        os.remove(file_path)
                        self.logger.info(f"Deleted old log file: {filename}")
                    except Exception as e:
                        self.logger.error(f"Failed to delete log file {filename}: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Retrieve current system health metrics
        
        Returns:
            Dict: Comprehensive system health report
        """
        return {
            'total_errors': self.system_health_metrics['total_errors'],
            'last_error_timestamp': self.system_health_metrics['last_error_timestamp'],
            'error_types': self.system_health_metrics['error_types'],
            'recent_errors': self.error_history[-10:]  # Last 10 errors
        }

def main():
    """
    Initialize and run Logging Management Agent
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(project_root, 'logs', 'logging_management.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging_agent = LoggingManagementAgent()
    
    # Periodic log cleanup
    logging_agent.clean_old_logs()

if __name__ == "__main__":
    main()
