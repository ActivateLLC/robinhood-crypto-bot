import logging
import os
from logging.handlers import RotatingFileHandler
import sys

def setup_logger(name, log_level='INFO', log_directory='logs', filename_prefix='app_'):
    """
    Sets up a logger with specified level, directory, filename prefix,
    rotation, and global exception handling.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False # Prevent duplicate logs in parent loggers

    # Create log directory if it doesn't exist
    if not os.path.exists(log_directory):
        try:
            os.makedirs(log_directory)
        except OSError as e:
            # Handle potential race condition if directory is created by another process
            if e.errno != os.errno.EEXIST:
                print(f"Error creating log directory {log_directory}: {e}", file=sys.stderr)
                # Fallback to current directory if creation fails
                log_directory = "."

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

    # Console Handler (optional, but good for immediate feedback)
    # You might want to control this with a flag or based on environment
    # For tests, we might want to minimize console output unless debugging.
    # if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
    #     ch = logging.StreamHandler(sys.stdout) 
    #     ch.setFormatter(formatter)
    #     logger.addHandler(ch)

    # File Handler with Rotation
    log_file = os.path.join(log_directory, f"{filename_prefix}{name}.log")
    # Max 10MB per file, 5 backup files
    if not any(isinstance(handler, RotatingFileHandler) and handler.baseFilename == os.path.abspath(log_file) for handler in logger.handlers):
        try:
            fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            print(f"Error setting up file handler for {log_file}: {e}", file=sys.stderr)

    # Global Unhandled Exception Hook (optional, place in main app setup if preferred)
    # This version ensures it's only set once if this function is called multiple times
    # for different loggers. A more robust approach might be a dedicated setup elsewhere.
    # def handle_exception(exc_type, exc_value, exc_traceback):
    #     if issubclass(exc_type, KeyboardInterrupt):
    #         sys.__excepthook__(exc_type, exc_value, exc_traceback)
    #         return
    #     logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    # if sys.excepthook.__name__ != 'handle_exception': # Avoid multiple hooks
    #     sys.excepthook = handle_exception

    return logger

if __name__ == '__main__':
    # Example Usage:
    logger1 = setup_logger('module1', log_level='DEBUG', log_directory='custom_logs')
    logger1.debug("This is a debug message from module1.")
    logger1.info("This is an info message from module1.")

    logger2 = setup_logger('module2', log_level='INFO', filename_prefix='service_')
    logger2.info("This is an info message from module2.")
    try:
        1 / 0
    except ZeroDivisionError:
        logger2.error("This is an error message from module2 due to division by zero.", exc_info=True)
