import os
import logging
from pathlib import Path

def setup_logging():
    """Set up logging configuration for all modules."""
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).parent.parent / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Define handlers for different loggers
    handlers = {
        'main': [
            logging.StreamHandler(),
            logging.FileHandler(str(logs_dir / 'main.log'), mode='a')
        ],
        'azure_client': [
            logging.StreamHandler(),
            logging.FileHandler(str(logs_dir / 'azure_client.log'), mode='a')
        ],
        'token_limiter': [
            logging.StreamHandler(),
            logging.FileHandler(str(logs_dir / 'token_limiter.log'), mode='a')
        ]
    }
    
    # Configure format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure each logger
    for logger_name, logger_handlers in handlers.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        logger.handlers = []
        
        # Add and configure new handlers
        for handler in logger_handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
    # Disable propagation to avoid duplicate logs
    for logger_name in handlers.keys():
        logging.getLogger(logger_name).propagate = False 