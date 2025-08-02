import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    """Initializes and returns a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Log file name based on current date
    log_filename = datetime.now().strftime('%Y-%m-%d') + '.log'
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, log_filename))
    file_handler.setLevel(logging.INFO)

    # Console handler for real-time output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Log format
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger