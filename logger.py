import os
import logging
import logging.handlers
from datetime import datetime
from typing import Optional

def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, a default path is used.
        console_output: Whether to output logs to console
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file is None:
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/coin_master_bot_{timestamp}.log"
    else:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Convert log level string to logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler (always enabled)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized at level {log_level}")
    logger.info(f"Log file: {log_file}")
    
    return logger


class UILogHandler(logging.Handler):
    """
    Custom logging handler that forwards log messages to the UI.
    """
    
    def __init__(self, ui_handler, level=logging.NOTSET):
        """
        Initialize the UI log handler.
        
        Args:
            ui_handler: UI handler instance with add_log method
            level: Logging level
        """
        super().__init__(level)
        self.ui_handler = ui_handler
    
    def emit(self, record):
        """
        Emit a log record to the UI.
        
        Args:
            record: Log record
        """
        try:
            msg = self.format(record)
            self.ui_handler.add_log(msg)
        except Exception:
            self.handleError(record)


def add_ui_handler(logger: logging.Logger, ui_handler: any) -> None:
    """
    Add a UI handler to the logger.
    
    Args:
        logger: Logger instance
        ui_handler: UI handler instance with add_log method
    """
    # Create UI handler
    ui_log_handler = UILogHandler(ui_handler)
    ui_log_handler.setLevel(logging.INFO)  # Only show INFO and above in UI
    ui_log_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    # Add to logger
    logger.addHandler(ui_log_handler)
    logger.info("UI logging enabled")
