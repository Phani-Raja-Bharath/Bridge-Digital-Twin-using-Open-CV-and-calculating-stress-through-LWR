"""
Logging configuration for Bridge_V3 application
Provides centralized logging setup and utilities
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from config import Config


class CustomFormatter(logging.Formatter):
    """Custom formatter with color coding for console output"""
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logging(app_name: str = "bridge_v3") -> logging.Logger:
    """Setup comprehensive logging configuration"""
    
    # Create logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper()))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler with color formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(CustomFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler for all logs
    log_file = os.path.join(log_dir, f"{app_name}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler for errors and above
    error_file = os.path.join(log_dir, f"{app_name}_errors.log")
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)
    
    # Performance log for timing critical operations
    perf_file = os.path.join(log_dir, f"{app_name}_performance.log")
    perf_handler = logging.handlers.RotatingFileHandler(
        perf_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    perf_handler.setLevel(logging.INFO)
    perf_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - PERF - %(message)s'
    )
    perf_handler.setFormatter(perf_formatter)
    
    # Create performance logger
    perf_logger = logging.getLogger(f"{app_name}.performance")
    perf_logger.addHandler(perf_handler)
    perf_logger.propagate = False
    
    # Application logger
    app_logger = logging.getLogger(app_name)
    app_logger.info(f"Logging initialized for {app_name}")
    app_logger.info(f"Log level: {Config.LOG_LEVEL}")
    app_logger.info(f"Log files: {log_file}, {error_file}, {perf_file}")
    
    return app_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with consistent configuration"""
    return logging.getLogger(name)


def log_performance(logger: logging.Logger, operation: str, duration: float, **kwargs):
    """Log performance metrics for operations"""
    perf_logger = logging.getLogger("bridge_v3.performance")
    
    extras = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    message = f"Operation: {operation}, Duration: {duration:.3f}s"
    if extras:
        message += f", {extras}"
    
    perf_logger.info(message)


def log_function_call(func):
    """Decorator to log function calls with timing"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        
        logger.debug(f"Calling {func.__name__} with args={len(args)}, kwargs={list(kwargs.keys())}")
        
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Log performance for potentially slow operations
            if duration > 0.1:  # Operations taking more than 100ms
                log_performance(logger, func.__name__, duration)
            
            logger.debug(f"Completed {func.__name__} in {duration:.3f}s")
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed {func.__name__} after {duration:.3f}s: {e}")
            raise
            
    return wrapper


class LogContext:
    """Context manager for logging operations with automatic timing"""
    
    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
        self.logger.info(f"Starting {self.operation}" + (f" ({context_str})" if context_str else ""))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed {self.operation} in {duration:.3f}s")
            log_performance(self.logger, self.operation, duration, **self.context)
        else:
            self.logger.error(f"Failed {self.operation} after {duration:.3f}s: {exc_val}")
        
        return False  # Don't suppress exceptions


# Initialize logging when module is imported
if not logging.getLogger().handlers:
    setup_logging()