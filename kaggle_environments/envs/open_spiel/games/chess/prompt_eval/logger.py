"""Simple logger for chess prompt evaluation."""
import logging
import sys

def get_logger(name: str = "chess_prompt_eval") -> logging.Logger:
    """
    Get a logger instance for the chess prompt evaluation tool.
    
    Args:
        name: Name for the logger instance
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handler if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
    return logger