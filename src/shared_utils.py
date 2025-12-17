"""
Shared utilities for agents.
"""

import os
from openai import OpenAI
from typing import TypeAlias

OpenAISchema: TypeAlias = str

# --- Logging ---

import logging
import sys
from logging.handlers import TimedRotatingFileHandler

def setup_logger(name: str, log_dir: str = "logs"):
    """Sets up a logger that writes to a file and streams to the console."""
    
    # If log_dir is not provided, use the environment variable or a default
    if not log_dir:
        log_dir = os.getenv("AGENT_LOG_DIR", "logs")

    # Create log directory if it doesn't exist
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_path = os.path.join(project_root, log_dir)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        
    log_file = os.path.join(log_path, f"{name}.log")
    
    # Get logger and prevent duplicate handlers
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger  # Already configured

    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

def get_pydantic_ai_model(agent_type=None) -> str:
    """
    Get the model name from environment configuration and return as-is.
    Environment variables should be configured with all necessary provider settings.
    
    Args:
        agent_type (str, optional): Either 'green', 'purple', or None for default behavior
    """
    # Get model configuration from environment - agent-specific or fallback to general
    if agent_type == 'green':
        # Auto-detect green agent model based on available API keys
        if os.getenv("AZURE_OPENAI_API_KEY"):
            model_name = "azure:gpt-4o"
        elif os.getenv("OPENAI_API_KEY"):
            model_name = "openai:gpt-4o"
        else:
            model_name = "azure:gpt-4o"  # Default to Azure
    elif agent_type == 'purple':
        model_name = os.getenv("PURPLE_AGENT_MODEL") or "openai:gpt-4o" 
    else:
        # Default case: same logic as green agent
        if os.getenv("AZURE_OPENAI_API_KEY"):
            model_name = "azure:gpt-4o"
        elif os.getenv("OPENAI_API_KEY"):
            model_name = "openai:gpt-4o"
        else:
            model_name = "azure:gpt-4o"  # Default to Azure
    
    print(f"🤖 Using model: {model_name}")
    
    # Return the model name as configured in environment
    # All provider-specific environment variables should be set in .env file
    return model_name