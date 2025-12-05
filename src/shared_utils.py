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

def setup_llm_client(agent_type=None):
    """
    Setup LLM client using LiteLLM for both Green and Purple agents.
    Returns (llm_client, model_name).
    
    Args:
        agent_type (str, optional): Either 'green', 'purple', or None for default behavior
    """
    # Get model configuration from environment - agent-specific or fallback to general
    if agent_type == 'green':
        model_name = os.getenv("GREEN_AGENT_MODEL")
    elif agent_type == 'purple':
        model_name = os.getenv("PURPLE_AGENT_MODEL")
    else:
        # Default behavior - try GREEN first, then PURPLE,
        model_name = os.getenv("GREEN_AGENT_MODEL") or os.getenv("PURPLE_AGENT_MODEL")
    
    api_key = os.getenv("LLM_API_KEY")
    
    if not api_key:
        print("⚠️  No LLM_API_KEY found, LLM functionality disabled")
        return None, model_name
    
    try:
        import litellm
        
        # Set API keys for LiteLLM
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["ANTHROPIC_API_KEY"] = api_key
            os.environ["AZURE_API_KEY"] = api_key
            
            # Azure specific settings
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if azure_endpoint:
                # Extract base URL from full endpoint if needed
                if "/openai/deployments/" in azure_endpoint:
                    base_url = azure_endpoint.split("/openai/deployments/")[0] + "/"
                else:
                    base_url = azure_endpoint
                
                os.environ["AZURE_API_BASE"] = base_url
                api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
                os.environ["AZURE_API_VERSION"] = api_version
                
                # Use Azure model format for LiteLLM
                model_name = f"azure/{model_name}"
                print(f"🔑 Using Azure OpenAI with base URL: {base_url}")
                print(f"🔑 Using Azure model: {model_name}")
            else:
                print(f"🔑 Using standard OpenAI model: {model_name}")
        
        print(f"🤖 LiteLLM configured with model: {model_name}")
        return litellm, model_name
        
    except ImportError as e:
        print(f"⚠️  LiteLLM not available: {e}")
        return None, model_name
    except Exception as e:
        print(f"⚠️  LLM setup error: {e}")
        return None, model_name

def get_pydantic_ai_model(agent_type=None) -> str:
    """
    Get the correct model name format for Pydantic AI based on environment configuration.
    Expects model names in Pydantic AI format: provider:model_name
    
    Args:
        agent_type (str, optional): Either 'green', 'purple', or None for default behavior
    """
    # Get model configuration from environment - agent-specific or fallback to general
    if agent_type == 'green':
        model_name = os.getenv("GREEN_AGENT_MODEL", "openai:gpt-4o")
    elif agent_type == 'purple':
        model_name = os.getenv("PURPLE_AGENT_MODEL") or "openai:gpt-4o" 
    else:
        model_name = os.getenv("GREEN_AGENT_MODEL", "openai:gpt-4o")
    
    api_key = os.getenv("LLM_API_KEY")
    
    if not api_key:
        print("⚠️  No LLM_API_KEY found")
        return model_name
    
    # Parse provider and model from format: provider:model_name
    if ":" not in model_name:
        print(f"⚠️  Invalid model format: {model_name}. Expected format: provider:model_name")
        print("   Examples: openai:gpt-4o, azure:gpt-4, anthropic:claude-3-sonnet")
        return f"openai:{model_name}"  # Fallback to OpenAI
    
    provider, model = model_name.split(":", 1)
    
    if provider == "azure":
        print(f"🔑 Using Azure OpenAI model: {model_name}")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not azure_endpoint:
            print("⚠️  AZURE_OPENAI_ENDPOINT not set for Azure model")
            return model_name
        
        # Set up environment variables for Azure
        if "/openai/deployments/" in azure_endpoint:
            base_url = azure_endpoint.split("/openai/deployments/")[0]
        else:
            base_url = azure_endpoint.rstrip("/")
        
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
        os.environ["AZURE_OPENAI_ENDPOINT"] = base_url
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        os.environ["AZURE_OPENAI_API_VERSION"] = api_version
        os.environ["OPENAI_API_VERSION"] = api_version
        
        print(f"🔑 Base URL: {base_url}")
        print(f"🔑 Deployment: {model}")
        
    elif provider == "anthropic":
        print(f"🔑 Using Anthropic model: {model_name}")
        os.environ["ANTHROPIC_API_KEY"] = api_key
        
    elif provider == "gemini":
        print(f"🔑 Using Google Gemini model: {model_name}")
        os.environ["GOOGLE_API_KEY"] = api_key
        
    elif provider == "groq":
        print(f"🔑 Using Groq model: {model_name}")
        os.environ["GROQ_API_KEY"] = api_key
        
    elif provider == "openai":
        print(f"🔑 Using OpenAI model: {model_name}")
        os.environ["OPENAI_API_KEY"] = api_key
        
    else:
        print(f"⚠️  Unknown provider: {provider}. Supported: openai, azure, anthropic, gemini, groq")
        print("   Falling back to OpenAI format")
        os.environ["OPENAI_API_KEY"] = api_key
        return f"openai:{model}"
    
    return model_name