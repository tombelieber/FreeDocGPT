import logging
from typing import Dict, List, Optional

import requests
import streamlit as st

from ..config import get_settings

logger = logging.getLogger(__name__)


def check_ollama_status() -> Dict[str, any]:
    """Check Ollama connection and available models."""
    settings = get_settings()
    status = {
        "connected": False,
        "models": [],
        "embed_model_available": False,
        "gen_model_available": False,
        "error": None
    }
    
    try:
        response = requests.get(f"{settings.ollama_host}/api/tags")
        
        if response.status_code == 200:
            models_data = response.json()
            
            if "models" in models_data and models_data["models"]:
                available_models = [m["name"] for m in models_data["models"]]
                status["connected"] = True
                status["models"] = available_models
                
                # Check if required models are installed
                status["embed_model_available"] = any(
                    settings.embed_model.lower() in model.lower() 
                    for model in available_models
                )
                status["gen_model_available"] = any(
                    settings.gen_model.lower() in model.lower() 
                    for model in available_models
                )
            else:
                status["error"] = "No models installed"
        else:
            status["error"] = f"HTTP {response.status_code}"
            
    except requests.ConnectionError:
        status["error"] = "Cannot connect to Ollama"
    except Exception as e:
        status["error"] = str(e)
    
    return status


def display_ollama_status():
    """Display Ollama status in Streamlit UI."""
    settings = get_settings()
    status = check_ollama_status()
    
    if status["connected"]:
        st.success(f"✅ Ollama is running! Available models: {', '.join(status['models'])}")
        
        if not status["embed_model_available"]:
            st.error(
                f"❌ Embedding model '{settings.embed_model}' not found. "
                f"Please run: `ollama pull {settings.embed_model}`"
            )
        else:
            st.info(f"✅ Embedding model '{settings.embed_model}' is available")
        
        if not status["gen_model_available"]:
            st.error(
                f"❌ Generation model '{settings.gen_model}' not found. "
                f"Please run: `ollama pull {settings.gen_model}`"
            )
        else:
            st.info(f"✅ Generation model '{settings.gen_model}' is available")
    else:
        error_msg = status["error"]
        if "Cannot connect" in error_msg:
            st.error("❌ Cannot connect to Ollama. Make sure it's running with: `ollama serve`")
        else:
            st.error(f"❌ Ollama error: {error_msg}")