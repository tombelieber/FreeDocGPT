import logging
import time
from typing import Dict, Generator, Optional, Tuple

import ollama

from ..config import get_settings

logger = logging.getLogger(__name__)


class ChatService:
    """Handles chat interactions with LLM."""
    
    def __init__(self, model: Optional[str] = None):
        settings = get_settings()
        self.model = model or settings.gen_model
    
    def stream_chat(self, messages: list) -> Generator[str, None, Dict]:
        """Stream chat responses with statistics."""
        response_text = ""
        token_count = 0
        start_time = time.time()
        first_token_time = None
        
        try:
            stream = ollama.chat(
                model=self.model, 
                messages=messages, 
                stream=True
            )
            
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        response_text += content
                        token_count += len(content.split())
                        yield content
                        
        except Exception as e:
            logger.error(f"Chat streaming error: {e}")
            yield f"\nError: {e}"
        
        # Calculate statistics
        end_time = time.time()
        total_time = end_time - start_time
        time_to_first_token = (first_token_time - start_time) if first_token_time else 0
        
        stats = {
            "total_time": total_time,
            "time_to_first_token": time_to_first_token,
            "tokens": token_count,
            "tokens_per_sec": token_count / total_time if total_time > 0 else 0,
        }
        
        return stats