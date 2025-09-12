import logging
import time
from typing import Dict, Generator, Optional, Tuple, List, Any

import ollama

from ..config import get_settings
from .context_manager import ContextManager, ContextUsage

logger = logging.getLogger(__name__)


class ChatService:
    """Handles chat interactions with LLM with context management."""
    
    def __init__(self, model: Optional[str] = None):
        settings = get_settings()
        self.model = model or settings.gen_model
        self.settings = settings
        self.context_manager = ContextManager(self.model)
    
    def build_messages_with_history(
        self, 
        system_prompt: str,
        user_prompt: str,
        conversation_history: Optional[List[Dict]] = None,
        search_results_tokens: int = 0,
        enable_sliding_window: bool = True
    ) -> Tuple[List[Dict], ContextUsage, Dict[str, Any]]:
        """Build messages list including conversation history with context management.
        
        Args:
            system_prompt: System message with context
            user_prompt: Current user question
            conversation_history: Previous conversation turns
            search_results_tokens: Tokens used by search results in context
            enable_sliding_window: Whether to apply sliding window if needed
            
        Returns:
            Tuple of (messages_list, context_usage, optimization_info)
        """
        # Build initial message list
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if provided - use ALL history initially
        if conversation_history:
            for msg in conversation_history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current user prompt
        messages.append({"role": "user", "content": user_prompt})
        
        # Apply context optimization if enabled
        if enable_sliding_window:
            optimized_messages, context_usage, optimization_info = (
                self.context_manager.optimize_messages_for_context(
                    messages, search_results_tokens
                )
            )
            return optimized_messages, context_usage, optimization_info
        else:
            # Just analyze usage without optimization
            context_usage = self.context_manager.analyze_context_usage(
                messages, search_results_tokens
            )
            return messages, context_usage, {'sliding_applied': False}
    
    def get_context_analysis(
        self,
        messages: List[Dict],
        search_results_tokens: int = 0
    ) -> ContextUsage:
        """Get context usage analysis for UI display."""
        return self.context_manager.analyze_context_usage(messages, search_results_tokens)
    
    def get_context_recommendations(self, context_usage: ContextUsage) -> List[str]:
        """Get context optimization recommendations."""
        return self.context_manager.get_context_recommendations(context_usage)
    
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