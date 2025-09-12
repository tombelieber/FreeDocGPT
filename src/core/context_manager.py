"""
Smart Context Management for Persistent Chat History

Handles token counting, context window management, and sliding window fallback
for maintaining long conversations within model context limits.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import tiktoken
from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ContextUsage:
    """Context usage statistics."""
    system_tokens: int
    history_tokens: int
    current_tokens: int
    documents_tokens: int
    total_tokens: int
    free_tokens: int
    usage_percentage: float
    warning_level: str  # 'green', 'yellow', 'red'


@dataclass
class ContextWindow:
    """Context window configuration."""
    max_tokens: int
    warning_threshold: float
    critical_threshold: float
    sliding_window_size: int


class ContextManager:
    """
    Smart context management for persistent chat conversations.
    
    Features:
    - Maintains full conversation history until context limit is reached
    - Automatic sliding window when approaching limits
    - Token counting and usage statistics
    - Context optimization strategies
    """
    
    def __init__(self, model_name: str = "gpt-oss:20b"):
        """Initialize context manager."""
        self.settings = get_settings()
        self.model_name = model_name
        
        # Initialize tokenizer for accurate counting
        try:
            # Use tiktoken for OpenAI-compatible models or fallback to character approximation
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.use_tiktoken = True
            logger.info("Using tiktoken for accurate token counting")
        except Exception as e:
            logger.warning(f"tiktoken not available, using character approximation: {e}")
            self.tokenizer = None
            self.use_tiktoken = False
        
        # Context window configuration
        self.context_window = self._get_context_window_config()
        logger.info(f"Context window: {self.context_window.max_tokens} tokens, "
                   f"warning at {self.context_window.warning_threshold*100:.0f}%")
    
    def _get_context_window_config(self) -> ContextWindow:
        """Get context window configuration based on settings and model."""
        # Model-specific context windows
        model_contexts = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "claude-2": 100000,
            "claude-instant": 100000,
            "llama-2-7b": 4096,
            "llama-2-13b": 4096,
            "llama-2-70b": 4096,
            "mistral-7b": 8192,
            "gpt-oss:20b": 128000,  # Generous context for user's model
            "embeddinggemma:300m": 2048,
        }
        
        # Get max tokens from settings or model default
        max_tokens = getattr(self.settings, 'max_context_tokens', 
                           model_contexts.get(self.model_name, 128000))
        
        return ContextWindow(
            max_tokens=max_tokens,
            warning_threshold=getattr(self.settings, 'context_warning_threshold', 0.75),
            critical_threshold=getattr(self.settings, 'context_critical_threshold', 0.85),
            sliding_window_size=getattr(self.settings, 'sliding_window_size', 4000)
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
            
        if self.use_tiktoken and self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"tiktoken encoding failed, falling back to approximation: {e}")
                self.use_tiktoken = False
        
        # Fallback: rough approximation (4 characters ≈ 1 token)
        return max(1, len(text) // 4)
    
    def analyze_context_usage(
        self,
        messages: List[Dict[str, str]],
        search_results_tokens: int = 0
    ) -> ContextUsage:
        """
        Analyze current context usage from messages.
        
        Args:
            messages: List of conversation messages
            search_results_tokens: Tokens used by retrieved documents
            
        Returns:
            Context usage analysis
        """
        system_tokens = 0
        history_tokens = 0
        current_tokens = 0
        
        for i, msg in enumerate(messages):
            content = msg.get('content', '')
            tokens = self.count_tokens(content)
            
            if msg.get('role') == 'system':
                system_tokens += tokens
            elif i == len(messages) - 1:  # Last message (current)
                current_tokens += tokens
            else:  # History
                history_tokens += tokens
        
        total_tokens = system_tokens + history_tokens + current_tokens + search_results_tokens
        free_tokens = max(0, self.context_window.max_tokens - total_tokens)
        usage_percentage = total_tokens / self.context_window.max_tokens
        
        # Determine warning level
        if usage_percentage >= self.context_window.critical_threshold:
            warning_level = 'red'
        elif usage_percentage >= self.context_window.warning_threshold:
            warning_level = 'yellow'
        else:
            warning_level = 'green'
        
        return ContextUsage(
            system_tokens=system_tokens,
            history_tokens=history_tokens,
            current_tokens=current_tokens,
            documents_tokens=search_results_tokens,
            total_tokens=total_tokens,
            free_tokens=free_tokens,
            usage_percentage=usage_percentage,
            warning_level=warning_level
        )
    
    def should_use_sliding_window(self, context_usage: ContextUsage) -> bool:
        """Determine if sliding window should be applied."""
        return context_usage.usage_percentage >= self.context_window.warning_threshold
    
    def apply_sliding_window(
        self,
        messages: List[Dict[str, str]],
        target_history_tokens: Optional[int] = None
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Apply sliding window to keep conversation within context limits.
        
        Args:
            messages: Original message list
            target_history_tokens: Target token count for history (optional)
            
        Returns:
            Tuple of (trimmed_messages, sliding_info)
        """
        if not messages:
            return messages, {}
        
        if target_history_tokens is None:
            target_history_tokens = self.context_window.sliding_window_size
        
        # Separate system messages from conversation
        system_messages = [msg for msg in messages if msg.get('role') == 'system']
        conversation_messages = [msg for msg in messages if msg.get('role') != 'system']
        
        if not conversation_messages:
            return messages, {}
        
        # Keep messages from the end until we reach target token count
        kept_messages = []
        current_tokens = 0
        messages_kept = 0
        
        # Work backwards through conversation
        for msg in reversed(conversation_messages):
            msg_tokens = self.count_tokens(msg.get('content', ''))
            
            if current_tokens + msg_tokens <= target_history_tokens:
                kept_messages.insert(0, msg)
                current_tokens += msg_tokens
                messages_kept += 1
            else:
                break
        
        # Combine system messages with kept conversation
        trimmed_messages = system_messages + kept_messages
        
        sliding_info = {
            'original_count': len(messages),
            'trimmed_count': len(trimmed_messages),
            'messages_removed': len(conversation_messages) - messages_kept,
            'tokens_saved': sum(self.count_tokens(msg.get('content', '')) 
                              for msg in conversation_messages[:-messages_kept]),
            'sliding_applied': True
        }
        
        logger.info(f"Sliding window applied: kept {messages_kept}/{len(conversation_messages)} "
                   f"conversation messages ({current_tokens} tokens)")
        
        return trimmed_messages, sliding_info
    
    def optimize_messages_for_context(
        self,
        messages: List[Dict[str, str]],
        search_results_tokens: int = 0,
        reserve_for_response: int = 2000
    ) -> Tuple[List[Dict[str, str]], ContextUsage, Dict[str, Any]]:
        """
        Optimize message list for context window, applying sliding window if needed.
        
        Args:
            messages: Original message list
            search_results_tokens: Tokens used by search results
            reserve_for_response: Tokens to reserve for model response
            
        Returns:
            Tuple of (optimized_messages, context_usage, optimization_info)
        """
        # Initial analysis
        context_usage = self.analyze_context_usage(messages, search_results_tokens)
        optimization_info = {'sliding_applied': False}
        
        # Check if we need to apply sliding window
        total_with_response = context_usage.total_tokens + reserve_for_response
        available_space = self.context_window.max_tokens - search_results_tokens - reserve_for_response
        
        if total_with_response > self.context_window.max_tokens:
            # Apply sliding window
            system_tokens = context_usage.system_tokens
            current_tokens = context_usage.current_tokens
            
            # Calculate how much history we can keep
            target_history_tokens = available_space - system_tokens - current_tokens
            target_history_tokens = max(0, target_history_tokens)
            
            optimized_messages, sliding_info = self.apply_sliding_window(
                messages, target_history_tokens
            )
            
            # Update context usage after optimization
            context_usage = self.analyze_context_usage(optimized_messages, search_results_tokens)
            optimization_info.update(sliding_info)
            
            logger.info(f"Context optimized: {context_usage.usage_percentage:.1%} usage "
                       f"({context_usage.total_tokens}/{self.context_window.max_tokens} tokens)")
        else:
            optimized_messages = messages
        
        return optimized_messages, context_usage, optimization_info
    
    def get_context_recommendations(self, context_usage: ContextUsage) -> List[str]:
        """Get recommendations based on context usage."""
        recommendations = []
        
        if context_usage.warning_level == 'red':
            recommendations.extend([
                "🔴 Chat memory is very full (>85%). Consider starting a new conversation.",
                "💡 Some older messages may be automatically summarized to maintain performance.",
                "📝 Try more specific queries to reduce document retrieval size."
            ])
        elif context_usage.warning_level == 'yellow':
            recommendations.extend([
                "🟡 Chat memory is getting full (>75%). Monitor memory usage.",
                "💡 Consider starting a new chat if the conversation becomes unfocused.",
                "🔍 Use targeted queries for better efficiency."
            ])
        else:
            recommendations.append(
                "🟢 Chat memory is healthy. Plenty of room for conversation."
            )
        
        if context_usage.documents_tokens > 8000:
            recommendations.append(
                "📄 Large amount of document content in memory. Try more specific search terms."
            )
        
        if context_usage.history_tokens > 15000:
            recommendations.append(
                "💬 Long conversation history. Main points from earlier may be summarized automatically."
            )
        
        return recommendations