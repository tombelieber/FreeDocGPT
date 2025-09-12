import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """Manages chat conversation history with persistence and search capabilities."""
    
    def __init__(self, history_dir: str = ".chat_history"):
        self.history_dir = Path(history_dir)
        self.conversations_dir = self.history_dir / "conversations"
        self.metadata_file = self.history_dir / "metadata.json"
        self.settings_file = self.history_dir / "settings.json"
        
        # Ensure directories exist
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata if not exists
        if not self.metadata_file.exists():
            self._initialize_metadata()
        
        # Initialize settings if not exists
        if not self.settings_file.exists():
            self._initialize_settings()
    
    def _initialize_metadata(self):
        """Initialize the metadata index file."""
        metadata = {
            "conversations": {},
            "created_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _initialize_settings(self):
        """Initialize user settings for chat history."""
        settings = {
            "auto_save_enabled": True,
            "auto_save_interval": 5,  # Save every 5 messages
            "max_conversations": 100,
            "default_name_strategy": "ai_summary",  # "first_message" or "ai_summary"
            "auto_cleanup_days": 90
        }
        with open(self.settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    
    def _load_metadata(self) -> Dict:
        """Load conversation metadata."""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            self._initialize_metadata()
            return self._load_metadata()
    
    def _save_metadata(self, metadata: Dict):
        """Save conversation metadata."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _generate_conversation_id(self) -> str:
        """Generate a unique conversation ID."""
        return str(uuid.uuid4())
    
    def _generate_auto_name(self, messages: List[Dict], strategy: str = "first_message") -> str:
        """Generate an automatic name for the conversation."""
        if not messages:
            return "Empty Conversation"
        
        # Find first user message
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            return "Empty Conversation"
        
        first_message = user_messages[0].get("content", "").strip()
        
        if strategy == "first_message":
            # Use first 50 characters of first message
            return first_message[:50] + ("..." if len(first_message) > 50 else "")
        elif strategy == "ai_summary":
            # For now, fallback to first message strategy
            # TODO: Implement AI summarization
            return first_message[:50] + ("..." if len(first_message) > 50 else "")
        else:
            return first_message[:50] + ("..." if len(first_message) > 50 else "")
    
    def _extract_keywords(self, messages: List[Dict]) -> List[str]:
        """Extract keywords from conversation for search indexing."""
        # Simple keyword extraction - can be enhanced later
        all_text = " ".join([
            msg.get("content", "") 
            for msg in messages 
            if msg.get("role") in ["user", "assistant"]
        ])
        
        # Basic keyword extraction (can be improved with NLP)
        words = all_text.lower().split()
        # Filter out common words and get unique keywords
        common_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", 
            "for", "of", "with", "by", "is", "are", "was", "were", "been", 
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "this", "that", "these",
            "those", "i", "you", "he", "she", "it", "we", "they", "my", "your",
            "his", "her", "its", "our", "their", "me", "him", "us", "them"
        }
        
        keywords = list(set([
            word.strip(".,!?;:\"'()[]{}") 
            for word in words 
            if len(word) > 2 and word.lower() not in common_words
        ]))[:20]  # Limit to 20 keywords
        
        return keywords
    
    def save_conversation(
        self, 
        messages: List[Dict], 
        conversation_name: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Save a conversation to history.
        
        Args:
            messages: List of conversation messages
            conversation_name: Optional custom name
            conversation_id: Optional existing conversation ID (for updates)
            metadata: Optional additional metadata
            
        Returns:
            The conversation ID
        """
        if not messages:
            logger.warning("Attempted to save empty conversation")
            return ""
        
        # Generate or use existing conversation ID
        if conversation_id is None:
            conversation_id = self._generate_conversation_id()
        
        # Load existing conversation to preserve created_at and other data
        existing_data = self.load_conversation(conversation_id) if conversation_id else None
        created_at = existing_data.get("created_at") if existing_data else datetime.now().isoformat()
        
        # Generate name if not provided
        if conversation_name is None:
            if existing_data and existing_data.get("name"):
                # Keep existing name if conversation already exists
                conversation_name = existing_data["name"]
            else:
                # Generate new name for new conversations
                settings = self._load_settings()
                strategy = settings.get("default_name_strategy", "first_message")
                conversation_name = self._generate_auto_name(messages, strategy)
        
        # Create conversation data
        conversation_data = {
            "id": conversation_id,
            "name": conversation_name,
            "messages": messages,
            "created_at": created_at,
            "updated_at": datetime.now().isoformat(),
            "message_count": len(messages),
            "keywords": self._extract_keywords(messages),
            "metadata": metadata or {},
            "pinned": existing_data.get("pinned", False) if existing_data else False,
            "archived": existing_data.get("archived", False) if existing_data else False,
        }
        
        # Save conversation file
        conversation_file = self.conversations_dir / f"conv_{conversation_id}.json"
        try:
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save conversation {conversation_id}: {e}")
            return ""
        
        # Update metadata index
        metadata_index = self._load_metadata()
        metadata_index["conversations"][conversation_id] = {
            "name": conversation_name,
            "created_at": created_at,
            "updated_at": conversation_data["updated_at"],
            "message_count": len(messages),
            "keywords": conversation_data["keywords"][:5],  # Store first 5 keywords in index
            "pinned": conversation_data["pinned"],
            "archived": conversation_data["archived"]
        }
        self._save_metadata(metadata_index)
        
        logger.info(f"Saved conversation '{conversation_name}' with ID: {conversation_id}")
        return conversation_id
    
    def auto_save_conversation(self, messages: List[Dict], conversation_id: Optional[str] = None) -> str:
        """Auto-save conversation (LobeHub/ChatGPT style).
        
        This creates a new conversation on first message or updates existing one.
        Always saves silently without user intervention.
        
        Args:
            messages: List of conversation messages
            conversation_id: Optional existing conversation ID
            
        Returns:
            The conversation ID
        """
        return self.save_conversation(
            messages=messages,
            conversation_id=conversation_id,
            # Don't pass conversation_name - let it auto-generate or keep existing
        )
    
    def load_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Load a conversation by ID.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Conversation data or None if not found
        """
        conversation_file = self.conversations_dir / f"conv_{conversation_id}.json"
        if not conversation_file.exists():
            logger.warning(f"Conversation not found: {conversation_id}")
            return None
        
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load conversation {conversation_id}: {e}")
            return None
    
    def list_conversations(self, limit: Optional[int] = None, include_archived: bool = False) -> List[Dict]:
        """List all conversations with metadata.
        
        Args:
            limit: Maximum number of conversations to return
            include_archived: Whether to include archived conversations
            
        Returns:
            List of conversation metadata, sorted by pinned then updated_at descending
        """
        metadata_index = self._load_metadata()
        conversations = []
        
        for conv_id, conv_meta in metadata_index["conversations"].items():
            # Skip archived conversations unless requested
            if not include_archived and conv_meta.get("archived", False):
                continue
                
            conversations.append({
                "id": conv_id,
                "name": conv_meta["name"],
                "created_at": conv_meta["created_at"],
                "updated_at": conv_meta["updated_at"],
                "message_count": conv_meta.get("message_count", 0),
                "keywords": conv_meta.get("keywords", []),
                "pinned": conv_meta.get("pinned", False),
                "archived": conv_meta.get("archived", False)
            })
        
        # Sort by pinned (pinned first) then by updated_at descending
        conversations.sort(key=lambda x: (not x["pinned"], x["updated_at"]), reverse=True)
        
        if limit:
            conversations = conversations[:limit]
        
        return conversations
    
    def rename_conversation(self, conversation_id: str, new_name: str) -> bool:
        """Rename a conversation.
        
        Args:
            conversation_id: The conversation ID
            new_name: New name for the conversation
            
        Returns:
            True if successful, False otherwise
        """
        # Load conversation data
        conversation_data = self.load_conversation(conversation_id)
        if not conversation_data:
            return False
        
        # Update name and save
        conversation_data["name"] = new_name
        conversation_data["updated_at"] = datetime.now().isoformat()
        
        conversation_file = self.conversations_dir / f"conv_{conversation_id}.json"
        try:
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to rename conversation {conversation_id}: {e}")
            return False
        
        # Update metadata index
        metadata_index = self._load_metadata()
        if conversation_id in metadata_index["conversations"]:
            metadata_index["conversations"][conversation_id]["name"] = new_name
            metadata_index["conversations"][conversation_id]["updated_at"] = conversation_data["updated_at"]
            self._save_metadata(metadata_index)
        
        logger.info(f"Renamed conversation {conversation_id} to '{new_name}'")
        return True
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            True if successful, False otherwise
        """
        conversation_file = self.conversations_dir / f"conv_{conversation_id}.json"
        
        try:
            # Delete file
            if conversation_file.exists():
                conversation_file.unlink()
            
            # Remove from metadata index
            metadata_index = self._load_metadata()
            if conversation_id in metadata_index["conversations"]:
                del metadata_index["conversations"][conversation_id]
                self._save_metadata(metadata_index)
            
            logger.info(f"Deleted conversation: {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete conversation {conversation_id}: {e}")
            return False
    
    def pin_conversation(self, conversation_id: str, pinned: bool = True) -> bool:
        """Pin or unpin a conversation.
        
        Args:
            conversation_id: The conversation ID
            pinned: True to pin, False to unpin
            
        Returns:
            True if successful, False otherwise
        """
        conversation_data = self.load_conversation(conversation_id)
        if not conversation_data:
            return False
        
        # Update pinned status
        conversation_data["pinned"] = pinned
        conversation_data["updated_at"] = datetime.now().isoformat()
        
        # Save conversation
        conversation_file = self.conversations_dir / f"conv_{conversation_id}.json"
        try:
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to pin/unpin conversation {conversation_id}: {e}")
            return False
        
        # Update metadata index
        metadata_index = self._load_metadata()
        if conversation_id in metadata_index["conversations"]:
            metadata_index["conversations"][conversation_id]["pinned"] = pinned
            metadata_index["conversations"][conversation_id]["updated_at"] = conversation_data["updated_at"]
            self._save_metadata(metadata_index)
        
        action = "Pinned" if pinned else "Unpinned"
        logger.info(f"{action} conversation: {conversation_id}")
        return True
    
    def archive_conversation(self, conversation_id: str, archived: bool = True) -> bool:
        """Archive or unarchive a conversation.
        
        Args:
            conversation_id: The conversation ID
            archived: True to archive, False to unarchive
            
        Returns:
            True if successful, False otherwise
        """
        conversation_data = self.load_conversation(conversation_id)
        if not conversation_data:
            return False
        
        # Update archived status
        conversation_data["archived"] = archived
        conversation_data["updated_at"] = datetime.now().isoformat()
        
        # Save conversation
        conversation_file = self.conversations_dir / f"conv_{conversation_id}.json"
        try:
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to archive/unarchive conversation {conversation_id}: {e}")
            return False
        
        # Update metadata index
        metadata_index = self._load_metadata()
        if conversation_id in metadata_index["conversations"]:
            metadata_index["conversations"][conversation_id]["archived"] = archived
            metadata_index["conversations"][conversation_id]["updated_at"] = conversation_data["updated_at"]
            self._save_metadata(metadata_index)
        
        action = "Archived" if archived else "Unarchived"
        logger.info(f"{action} conversation: {conversation_id}")
        return True
    
    def search_conversations(self, query: str, limit: int = 20) -> List[Dict]:
        """Search conversations by content or name.
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of matching conversations with relevance scores
        """
        query_lower = query.lower()
        results = []
        
        for conv_meta in self.list_conversations():
            conv_id = conv_meta["id"]
            conversation = self.load_conversation(conv_id)
            if not conversation:
                continue
            
            # Calculate relevance score
            score = 0
            
            # Name match (higher weight)
            if query_lower in conv_meta["name"].lower():
                score += 10
            
            # Keyword match (medium weight)
            for keyword in conv_meta.get("keywords", []):
                if query_lower in keyword.lower():
                    score += 3
            
            # Content match (lower weight but comprehensive)
            for message in conversation.get("messages", []):
                if query_lower in message.get("content", "").lower():
                    score += 1
            
            if score > 0:
                results.append({
                    **conv_meta,
                    "relevance_score": score
                })
        
        # Sort by relevance score descending
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return results[:limit]
    
    def get_conversation_summary(self, conversation_id: str) -> str:
        """Get a brief summary of a conversation."""
        conversation = self.load_conversation(conversation_id)
        if not conversation:
            return "Conversation not found"
        
        messages = conversation.get("messages", [])
        if not messages:
            return "Empty conversation"
        
        # Simple summary: first user message + message count
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if user_messages:
            first_msg = user_messages[0].get("content", "")[:100]
            return f"{first_msg}... ({len(messages)} messages)"
        
        return f"Conversation with {len(messages)} messages"
    
    def _load_settings(self) -> Dict:
        """Load chat history settings."""
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            self._initialize_settings()
            return self._load_settings()
    
    def update_settings(self, new_settings: Dict):
        """Update chat history settings."""
        current_settings = self._load_settings()
        current_settings.update(new_settings)
        
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(current_settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to update settings: {e}")
    
    def cleanup_old_conversations(self, days: int = 90) -> int:
        """Delete conversations older than specified days.
        
        Args:
            days: Delete conversations older than this many days
            
        Returns:
            Number of conversations deleted
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        for conv_meta in self.list_conversations():
            try:
                created_at = datetime.fromisoformat(conv_meta["created_at"])
                if created_at < cutoff_date:
                    if self.delete_conversation(conv_meta["id"]):
                        deleted_count += 1
            except Exception as e:
                logger.error(f"Error during cleanup of conversation {conv_meta['id']}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old conversations")
        return deleted_count
    
    def export_conversation(self, conversation_id: str, format: str = "markdown") -> Optional[str]:
        """Export a conversation to a formatted string.
        
        Args:
            conversation_id: The conversation ID
            format: Export format ("markdown" or "json")
            
        Returns:
            Formatted conversation string or None if error
        """
        conversation = self.load_conversation(conversation_id)
        if not conversation:
            return None
        
        if format == "json":
            return json.dumps(conversation, indent=2, ensure_ascii=False)
        
        elif format == "markdown":
            lines = [
                f"# {conversation['name']}",
                f"",
                f"**Created:** {conversation['created_at']}",
                f"**Updated:** {conversation['updated_at']}",
                f"**Messages:** {conversation['message_count']}",
                f"",
                "---",
                ""
            ]
            
            for message in conversation.get("messages", []):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                
                if role == "user":
                    lines.append(f"## 👤 User\n{content}\n")
                elif role == "assistant":
                    lines.append(f"## 🤖 Assistant\n{content}\n")
                else:
                    lines.append(f"## {role.title()}\n{content}\n")
            
            return "\n".join(lines)
        
        return None