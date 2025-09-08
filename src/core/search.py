import logging
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from .database import DatabaseManager
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class SearchService:
    """Handles document search and context preparation."""
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.db_manager = db_manager or DatabaseManager()
        self.embedding_service = embedding_service or EmbeddingService()
    
    def search_similar(self, query: str, k: int = 5) -> Optional[pd.DataFrame]:
        """Search for similar documents."""
        try:
            embedding = self.embedding_service.embed_query(query)
            if embedding is None:
                return None
            
            results = self.db_manager.search(embedding, limit=k)
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            st.error(f"Search failed: {e}")
            return None
    
    def prepare_context(self, query: str, search_results: pd.DataFrame) -> Tuple[str, str, str]:
        """Prepare context for the LLM."""
        if search_results is None or search_results.empty:
            return None, None, None
        
        contexts = search_results.to_dict("records")
        bullets = [f"• {c['chunk'][:500]}" for c in contexts]
        cites = [f"[{i}] {c['source']}" for i, c in enumerate(contexts, 1)]
        
        system = (
            "You are a helpful assistant. Answer questions based on the provided context. "
            "If the context doesn't contain relevant information, say so."
        )
        
        user = f"Question: {query}\n\nContext:\n" + "\n".join(bullets)
        
        return system, user, "\n".join(cites)