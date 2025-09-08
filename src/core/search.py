import logging
from typing import Optional, Tuple, Dict

import pandas as pd
import streamlit as st

from .database import DatabaseManager
from .embeddings import EmbeddingService
from .hybrid_search import HybridSearch

logger = logging.getLogger(__name__)


class SearchService:
    """Handles document search and context preparation."""
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        use_hybrid: bool = True
    ):
        self.db_manager = db_manager or DatabaseManager()
        self.embedding_service = embedding_service or EmbeddingService()
        self.use_hybrid = use_hybrid
        
        # Initialize hybrid search if enabled
        if self.use_hybrid:
            try:
                self.hybrid_search = HybridSearch(
                    db_manager=self.db_manager,
                    embedding_service=self.embedding_service
                )
                logger.info("Hybrid search initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid search, falling back to vector-only: {e}")
                self.use_hybrid = False
                self.hybrid_search = None
        else:
            self.hybrid_search = None
    
    def search_similar(
        self,
        query: str,
        k: int = 5,
        search_mode: str = "hybrid",
        alpha: float = 0.5
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Search for similar documents using hybrid or vector search.
        
        Args:
            query: Search query
            k: Number of results
            search_mode: "hybrid", "vector", or "keyword"
            alpha: Weight for vector search in hybrid mode (0-1)
        
        Returns:
            Tuple of (results DataFrame, search statistics)
        """
        try:
            if self.use_hybrid and self.hybrid_search:
                # Use hybrid search
                results, stats = self.hybrid_search.search(
                    query, k, alpha, search_mode
                )
                return results, stats
            else:
                # Fallback to vector-only search
                embedding = self.embedding_service.embed_query(query)
                if embedding is None:
                    return None, {"error": "Failed to generate embedding"}
                
                results = self.db_manager.search(embedding, limit=k)
                stats = {
                    "mode": "vector",
                    "vector_results": len(results) if results is not None else 0
                }
                return results, stats
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            st.error(f"Search failed: {e}")
            return None, {"error": str(e)}
    
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