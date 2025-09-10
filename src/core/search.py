import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd
import streamlit as st

from .database import DatabaseManager
from .embeddings import EmbeddingService
from .cache import CachedEmbeddingService, get_shared_embedding_cache
from .hybrid_search import HybridSearch
from .reranker import Reranker, HybridReranker

logger = logging.getLogger(__name__)


class SearchService:
    """Handles document search and context preparation."""
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        use_hybrid: bool = True,
        use_reranking: bool = False,
        reranker_model: str = "balanced"
    ):
        self.db_manager = db_manager or DatabaseManager()
        base_embedding = embedding_service or EmbeddingService()
        # Enable embedding caching by default for query embeddings (shared cache instance)
        self.embedding_service = CachedEmbeddingService(base_embedding, cache=get_shared_embedding_cache())
        self.use_hybrid = use_hybrid
        self.use_reranking = use_reranking
        # Defer loading the system prompt until it is actually needed
        self._system_prompt: Optional[str] = None
        
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
        
        # Initialize reranker if enabled
        if self.use_reranking:
            try:
                self.reranker = Reranker(model_name=reranker_model)
                logger.info(f"Reranker initialized with model: {reranker_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
                self.reranker = None
                self.use_reranking = False
        else:
            self.reranker = None

    def _load_system_prompt(self) -> str:
        """Load system prompt from configured path, falling back to default."""
        default_prompt = (
            "You are a helpful assistant. Answer questions based on the provided context. "
            "If the context doesn't contain relevant information, say so."
        )
        try:
            from ..config import get_settings
            settings = get_settings()
            path_str = settings.system_prompt_path
            candidate = Path(path_str)
            if not candidate.is_absolute():
                # Resolve relative to repository root (two levels up from this file)
                repo_root = Path(__file__).resolve().parents[2]
                candidate = repo_root / candidate
            if candidate.exists() and candidate.is_file():
                content = candidate.read_text(encoding="utf-8").strip()
                if content:
                    logger.info(f"Loaded system prompt from {candidate}")
                    return content
                else:
                    logger.warning(f"System prompt file {candidate} is empty; using default prompt")
            else:
                logger.info(f"System prompt file not found at {candidate}; using default prompt")
        except Exception as e:
            logger.warning(f"Failed to load system prompt: {e}; using default prompt")
        return default_prompt

    def reload_system_prompt(self) -> None:
        """Reload system prompt from disk and update internal cache."""
        self._system_prompt = self._load_system_prompt()

    def _ensure_system_prompt(self) -> None:
        """Load the system prompt if it hasn't been loaded yet."""
        if self._system_prompt is None:
            self._system_prompt = self._load_system_prompt()

    def search_similar(
        self,
        query: str,
        k: int = 5,
        search_mode: str = "hybrid",
        alpha: float = 0.5,
        rerank_top_k: Optional[int] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Search for similar documents using hybrid or vector search with optional reranking.
        
        Args:
            query: Search query
            k: Number of results
            search_mode: "hybrid", "vector", or "keyword"
            alpha: Weight for vector search in hybrid mode (0-1)
            rerank_top_k: Number of results after reranking (None to keep all)
        
        Returns:
            Tuple of (results DataFrame, search statistics)
        """
        try:
            # Get initial search results
            if self.use_hybrid and self.hybrid_search:
                # Use hybrid search - get more results if reranking
                search_k = k * 3 if self.use_reranking else k
                results, stats = self.hybrid_search.search(
                    query, search_k, alpha, search_mode
                )
            else:
                # Fallback to vector-only search
                embedding = self.embedding_service.embed_query(query)
                if embedding is None:
                    return None, {"error": "Failed to generate embedding"}
                
                search_k = k * 3 if self.use_reranking else k
                results = self.db_manager.search(embedding, limit=search_k)
                stats = {
                    "mode": "vector",
                    "vector_results": len(results) if results is not None else 0
                }
            
            # Apply reranking if enabled
            if self.use_reranking and self.reranker and results is not None and not results.empty:
                import time
                rerank_start = time.time()
                
                # Convert DataFrame to list of dicts for reranking
                docs = results.to_dict('records')
                
                # Rerank documents
                reranked_docs = self.reranker.rerank_with_metadata(
                    query, 
                    docs, 
                    text_key='chunk',
                    top_k=rerank_top_k or k
                )
                
                # Convert back to DataFrame
                results = pd.DataFrame(reranked_docs)
                
                # Update stats
                rerank_time = time.time() - rerank_start
                stats['reranking'] = {
                    'enabled': True,
                    'model': self.reranker.model_name,
                    'time_ms': rerank_time * 1000,
                    'reranked_count': len(reranked_docs)
                }
            elif self.use_reranking:
                stats['reranking'] = {'enabled': False, 'reason': 'Reranker not available'}
            
            # Limit to requested k if not already done
            if results is not None and len(results) > k:
                results = results.head(k)
            
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
        
        # Ensure the prompt is loaded only when we actually need it
        self._ensure_system_prompt()
        system = self._system_prompt
        
        user = f"Question: {query}\n\nContext:\n" + "\n".join(bullets)
        
        return system, user, "\n".join(cites)
