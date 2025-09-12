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
from .query_analyzer import QueryAnalyzer
from .clustering import DocumentClusterer

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
        # Track current locale for prompt synchronization
        self._current_locale: Optional[str] = None
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
        
        # Initialize query analyzer and clusterer for smart search
        self.query_analyzer = QueryAnalyzer()
        self.clusterer = None  # Will be loaded from indexed data when needed

    def _load_system_prompt(self) -> str:
        """Load system prompt based on current UI language, falling back to default."""
        default_prompt = (
            "You are a helpful assistant. Answer questions based on the provided context. "
            "If the context doesn't contain relevant information, say so."
        )
        try:
            from ..config import get_settings
            from ..ui.i18n import get_locale
            
            settings = get_settings()
            current_locale = get_locale()
            
            # Map locales to prompt file names
            prompt_files = {
                "en": "rag_prompt_en.md",
                "zh-Hant": "rag_prompt_zh_hant.md", 
                "zh-Hans": "rag_prompt_zh_hans.md",
                "es": "rag_prompt_es.md",
                "ja": "rag_prompt_ja.md"
            }
            
            # Get the appropriate prompt file for current language
            prompt_filename = prompt_files.get(current_locale, "rag_prompt_en.md")
            
            # Try language-specific prompt first
            repo_root = Path(__file__).resolve().parents[2]
            candidate = repo_root / prompt_filename
            
            if candidate.exists() and candidate.is_file():
                content = candidate.read_text(encoding="utf-8").strip()
                if content:
                    logger.info(f"Loaded language-specific system prompt from {candidate} for locale {current_locale}")
                    return content
                else:
                    logger.warning(f"Language-specific prompt file {candidate} is empty")
            else:
                logger.info(f"Language-specific prompt file not found at {candidate}")
            
            # Fallback to configured path (backward compatibility)
            path_str = settings.system_prompt_path
            fallback_candidate = Path(path_str)
            if not fallback_candidate.is_absolute():
                fallback_candidate = repo_root / fallback_candidate
                
            if fallback_candidate.exists() and fallback_candidate.is_file():
                content = fallback_candidate.read_text(encoding="utf-8").strip()
                if content:
                    logger.info(f"Loaded fallback system prompt from {fallback_candidate}")
                    return content
                else:
                    logger.warning(f"Fallback prompt file {fallback_candidate} is empty; using default prompt")
            else:
                logger.info(f"Fallback prompt file not found at {fallback_candidate}; using default prompt")
                
        except Exception as e:
            logger.warning(f"Failed to load system prompt: {e}; using default prompt")
        return default_prompt

    def reload_system_prompt(self) -> None:
        """Reload system prompt from disk and update internal cache."""
        # Force cache invalidation first to ensure fresh load
        self._system_prompt = None
        self._current_locale = None
        # Use _ensure_system_prompt to properly set both prompt and locale
        self._ensure_system_prompt()
    
    def invalidate_prompt_cache(self) -> None:
        """Invalidate the cached prompt, forcing reload on next use."""
        self._system_prompt = None
        self._current_locale = None

    def _ensure_system_prompt(self) -> None:
        """Load the system prompt if it hasn't been loaded yet or locale changed."""
        try:
            from ..ui.i18n import get_locale
            current_locale = get_locale()
            
            # Reload prompt if locale changed or not loaded yet
            if self._system_prompt is None or self._current_locale != current_locale:
                self._current_locale = current_locale
                self._system_prompt = self._load_system_prompt()
        except Exception as e:
            # Fallback if locale detection fails
            if self._system_prompt is None:
                self._system_prompt = self._load_system_prompt()

    def search_similar(
        self,
        query: str,
        k: int = 5,
        search_mode: str = "hybrid",
        alpha: float = 0.5,
        rerank_top_k: Optional[int] = None,
        use_smart_filtering: bool = True
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Search for similar documents using hybrid or vector search with optional reranking.
        
        Args:
            query: Search query
            k: Number of results
            search_mode: "hybrid", "vector", or "keyword"
            alpha: Weight for vector search in hybrid mode (0-1)
            rerank_top_k: Number of results after reranking (None to keep all)
            use_smart_filtering: Enable smart cluster/metadata filtering
        
        Returns:
            Tuple of (results DataFrame, search statistics)
        """
        try:
            # Analyze query for smart filtering
            query_analysis = None
            filters = {}
            
            if use_smart_filtering:
                query_analysis = self.query_analyzer.analyze_query(query)
                logger.info(f"Query analysis: {query_analysis}")
                
                # Build filters based on analysis
                # Use doc_type_hint for soft filtering/boosting
                if query_analysis.get('doc_type_hint'):
                    filters['doc_type_hint'] = query_analysis['doc_type_hint']
                
                # Use temporal hint for sorting
                if query_analysis.get('temporal_hint'):
                    filters['temporal_hint'] = query_analysis['temporal_hint']
                
                # Use keywords for boosting
                if query_analysis.get('keywords'):
                    filters['keywords'] = query_analysis['keywords']
            # Stage 1: Focused search with smart filtering
            results = None
            stats = {"query_analysis": query_analysis} if query_analysis else {}
            
            if use_smart_filtering and filters:
                # Try focused search first
                results, focused_stats = self._smart_filtered_search(
                    query, k, search_mode, alpha, filters
                )
                stats.update(focused_stats)
                
                # Check if we need to expand search
                if results is not None and not results.empty:
                    # Calculate result quality (using score if available)
                    avg_score = results['_distance'].mean() if '_distance' in results.columns else 0.5
                    quality = 1.0 - avg_score  # Convert distance to quality
                    
                    if self.query_analyzer.should_expand_search(quality, threshold=0.3):
                        logger.info("Expanding search due to low quality results")
                        # Fall through to broad search
                        results = None
                        stats['search_expanded'] = True
            
            # Stage 2: Broad search (if needed or if smart filtering disabled)
            if results is None or results.empty:
                # Get initial search results
                if self.use_hybrid and self.hybrid_search:
                    # Use hybrid search - get more results if reranking
                    search_k = k * 3 if self.use_reranking else k
                    results, broad_stats = self.hybrid_search.search(
                        query, search_k, alpha, search_mode
                    )
                    stats.update(broad_stats)
                else:
                    # Fallback to vector-only search
                    embedding = self.embedding_service.embed_query(query)
                    if embedding is None:
                        return None, {"error": "Failed to generate embedding"}
                    
                    search_k = k * 3 if self.use_reranking else k
                    results = self.db_manager.search(embedding, limit=search_k)
                    stats.update({
                        "mode": "vector",
                        "vector_results": len(results) if results is not None else 0
                    })
            
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
        
        # Group chunks by source document for cleaner citation display
        from collections import OrderedDict
        doc_chunks = OrderedDict()
        for i, c in enumerate(contexts, 1):
            source = c['source']
            if source not in doc_chunks:
                doc_chunks[source] = []
            doc_chunks[source].append(i)
        
        # Format citations to show documents with their chunk counts
        citation_lines = []
        for doc_num, (doc_name, chunk_nums) in enumerate(doc_chunks.items(), 1):
            if len(chunk_nums) == 1:
                citation_lines.append(f"[{doc_num}] **{doc_name}** (1 relevant section)")
            else:
                citation_lines.append(f"[{doc_num}] **{doc_name}** ({len(chunk_nums)} relevant sections)")
        
        # Ensure the prompt is loaded only when we actually need it
        self._ensure_system_prompt()
        system = self._system_prompt
        
        user = f"Question: {query}\n\nContext:\n" + "\n".join(bullets)
        
        return system, user, "\n".join(citation_lines)
    
    def _smart_filtered_search(
        self,
        query: str,
        k: int,
        search_mode: str,
        alpha: float,
        filters: Dict
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Perform search with smart filtering based on clusters and metadata.
        
        Args:
            query: Search query
            k: Number of results
            search_mode: Search mode
            alpha: Hybrid search weight
            filters: Filter criteria
        
        Returns:
            Tuple of (filtered results, statistics)
        """
        import numpy as np
        stats = {"filtering": "enabled", "filters_applied": filters}
        
        try:
            # Get query embedding
            query_embedding = self.embedding_service.embed_query(query)
            if query_embedding is None:
                return None, {"error": "Failed to generate query embedding"}
            
            # Load cluster information if available
            table = self.db_manager.get_table()
            all_data = table.to_pandas()
            
            if all_data.empty:
                return None, {"error": "No indexed documents"}
            
            # Find relevant clusters
            cluster_filter = None
            if 'cluster_id' in all_data.columns and len(all_data['cluster_id'].unique()) > 1:
                # Initialize clusterer with saved cluster centers if needed
                if self.clusterer is None:
                    self.clusterer = DocumentClusterer()
                    # Get unique cluster centers from data
                    cluster_centers = []
                    for cluster_id in sorted(all_data['cluster_id'].unique()):
                        cluster_data = all_data[all_data['cluster_id'] == cluster_id]
                        if not cluster_data.empty:
                            # Use first vector as approximation of center
                            center = np.array(cluster_data.iloc[0]['vector'])
                            cluster_centers.append(center)
                    
                    if cluster_centers:
                        self.clusterer.cluster_centers = np.array(cluster_centers)
                        self.clusterer.n_clusters = len(cluster_centers)
                
                # Find nearest clusters
                if self.clusterer and self.clusterer.cluster_centers is not None:
                    query_embedding_np = np.array(query_embedding)
                    nearest_clusters = self.clusterer.find_nearest_clusters(
                        query_embedding_np, top_k=2
                    )
                    cluster_filter = nearest_clusters
                    stats['clusters_searched'] = nearest_clusters
            
            # Apply filters to create search subset
            filtered_data = all_data.copy()  # Create a copy to avoid SettingWithCopyWarning
            
            # Filter by clusters
            if cluster_filter is not None:
                filtered_data = filtered_data[filtered_data['cluster_id'].isin(cluster_filter)]
                stats['cluster_filtered_count'] = len(filtered_data)
            
            # Apply soft filtering/boosting based on doc type hint
            if filters.get('doc_type_hint') and 'doc_type' in filtered_data.columns:
                # Boost documents matching the doc type hint
                doc_type_hint = filters['doc_type_hint']
                # Priority order: exact match > source name match > partial match
                if doc_type_hint == 'meeting':
                    # For meeting queries, strongly prioritize files with "meeting" in the name
                    filtered_data['boost_score'] = filtered_data.apply(
                        lambda row: (
                            5.0 if 'meeting' in str(row.get('source', '')).lower() else
                            3.0 if row.get('doc_type') == 'meeting' else
                            1.0 if 'meeting' in str(row.get('chunk', '')).lower()[:200] else
                            0.0
                        ), axis=1
                    )
                else:
                    filtered_data['boost_score'] = filtered_data['doc_type'].apply(
                        lambda x: 2.0 if x == doc_type_hint else 0.0
                    )
                stats['doc_type_boost'] = doc_type_hint
            else:
                filtered_data['boost_score'] = 0.0
            
            # Apply temporal sorting if requested
            if filters.get('temporal_hint') == 'recent' and 'file_modified' in filtered_data.columns:
                # Sort by modification time for recent documents
                filtered_data = filtered_data.sort_values('file_modified', ascending=False)
                stats['temporal_sort'] = 'recent'
            
            # Boost by keywords in content or metadata
            if filters.get('keywords'):
                keyword_boost = 0.0
                for keyword in filters['keywords']:
                    keyword_lower = keyword.lower()
                    # Check in source name (highest boost)
                    filtered_data['boost_score'] += filtered_data['source'].apply(
                        lambda x: 1.5 if keyword_lower in str(x).lower() else 0.0
                    )
                    # Check in keywords field
                    if 'doc_keywords' in filtered_data.columns:
                        filtered_data['boost_score'] += filtered_data['doc_keywords'].apply(
                            lambda x: 1.0 if pd.notna(x) and keyword_lower in str(x).lower() else 0.0
                        )
                    # Check in chunk content (lower boost)
                    filtered_data['boost_score'] += filtered_data['chunk'].apply(
                        lambda x: 0.5 if keyword_lower in str(x).lower()[:500] else 0.0
                    )
                stats['keyword_boost'] = filters['keywords']
            
            if filtered_data.empty:
                logger.info("No documents match filters, returning None")
                return None, stats
            
            # Perform search on filtered subset
            if self.use_hybrid and self.hybrid_search:
                # For hybrid search, we need to pass the filtered IDs
                filtered_ids = filtered_data['id'].tolist()
                search_k = min(k * 2, len(filtered_data))  # Don't request more than available
                
                # Perform hybrid search with ID filter
                results, search_stats = self.hybrid_search.search(
                    query, search_k, alpha, search_mode,
                    filter_ids=filtered_ids  # Pass filtered IDs
                )
                stats.update(search_stats)
            else:
                # Vector search on filtered data
                filtered_vectors = np.array(filtered_data['vector'].tolist())
                query_vec = np.array(query_embedding)
                
                # Compute cosine similarities
                similarities = np.dot(filtered_vectors, query_vec) / (
                    np.linalg.norm(filtered_vectors, axis=1) * np.linalg.norm(query_vec)
                )
                
                # Combine similarity with boost scores
                boost_scores = filtered_data['boost_score'].values if 'boost_score' in filtered_data.columns else np.zeros(len(filtered_data))
                # Normalize boost scores to 0-0.3 range to not overwhelm similarity
                max_boost = max(boost_scores.max(), 1.0)
                normalized_boosts = (boost_scores / max_boost) * 0.3
                final_scores = similarities + normalized_boosts
                
                # Get top k results
                top_k_indices = np.argsort(final_scores)[::-1][:k]
                results = filtered_data.iloc[top_k_indices].copy()
                results['_distance'] = 1 - similarities[top_k_indices]  # Keep original distance
                results['_score'] = final_scores[top_k_indices]  # Add combined score
                
                stats['mode'] = 'vector_filtered'
                stats['filtered_search_space'] = len(filtered_data)
            
            stats['results_count'] = len(results) if results is not None else 0
            return results, stats
            
        except Exception as e:
            logger.error(f"Smart filtered search failed: {e}")
            return None, {"error": str(e)}
