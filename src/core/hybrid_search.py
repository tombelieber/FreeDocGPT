"""
Hybrid Search Implementation combining BM25 (Tantivy) and Vector Search (LanceDB)
Optimized for M1 Mac with ARM64 native support
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

import pandas as pd
import numpy as np
from tantivy import Index, Document, SchemaBuilder

from .database import DatabaseManager
from .embeddings import EmbeddingService
from ..config import get_settings

logger = logging.getLogger(__name__)


class TantivyIndex:
    """Tantivy-based BM25 search index."""
    
    def __init__(self, index_path: Optional[Path] = None):
        """Initialize Tantivy index."""
        self.settings = get_settings()
        
        # Use temp directory if no path provided
        if index_path is None:
            self.index_dir = Path(tempfile.mkdtemp(prefix="tantivy_"))
        else:
            self.index_dir = index_path
            self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.searcher = None
        self._create_index()
    
    def _create_index(self):
        """Create or load the Tantivy index."""
        try:
            # Define schema
            builder = SchemaBuilder()
            builder.add_text_field("doc_id", stored=True)
            builder.add_text_field("source", stored=True)
            builder.add_text_field("content", stored=True, tokenizer_name="default")
            builder.add_text_field("doc_type", stored=True)
            builder.add_text_field("language", stored=True)
            builder.add_unsigned_field("chunk_index", stored=True)
            
            schema = builder.build()
            
            # Create index
            index_path = str(self.index_dir)
            if (self.index_dir / "meta.json").exists():
                # Load existing index
                self.index = Index.open(index_path)
            else:
                # Create new index
                self.index = Index(schema, path=index_path)
            
            self.searcher = self.index.searcher()
            
        except Exception as e:
            logger.error(f"Failed to create Tantivy index: {e}")
            raise
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to the index."""
        writer = self.index.writer()
        
        try:
            for doc_data in documents:
                doc = Document()
                doc.add_text("doc_id", str(doc_data.get("id", "")))
                doc.add_text("source", doc_data.get("source", ""))
                doc.add_text("content", doc_data.get("chunk", ""))
                doc.add_text("doc_type", doc_data.get("doc_type", "general"))
                doc.add_text("language", doc_data.get("language", "english"))
                doc.add_unsigned("chunk_index", doc_data.get("chunk_index", 0))
                
                writer.add_document(doc)
            
            writer.commit()
            # Reload the index to see new documents
            self.index.reload()
            self.searcher = self.index.searcher()
            
        except Exception as e:
            logger.error(f"Failed to add documents to Tantivy: {e}")
            writer.rollback()
            raise
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search using BM25 scoring."""
        try:
            query_parser = self.index.parse_query(query, ["content", "source"])
            results = self.searcher.search(query_parser, limit)
            
            documents = []
            for score, doc_address in results.hits:
                doc = self.searcher.doc(doc_address)
                documents.append({
                    "doc_id": doc.get_first("doc_id"),
                    "source": doc.get_first("source"),
                    "content": doc.get_first("content"),
                    "score": score,
                    "doc_type": doc.get_first("doc_type"),
                    "language": doc.get_first("language"),
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Tantivy search failed: {e}")
            return []
    
    def clear(self):
        """Clear the index."""
        try:
            writer = self.index.writer()
            writer.delete_all_documents()
            writer.commit()
            self.searcher = self.index.searcher()
        except Exception as e:
            logger.error(f"Failed to clear Tantivy index: {e}")


class HybridSearch:
    """Hybrid search combining BM25 and vector search with RRF."""
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        tantivy_index: Optional[TantivyIndex] = None
    ):
        """Initialize hybrid search components."""
        self.db_manager = db_manager or DatabaseManager()
        self.embedding_service = embedding_service or EmbeddingService()
        self.tantivy_index = tantivy_index or TantivyIndex(
            Path(self.db_manager.settings.db_dir) / "tantivy_index"
        )
        self.settings = get_settings()
    
    def search(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.5,
        search_mode: str = "hybrid"
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform hybrid search with configurable modes.
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for vector search (0-1). 1 = pure vector, 0 = pure BM25
            search_mode: "hybrid", "vector", or "keyword"
        
        Returns:
            Tuple of (results DataFrame, search statistics)
        """
        stats = {"mode": search_mode, "query": query}
        
        try:
            if search_mode == "vector":
                # Vector-only search
                embedding = self.embedding_service.embed_query(query)
                if embedding is None:
                    return pd.DataFrame(), stats
                
                results = self.db_manager.search(embedding, limit=k)
                stats["vector_results"] = len(results) if results is not None else 0
                return results, stats
            
            elif search_mode == "keyword":
                # BM25-only search
                bm25_results = self.tantivy_index.search(query, k)
                stats["bm25_results"] = len(bm25_results)
                
                if not bm25_results:
                    return pd.DataFrame(), stats
                
                # Get full documents from database
                doc_ids = [r["doc_id"] for r in bm25_results]
                return self._get_documents_by_ids(doc_ids), stats
            
            else:  # hybrid
                # Get both result sets
                bm25_results = self.tantivy_index.search(query, k * 2)
                
                embedding = self.embedding_service.embed_query(query)
                vector_results = None
                if embedding is not None:
                    vector_results = self.db_manager.search(embedding, limit=k * 2)
                
                # Combine with RRF
                combined_results = self._reciprocal_rank_fusion(
                    bm25_results,
                    vector_results,
                    alpha,
                    k
                )
                
                stats["bm25_results"] = len(bm25_results)
                stats["vector_results"] = len(vector_results) if vector_results is not None else 0
                stats["combined_results"] = len(combined_results)
                stats["alpha"] = alpha
                
                return combined_results, stats
                
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            stats["error"] = str(e)
            return pd.DataFrame(), stats
    
    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict],
        vector_results: Optional[pd.DataFrame],
        alpha: float,
        k: int
    ) -> pd.DataFrame:
        """
        Combine BM25 and vector results using Reciprocal Rank Fusion.
        
        RRF score = Σ (1 / (k + rank_i))
        where k is a constant (typically 60) and rank_i is the rank in each list
        """
        rrf_k = 60  # Standard RRF constant
        scores = {}
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results, 1):
            doc_id = result["doc_id"]
            bm25_score = 1.0 / (rrf_k + rank)
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) * bm25_score
        
        # Process vector results
        if vector_results is not None and not vector_results.empty:
            for rank, row in enumerate(vector_results.itertuples(), 1):
                doc_id = str(row.id)
                vector_score = 1.0 / (rrf_k + rank)
                scores[doc_id] = scores.get(doc_id, 0) + alpha * vector_score
        
        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
        
        # Get full documents
        return self._get_documents_by_ids(sorted_ids)
    
    def _get_documents_by_ids(self, doc_ids: List[str]) -> pd.DataFrame:
        """Retrieve full documents from database by IDs."""
        try:
            table = self.db_manager.get_table()
            
            # Build filter expression
            id_list = [int(doc_id) for doc_id in doc_ids if doc_id.isdigit()]
            if not id_list:
                return pd.DataFrame()
            
            # Query the table
            results = table.to_pandas()
            filtered = results[results['id'].isin(id_list)]
            
            # Sort by the order of doc_ids
            id_order = {int(doc_id): i for i, doc_id in enumerate(doc_ids) if doc_id.isdigit()}
            filtered['order'] = filtered['id'].map(id_order)
            filtered = filtered.sort_values('order').drop('order', axis=1)
            
            return filtered
            
        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {e}")
            return pd.DataFrame()
    
    def index_documents(self, documents: List[Dict]):
        """Index documents in both Tantivy and vector database."""
        try:
            # Add to Tantivy for BM25 search
            self.tantivy_index.add_documents(documents)
            logger.info(f"Indexed {len(documents)} documents in Tantivy")
            
        except Exception as e:
            logger.error(f"Failed to index documents in hybrid search: {e}")
            raise
    
    def clear_index(self):
        """Clear both indexes."""
        try:
            self.tantivy_index.clear()
            # Vector index is cleared through db_manager.clear_index()
            logger.info("Cleared Tantivy index")
        except Exception as e:
            logger.error(f"Failed to clear hybrid search index: {e}")