"""
Reranking models for improving search result relevance.
Uses cross-encoder models for more accurate relevance scoring.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from sentence_transformers import CrossEncoder
import numpy as np
import time

logger = logging.getLogger(__name__)


class Reranker:
    """Advanced reranking with cross-encoder models."""
    
    # Model recommendations for different use cases
    MODEL_RECOMMENDATIONS = {
        "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # 22M params, fastest
        "balanced": "cross-encoder/ms-marco-MiniLM-L-12-v2",  # 33M params
        "accurate": "BAAI/bge-reranker-base",  # 110M params, most accurate
        "multilingual": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # Multilingual support
    }
    
    def __init__(self, model_name: Optional[str] = None, use_onnx: bool = True):
        """
        Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model or preset (fast/balanced/accurate)
            use_onnx: Whether to use ONNX acceleration (if available)
        """
        # Resolve model name
        if model_name in self.MODEL_RECOMMENDATIONS:
            model_name = self.MODEL_RECOMMENDATIONS[model_name]
        elif model_name is None:
            model_name = self.MODEL_RECOMMENDATIONS["balanced"]
        
        self.model_name = model_name
        
        try:
            # Initialize cross-encoder
            self.model = CrossEncoder(
                model_name,
                max_length=512,
                device="cpu"  # M1 Mac will use Neural Engine via CoreML if available
            )
            logger.info(f"Initialized reranker with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            self.model = None
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_scores: bool = False,
        batch_size: int = 32
    ) -> List[Tuple[str, float]] | List[str]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top documents to return (None for all)
            return_scores: Whether to return scores with documents
            batch_size: Batch size for processing
            
        Returns:
            List of (document, score) tuples if return_scores=True, else list of documents
        """
        if not self.model or not documents:
            return documents if not return_scores else [(doc, 0.0) for doc in documents]
        
        start_time = time.time()
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Score in batches for better performance
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            batch_scores = self.model.predict(batch)
            all_scores.extend(batch_scores)
        
        scores = np.array(all_scores)
        
        # Sort by score (descending)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Apply top_k if specified
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
        
        # Prepare results
        if return_scores:
            results = [(documents[i], float(scores[i])) for i in sorted_indices]
        else:
            results = [documents[i] for i in sorted_indices]
        
        rerank_time = time.time() - start_time
        logger.debug(f"Reranked {len(documents)} documents in {rerank_time:.3f}s")
        
        return results
    
    def rerank_with_metadata(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        text_key: str = "chunk",
        top_k: Optional[int] = None,
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents with metadata preservation.
        
        Args:
            query: Search query
            documents: List of document dictionaries with metadata
            text_key: Key containing the text to rerank on
            top_k: Number of top documents to return
            batch_size: Batch size for processing
            
        Returns:
            List of reranked document dictionaries with added 'rerank_score'
        """
        if not self.model or not documents:
            return documents
        
        # Extract texts for reranking
        texts = [doc.get(text_key, "") for doc in documents]
        
        # Rerank and get scores
        reranked = self.rerank(query, texts, top_k=None, return_scores=True, batch_size=batch_size)
        
        # Create a mapping from text to score
        text_to_score = {text: score for text, score in reranked}
        
        # Add scores to documents and sort
        scored_docs = []
        for doc in documents:
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = text_to_score.get(doc.get(text_key, ""), 0.0)
            scored_docs.append(doc_copy)
        
        # Sort by rerank score
        scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Apply top_k if specified
        if top_k is not None:
            scored_docs = scored_docs[:top_k]
        
        return scored_docs
    
    def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        top_k: Optional[int] = None
    ) -> List[List[str]]:
        """
        Rerank multiple queries in batch.
        
        Args:
            queries: List of queries
            documents_list: List of document lists (one per query)
            top_k: Number of top documents per query
            
        Returns:
            List of reranked document lists
        """
        results = []
        for query, documents in zip(queries, documents_list):
            reranked = self.rerank(query, documents, top_k=top_k)
            results.append(reranked)
        return results


class HybridReranker:
    """Combines multiple reranking strategies."""
    
    def __init__(self, use_cross_encoder: bool = True, use_bm25: bool = False):
        """
        Initialize hybrid reranker.
        
        Args:
            use_cross_encoder: Whether to use cross-encoder reranking
            use_bm25: Whether to use BM25 reranking
        """
        self.cross_encoder = Reranker("balanced") if use_cross_encoder else None
        self.use_bm25 = use_bm25
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        initial_scores: Optional[List[float]] = None,
        weight_cross_encoder: float = 0.7,
        weight_initial: float = 0.3,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid reranking combining multiple signals.
        
        Args:
            query: Search query
            documents: Documents with metadata
            initial_scores: Initial search scores (e.g., from vector search)
            weight_cross_encoder: Weight for cross-encoder scores
            weight_initial: Weight for initial scores
            top_k: Number of top documents to return
            
        Returns:
            Reranked documents with combined scores
        """
        if not documents:
            return documents
        
        # Get cross-encoder scores if available
        if self.cross_encoder:
            documents = self.cross_encoder.rerank_with_metadata(
                query, documents, top_k=None
            )
            ce_scores = [doc.get('rerank_score', 0.0) for doc in documents]
        else:
            ce_scores = [0.0] * len(documents)
        
        # Normalize scores
        ce_scores = self._normalize_scores(ce_scores)
        
        if initial_scores:
            initial_scores = self._normalize_scores(initial_scores)
        else:
            initial_scores = [0.0] * len(documents)
        
        # Combine scores
        combined_scores = []
        for i, doc in enumerate(documents):
            score = (weight_cross_encoder * ce_scores[i] + 
                    weight_initial * initial_scores[i])
            doc['final_score'] = score
            combined_scores.append(score)
        
        # Sort by combined score
        sorted_indices = np.argsort(combined_scores)[::-1]
        
        # Apply top_k if specified
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
        
        return [documents[i] for i in sorted_indices]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return scores
        
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized.tolist()


class RerankerEvaluator:
    """Evaluate reranker performance."""
    
    @staticmethod
    def evaluate_latency(
        reranker: Reranker,
        query: str,
        documents: List[str],
        iterations: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate reranker latency.
        
        Args:
            reranker: Reranker instance
            query: Test query
            documents: Test documents
            iterations: Number of iterations for averaging
            
        Returns:
            Dictionary with latency statistics
        """
        latencies = []
        
        for _ in range(iterations):
            start = time.time()
            reranker.rerank(query, documents, top_k=5)
            latencies.append(time.time() - start)
        
        return {
            "mean_latency": np.mean(latencies),
            "std_latency": np.std(latencies),
            "min_latency": np.min(latencies),
            "max_latency": np.max(latencies),
            "p50_latency": np.percentile(latencies, 50),
            "p95_latency": np.percentile(latencies, 95)
        }
    
    @staticmethod
    def compare_models(
        query: str,
        documents: List[str],
        models: List[str] = ["fast", "balanced", "accurate"]
    ) -> Dict[str, Dict]:
        """
        Compare different reranker models.
        
        Args:
            query: Test query
            documents: Test documents
            models: List of model names to compare
            
        Returns:
            Comparison results
        """
        results = {}
        
        for model_name in models:
            reranker = Reranker(model_name)
            
            # Measure performance
            start = time.time()
            reranked = reranker.rerank(query, documents, top_k=5, return_scores=True)
            latency = time.time() - start
            
            # Get model info
            model_params = {
                "fast": "22M",
                "balanced": "33M",
                "accurate": "110M"
            }.get(model_name, "Unknown")
            
            results[model_name] = {
                "latency": latency,
                "model_params": model_params,
                "top_result": reranked[0] if reranked else None
            }
        
        return results