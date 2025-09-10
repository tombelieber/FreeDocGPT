"""
Caching strategy for embeddings and search results.
Uses diskcache for persistent caching with size limits.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import xxhash
import pickle
import time
from diskcache import Cache
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for document embeddings to avoid re-computation."""
    
    def __init__(
        self,
        cache_dir: str = ".cache/embeddings",
        size_limit: int = 2**30,  # 1GB default
        eviction_policy: str = "least-recently-used"
    ):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory for cache storage
            size_limit: Maximum cache size in bytes
            eviction_policy: Cache eviction policy
        """
        self.cache = Cache(
            cache_dir,
            size_limit=size_limit,
            eviction_policy=eviction_policy
        )
        self.hits = 0
        self.misses = 0
        logger.info(f"Initialized embedding cache at {cache_dir} with {size_limit/1e9:.2f}GB limit")
    
    def _generate_key(self, text: str, model: str = "default") -> str:
        """
        Generate cache key for text.
        
        Args:
            text: Text to generate key for
            model: Model name for namespacing
            
        Returns:
            Cache key
        """
        # Use xxhash for fast hashing
        content = f"{model}:{text}"
        return xxhash.xxh64(content.encode()).hexdigest()
    
    def get(self, text: str, model: str = "default") -> Optional[List[float]]:
        """
        Get embedding from cache.
        
        Args:
            text: Text to get embedding for
            model: Model name
            
        Returns:
            Embedding if cached, None otherwise
        """
        key = self._generate_key(text, model)
        
        try:
            embedding = self.cache.get(key)
            if embedding is not None:
                self.hits += 1
                logger.debug(f"Cache hit for text (length={len(text)})")
                return embedding
            else:
                self.misses += 1
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.misses += 1
            return None
    
    def set(self, text: str, embedding: List[float], model: str = "default", ttl: Optional[int] = None):
        """
        Store embedding in cache.
        
        Args:
            text: Text that was embedded
            embedding: Embedding vector
            model: Model name
            ttl: Time to live in seconds (None for no expiration)
        """
        key = self._generate_key(text, model)
        
        try:
            self.cache.set(key, embedding, expire=ttl)
            logger.debug(f"Cached embedding for text (length={len(text)})")
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def get_batch(self, texts: List[str], model: str = "default") -> Tuple[Dict[int, List[float]], List[int]]:
        """
        Get multiple embeddings from cache.
        
        Args:
            texts: List of texts
            model: Model name
            
        Returns:
            Tuple of (cached embeddings dict, indices of cache misses)
        """
        cached = {}
        missing_indices = []
        
        for i, text in enumerate(texts):
            embedding = self.get(text, model)
            if embedding is not None:
                cached[i] = embedding
            else:
                missing_indices.append(i)
        
        return cached, missing_indices
    
    def set_batch(self, texts: List[str], embeddings: List[List[float]], model: str = "default"):
        """
        Store multiple embeddings in cache.
        
        Args:
            texts: List of texts
            embeddings: List of embeddings
            model: Model name
        """
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding, model)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size_bytes": self.cache.volume(),
            "item_count": len(self.cache)
        }
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Embedding cache cleared")
    
    def close(self):
        """Close the cache."""
        self.cache.close()


# Shared cache instance to avoid duplicate in-memory caches and logs
_shared_embedding_cache: Optional[EmbeddingCache] = None


def get_shared_embedding_cache() -> EmbeddingCache:
    """Return a process-wide shared EmbeddingCache instance."""
    global _shared_embedding_cache
    if _shared_embedding_cache is None:
        _shared_embedding_cache = EmbeddingCache()
    return _shared_embedding_cache


class SearchResultCache:
    """Cache for search results to speed up repeated queries."""
    
    def __init__(
        self,
        cache_dir: str = ".cache/search",
        size_limit: int = 2**29,  # 512MB default
        ttl: int = 3600  # 1 hour default TTL
    ):
        """
        Initialize search result cache.
        
        Args:
            cache_dir: Directory for cache storage
            size_limit: Maximum cache size in bytes
            ttl: Default time to live in seconds
        """
        self.cache = Cache(cache_dir, size_limit=size_limit)
        self.default_ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def _generate_key(
        self,
        query: str,
        k: int,
        search_mode: str,
        alpha: float
    ) -> str:
        """Generate cache key for search query."""
        content = f"{query}:{k}:{search_mode}:{alpha}"
        return xxhash.xxh64(content.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        k: int,
        search_mode: str,
        alpha: float
    ) -> Optional[Tuple[Any, Dict]]:
        """
        Get search results from cache.
        
        Args:
            query: Search query
            k: Number of results
            search_mode: Search mode
            alpha: Hybrid search weight
            
        Returns:
            Cached results if available
        """
        key = self._generate_key(query, k, search_mode, alpha)
        
        try:
            result = self.cache.get(key)
            if result is not None:
                self.hits += 1
                logger.debug(f"Search cache hit for query: {query[:50]}...")
                return result
            else:
                self.misses += 1
                return None
        except Exception as e:
            logger.error(f"Search cache get error: {e}")
            self.misses += 1
            return None
    
    def set(
        self,
        query: str,
        k: int,
        search_mode: str,
        alpha: float,
        results: Tuple[Any, Dict],
        ttl: Optional[int] = None
    ):
        """
        Cache search results.
        
        Args:
            query: Search query
            k: Number of results
            search_mode: Search mode
            alpha: Hybrid search weight
            results: Search results to cache
            ttl: Time to live (uses default if None)
        """
        key = self._generate_key(query, k, search_mode, alpha)
        expire = ttl or self.default_ttl
        
        try:
            self.cache.set(key, results, expire=expire)
            logger.debug(f"Cached search results for query: {query[:50]}...")
        except Exception as e:
            logger.error(f"Search cache set error: {e}")
    
    def invalidate_pattern(self, pattern: str):
        """
        Invalidate cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match (uses query prefix)
        """
        # Since diskcache doesn't support pattern matching directly,
        # we need to iterate through keys
        keys_to_delete = []
        for key in self.cache:
            if pattern in str(key):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.cache[key]
        
        logger.info(f"Invalidated {len(keys_to_delete)} cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size_bytes": self.cache.volume(),
            "item_count": len(self.cache)
        }
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Search cache cleared")


class CachedEmbeddingService:
    """Embedding service with caching layer."""
    
    def __init__(
        self,
        embedding_service,
        cache: Optional[EmbeddingCache] = None
    ):
        """
        Initialize cached embedding service.
        
        Args:
            embedding_service: Base embedding service
            cache: Embedding cache (creates default if None)
        """
        self.embedding_service = embedding_service
        self.cache = cache or EmbeddingCache()
        self.model_name = getattr(embedding_service, 'model_name', 'default')
    
    def embed_query(self, query: str) -> Optional[List[float]]:
        """
        Generate embedding for query with caching.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        # Check cache first
        embedding = self.cache.get(query, self.model_name)
        if embedding is not None:
            return embedding
        
        # Generate embedding
        embedding = self.embedding_service.embed_query(query)
        
        # Cache the result
        if embedding is not None:
            self.cache.set(query, embedding, self.model_name)
        
        return embedding
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with caching.
        
        Args:
            texts: List of texts
            
        Returns:
            List of embeddings
        """
        # Check cache for all texts
        cached_embeddings, missing_indices = self.cache.get_batch(texts, self.model_name)
        
        # If all are cached, return them
        if not missing_indices:
            return [cached_embeddings[i] for i in range(len(texts))]
        
        # Generate embeddings for missing texts
        missing_texts = [texts[i] for i in missing_indices]
        new_embeddings = self.embedding_service.embed_texts(missing_texts)
        
        # Cache new embeddings
        if new_embeddings:
            self.cache.set_batch(missing_texts, new_embeddings, self.model_name)
        
        # Combine cached and new embeddings
        result = []
        new_idx = 0
        for i in range(len(texts)):
            if i in cached_embeddings:
                result.append(cached_embeddings[i])
            else:
                if new_idx < len(new_embeddings):
                    result.append(new_embeddings[new_idx])
                    new_idx += 1
                else:
                    result.append(None)
        
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()


class CachedSearchService:
    """Search service with result caching."""
    
    def __init__(
        self,
        search_service,
        cache: Optional[SearchResultCache] = None
    ):
        """
        Initialize cached search service.
        
        Args:
            search_service: Base search service
            cache: Search result cache (creates default if None)
        """
        self.search_service = search_service
        self.cache = cache or SearchResultCache()
    
    def search_similar(
        self,
        query: str,
        k: int = 5,
        search_mode: str = "hybrid",
        alpha: float = 0.5,
        use_cache: bool = True
    ) -> Tuple[Any, Dict]:
        """
        Search with caching.
        
        Args:
            query: Search query
            k: Number of results
            search_mode: Search mode
            alpha: Hybrid weight
            use_cache: Whether to use cache
            
        Returns:
            Search results and statistics
        """
        if use_cache:
            # Check cache first
            cached = self.cache.get(query, k, search_mode, alpha)
            if cached is not None:
                results, stats = cached
                stats['cached'] = True
                return results, stats
        
        # Perform search
        results, stats = self.search_service.search_similar(
            query, k, search_mode, alpha
        )
        
        # Cache results
        if use_cache and results is not None:
            self.cache.set(query, k, search_mode, alpha, (results, stats))
        
        stats['cached'] = False
        return results, stats
    
    def invalidate_cache(self):
        """Invalidate all cached results."""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


class CacheManager:
    """Manage all caching layers."""
    
    def __init__(
        self,
        cache_dir: str = ".cache",
        embedding_cache_size: int = 2**30,  # 1GB
        search_cache_size: int = 2**29,  # 512MB
        enable_embedding_cache: bool = True,
        enable_search_cache: bool = True
    ):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Base cache directory
            embedding_cache_size: Size limit for embedding cache
            search_cache_size: Size limit for search cache
            enable_embedding_cache: Whether to enable embedding caching
            enable_search_cache: Whether to enable search caching
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.embedding_cache = None
        self.search_cache = None
        
        if enable_embedding_cache:
            self.embedding_cache = EmbeddingCache(
                str(self.cache_dir / "embeddings"),
                embedding_cache_size
            )
        
        if enable_search_cache:
            self.search_cache = SearchResultCache(
                str(self.cache_dir / "search"),
                search_cache_size
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {}
        
        if self.embedding_cache:
            stats['embedding_cache'] = self.embedding_cache.get_stats()
        
        if self.search_cache:
            stats['search_cache'] = self.search_cache.get_stats()
        
        return stats
    
    def clear_all(self):
        """Clear all caches."""
        if self.embedding_cache:
            self.embedding_cache.clear()
        
        if self.search_cache:
            self.search_cache.clear()
        
        logger.info("All caches cleared")
    
    def close(self):
        """Close all caches."""
        if self.embedding_cache:
            self.embedding_cache.close()
        
        if self.search_cache:
            self.search_cache.cache.close()
