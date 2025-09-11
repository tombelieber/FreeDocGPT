"""
Simple clustering module for document grouping.
Uses K-means for lightweight semantic clustering of documents.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class DocumentClusterer:
    """Handles document clustering for smart search optimization."""
    
    def __init__(self, n_clusters: Optional[int] = None, min_clusters: int = 3, max_clusters: int = 10):
        """
        Initialize the document clusterer.
        
        Args:
            n_clusters: Fixed number of clusters (if None, auto-detect)
            min_clusters: Minimum clusters for auto-detection
            max_clusters: Maximum clusters for auto-detection
        """
        self.n_clusters = n_clusters
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.kmeans = None
        self.cluster_centers = None
        
    def determine_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """
        Determine optimal number of clusters using silhouette score.
        
        Args:
            embeddings: Document embeddings array
            
        Returns:
            Optimal number of clusters
        """
        n_samples = len(embeddings)
        
        # Adjust range based on number of documents
        min_k = min(self.min_clusters, n_samples - 1)
        max_k = min(self.max_clusters, n_samples - 1)
        
        # For small datasets, use fewer clusters
        if n_samples < 20:
            return min(3, n_samples - 1)
        elif n_samples < 50:
            return 5
        
        # For larger datasets, find optimal k using silhouette score
        best_score = -1
        best_k = min_k
        
        for k in range(min_k, min(max_k + 1, n_samples)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                logger.warning(f"Failed to compute silhouette score for k={k}: {e}")
                
        logger.info(f"Optimal clusters determined: {best_k} (silhouette score: {best_score:.3f})")
        return best_k
    
    def fit(self, embeddings: np.ndarray) -> 'DocumentClusterer':
        """
        Fit the clustering model on document embeddings.
        
        Args:
            embeddings: Array of document embeddings
            
        Returns:
            Self for chaining
        """
        if len(embeddings) < 2:
            logger.warning("Not enough documents for clustering (need at least 2)")
            return self
            
        # Determine number of clusters
        if self.n_clusters is None:
            self.n_clusters = self.determine_optimal_clusters(embeddings)
        else:
            # Ensure n_clusters doesn't exceed number of samples
            self.n_clusters = min(self.n_clusters, len(embeddings) - 1)
        
        # Fit K-means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        self.kmeans.fit(embeddings)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        logger.info(f"Clustering complete: {self.n_clusters} clusters created")
        return self
    
    def predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict cluster assignments and distances for embeddings.
        
        Args:
            embeddings: Array of embeddings to assign
            
        Returns:
            Tuple of (cluster_ids, distances_to_centroids)
        """
        if self.kmeans is None:
            # No clustering available, return defaults
            return np.zeros(len(embeddings), dtype=int), np.zeros(len(embeddings))
        
        # Predict clusters
        cluster_ids = self.kmeans.predict(embeddings)
        
        # Calculate distances to assigned cluster centers
        distances = np.zeros(len(embeddings))
        for i, (embedding, cluster_id) in enumerate(zip(embeddings, cluster_ids)):
            center = self.cluster_centers[cluster_id]
            distances[i] = np.linalg.norm(embedding - center)
        
        return cluster_ids, distances
    
    def find_nearest_clusters(self, query_embedding: np.ndarray, top_k: int = 2) -> List[int]:
        """
        Find the nearest clusters to a query embedding.
        
        Args:
            query_embedding: Query vector
            top_k: Number of nearest clusters to return
            
        Returns:
            List of cluster IDs sorted by distance
        """
        if self.cluster_centers is None:
            return []
        
        # Calculate distances to all cluster centers
        distances = np.array([
            np.linalg.norm(query_embedding - center)
            for center in self.cluster_centers
        ])
        
        # Get top-k nearest clusters
        top_k = min(top_k, len(self.cluster_centers))
        nearest_indices = np.argsort(distances)[:top_k]
        
        return nearest_indices.tolist()
    
    def get_cluster_info(self) -> Dict:
        """
        Get information about the clusters.
        
        Returns:
            Dictionary with cluster statistics
        """
        if self.kmeans is None:
            return {"n_clusters": 0, "status": "not_fitted"}
        
        return {
            "n_clusters": self.n_clusters,
            "inertia": float(self.kmeans.inertia_),
            "status": "fitted"
        }