"""
Content-based change detection and deduplication using xxHash and Faiss
Optimized for M1 Mac with NEON SIMD acceleration
"""

import logging
import hashlib
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
from datetime import datetime

import xxhash
import numpy as np
import faiss
import pandas as pd

from .database import DatabaseManager
from ..config import get_settings

logger = logging.getLogger(__name__)


class DocumentHasher:
    """Fast content-based hashing for change detection."""
    
    @staticmethod
    def fast_hash(content: str) -> str:
        """
        Quick hash for change detection using xxHash.
        xxHash is optimized for ARM64 and provides 5GB/s throughput on M1.
        """
        if isinstance(content, str):
            content = content.encode('utf-8')
        return xxhash.xxh64(content).hexdigest()
    
    @staticmethod
    def secure_hash(content: str) -> str:
        """
        Cryptographic hash for verification when needed.
        Use for critical deduplication or integrity checks.
        """
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.sha256(content).hexdigest()
    
    @staticmethod
    def file_hash(file_path: Path) -> Tuple[str, datetime]:
        """
        Generate hash for a file and get modification time.
        Returns (content_hash, modification_time)
        """
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            content_hash = xxhash.xxh64(content).hexdigest()
            mod_time = datetime.fromtimestamp(file_path.stat().st_mtime).replace(microsecond=0)
            
            return content_hash, mod_time
            
        except Exception as e:
            logger.error(f"Failed to hash file {file_path}: {e}")
            return None, None


class ChunkDeduplicator:
    """Deduplicate chunks using Faiss for similarity detection."""
    
    def __init__(self, threshold: float = 0.95, embedding_dim: int = 768):
        """
        Initialize deduplicator with similarity threshold.
        
        Args:
            threshold: Similarity threshold (0-1). Higher = more similar required for dedup
            embedding_dim: Dimension of embeddings
        """
        self.threshold = threshold
        self.embedding_dim = embedding_dim
        
        # Use IndexFlatIP for inner product similarity
        # Faiss-CPU is optimized for ARM64 NEON instructions
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # Track unique chunks
        self.unique_hashes: Set[str] = set()
        self.unique_embeddings: List[np.ndarray] = []
        
    def deduplicate_by_content(self, chunks: List[Dict]) -> List[Dict]:
        """
        Remove exact duplicate chunks based on content hash.
        
        Args:
            chunks: List of chunk dictionaries with 'chunk' field
        
        Returns:
            List of unique chunks
        """
        unique_chunks = []
        seen_hashes = set()
        
        for chunk_data in chunks:
            content = chunk_data.get('chunk', '')
            content_hash = DocumentHasher.fast_hash(content)
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                chunk_data['content_hash'] = content_hash
                unique_chunks.append(chunk_data)
            else:
                logger.debug(f"Removed duplicate chunk with hash {content_hash}")
        
        logger.info(f"Content deduplication: {len(chunks)} -> {len(unique_chunks)} chunks")
        return unique_chunks
    
    def deduplicate_by_similarity(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Remove near-duplicate chunks based on embedding similarity.
        Uses Faiss for efficient similarity search.
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: Numpy array of embeddings (n_chunks x embedding_dim)
        
        Returns:
            Tuple of (unique_chunks, unique_embeddings)
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Number of chunks must match number of embeddings")
        
        unique_chunks = []
        unique_embeddings = []
        
        # Ensure embeddings are float32 and C-contiguous for FAISS
        import numpy as np
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        for i, (chunk_data, embedding) in enumerate(zip(chunks, embeddings)):
            embedding = np.ascontiguousarray(embedding.reshape(1, -1), dtype=np.float32)
            
            if self.index.ntotal > 0:
                # Search for similar embeddings
                D, I = self.index.search(embedding, 1)
                similarity = D[0][0]
                
                if similarity < self.threshold:
                    # Not similar enough to existing chunks
                    unique_chunks.append(chunk_data)
                    unique_embeddings.append(embedding)
                    self.index.add(embedding)
                else:
                    logger.debug(f"Removed near-duplicate chunk (similarity: {similarity:.3f})")
            else:
                # First chunk, always unique
                unique_chunks.append(chunk_data)
                unique_embeddings.append(embedding)
                self.index.add(embedding)
        
        if unique_embeddings:
            unique_embeddings = np.vstack(unique_embeddings)
        else:
            unique_embeddings = np.array([])
        
        logger.info(f"Similarity deduplication: {len(chunks)} -> {len(unique_chunks)} chunks")
        return unique_chunks, unique_embeddings
    
    def reset(self):
        """Reset the deduplicator index."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.unique_hashes.clear()
        self.unique_embeddings.clear()


class IncrementalIndexer:
    """Handle incremental indexing with change detection."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize incremental indexer."""
        self.db_manager = db_manager or DatabaseManager()
        self.settings = get_settings()
        self.hasher = DocumentHasher()
        
    def get_indexed_files(self) -> Dict[str, Dict]:
        """
        Get information about already indexed files.
        
        Returns:
            Dictionary mapping file paths to their metadata
        """
        try:
            table = self.db_manager.get_table()
            df = table.to_pandas()
            
            if df.empty:
                return {}
            
            # Group by source file
            indexed_files = {}
            for source in df['source'].unique():
                source_df = df[df['source'] == source]
                
                # Get file metadata
                indexed_files[source] = {
                    'chunk_count': len(source_df),
                    'timestamp': source_df['timestamp'].iloc[0],
                    'content_hash': source_df['content_hash'].iloc[0] if 'content_hash' in source_df.columns else None,
                    'file_modified': source_df['file_modified'].iloc[0] if 'file_modified' in source_df.columns else None,
                }
            
            return indexed_files
            
        except Exception as e:
            logger.error(f"Failed to get indexed files: {e}")
            return {}
    
    def needs_reindexing(self, file_path: Path, indexed_info: Dict) -> bool:
        """
        Check if a file needs to be reindexed.
        
        Args:
            file_path: Path to the file
            indexed_info: Information about the indexed version
        
        Returns:
            True if file needs reindexing
        """
        # Get current file hash and modification time
        current_hash, current_mod_time = self.hasher.file_hash(file_path)
        
        if current_hash is None:
            return False  # Can't read file
        
        # Check content hash first (most reliable)
        if indexed_info.get('content_hash'):
            if current_hash != indexed_info['content_hash']:
                logger.info(f"File {file_path.name} has changed (hash mismatch)")
                return True
        
        # Check modification time as fallback
        if indexed_info.get('file_modified'):
            indexed_mod_time = pd.to_datetime(indexed_info['file_modified'])
            if current_mod_time > indexed_mod_time.to_pydatetime():
                logger.info(f"File {file_path.name} has been modified")
                return True
        
        return False
    
    def filter_changed_files(self, files: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        Filter files into new and changed categories.
        
        Args:
            files: List of file paths to check
        
        Returns:
            Tuple of (new_files, changed_files)
        """
        indexed_files = self.get_indexed_files()
        docs_folder = self.settings.get_documents_path()
        
        new_files = []
        changed_files = []
        
        for file_path in files:
            # Get relative path for comparison
            try:
                rel_path = file_path.relative_to(docs_folder)
                source_key = str(rel_path)
            except ValueError:
                source_key = file_path.name
            
            if source_key in indexed_files:
                # File is indexed, check if it changed
                if self.needs_reindexing(file_path, indexed_files[source_key]):
                    changed_files.append(file_path)
            else:
                # New file
                new_files.append(file_path)
        
        return new_files, changed_files
    
    def remove_old_chunks(self, source: str):
        """
        Remove old chunks for a file before reindexing.
        
        Args:
            source: Source file identifier
        """
        try:
            table = self.db_manager.get_table()
            
            # Delete old chunks for this source
            # Note: LanceDB doesn't have direct delete by condition,
            # so we need to filter and recreate
            df = table.to_pandas()
            filtered_df = df[df['source'] != source]
            
            # Recreate table with filtered data
            self.db_manager.db.drop_table(self.settings.table_name)
            if not filtered_df.empty:
                self.db_manager.db.create_table(
                    self.settings.table_name,
                    data=filtered_df
                )
            else:
                # Create empty table with schema
                self.db_manager.table = None
                self.db_manager.get_table()
            
            logger.info(f"Removed old chunks for {source}")
            
        except Exception as e:
            logger.error(f"Failed to remove old chunks for {source}: {e}")