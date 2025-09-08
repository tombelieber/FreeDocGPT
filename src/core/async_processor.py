"""
Async and batch processing for improved performance.
Leverages M1 efficiency and performance cores effectively.
"""

import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import logging
import time
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result from async processing."""
    file_path: Path
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    processing_time: float = 0.0


class AsyncDocumentProcessor:
    """Async document processing with optimized resource utilization."""
    
    def __init__(
        self,
        io_workers: int = 4,  # Use efficiency cores for I/O
        compute_workers: int = 8,  # Use performance cores for compute
        batch_size: int = 10
    ):
        """
        Initialize async processor.
        
        Args:
            io_workers: Number of workers for I/O operations
            compute_workers: Number of workers for compute operations
            batch_size: Size of processing batches
        """
        self.io_executor = ThreadPoolExecutor(max_workers=io_workers)
        self.compute_executor = ThreadPoolExecutor(max_workers=compute_workers)
        self.batch_size = batch_size
    
    async def read_document_async(self, file_path: Path) -> str:
        """
        Read a document asynchronously.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document content
        """
        try:
            async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
                content = await f.read()
            return content
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            raise
    
    async def process_documents(
        self,
        file_paths: List[Path],
        process_func: Callable,
        **kwargs
    ) -> List[ProcessingResult]:
        """
        Process multiple documents in parallel.
        
        Args:
            file_paths: List of file paths to process
            process_func: Function to process each document
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of processing results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(file_paths), self.batch_size):
            batch = file_paths[i:i + self.batch_size]
            batch_results = await self._process_batch(batch, process_func, **kwargs)
            results.extend(batch_results)
            
            # Log progress
            logger.info(f"Processed {min(i + self.batch_size, len(file_paths))}/{len(file_paths)} documents")
        
        return results
    
    async def _process_batch(
        self,
        file_paths: List[Path],
        process_func: Callable,
        **kwargs
    ) -> List[ProcessingResult]:
        """
        Process a batch of documents.
        
        Args:
            file_paths: Batch of file paths
            process_func: Processing function
            **kwargs: Additional arguments
            
        Returns:
            List of results for the batch
        """
        tasks = []
        for file_path in file_paths:
            task = self._process_single_document(file_path, process_func, **kwargs)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert results to ProcessingResult objects
        processed_results = []
        for file_path, result in zip(file_paths, results):
            if isinstance(result, Exception):
                processed_results.append(
                    ProcessingResult(
                        file_path=file_path,
                        success=False,
                        error=str(result)
                    )
                )
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single_document(
        self,
        file_path: Path,
        process_func: Callable,
        **kwargs
    ) -> ProcessingResult:
        """
        Process a single document.
        
        Args:
            file_path: Path to document
            process_func: Processing function
            **kwargs: Additional arguments
            
        Returns:
            Processing result
        """
        start_time = time.time()
        
        try:
            # Read document asynchronously
            content = await self.read_document_async(file_path)
            
            # Process in executor (CPU-bound work)
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                self.compute_executor,
                process_func,
                content,
                **kwargs
            )
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                file_path=file_path,
                success=True,
                data=data,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to process {file_path}: {e}")
            
            return ProcessingResult(
                file_path=file_path,
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    def __del__(self):
        """Cleanup executors."""
        self.io_executor.shutdown(wait=False)
        self.compute_executor.shutdown(wait=False)


class BatchEmbeddingProcessor:
    """Batch processing for embeddings generation."""
    
    def __init__(self, embedding_service, batch_size: int = 32):
        """
        Initialize batch embedding processor.
        
        Args:
            embedding_service: Service for generating embeddings
            batch_size: Size of embedding batches
        """
        self.embedding_service = embedding_service
        self.batch_size = batch_size
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress
            
        Returns:
            List of embeddings
        """
        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Generate embeddings for batch
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                None,
                self.embedding_service.embed_texts,
                batch
            )
            
            embeddings.extend(batch_embeddings)
            
            if show_progress:
                batch_num = i // self.batch_size + 1
                logger.info(f"Generated embeddings for batch {batch_num}/{total_batches}")
        
        return embeddings


class ParallelSearchProcessor:
    """Parallel processing for search operations."""
    
    def __init__(self, search_service, max_workers: int = 4):
        """
        Initialize parallel search processor.
        
        Args:
            search_service: Service for searching
            max_workers: Maximum parallel search workers
        """
        self.search_service = search_service
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def parallel_search(
        self,
        queries: List[str],
        k: int = 5,
        search_mode: str = "hybrid"
    ) -> List[Tuple[Any, Dict]]:
        """
        Execute multiple searches in parallel.
        
        Args:
            queries: List of search queries
            k: Number of results per query
            search_mode: Search mode to use
            
        Returns:
            List of (results, stats) tuples
        """
        tasks = []
        loop = asyncio.get_event_loop()
        
        for query in queries:
            task = loop.run_in_executor(
                self.executor,
                self.search_service.search_similar,
                query,
                k,
                search_mode
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    def __del__(self):
        """Cleanup executor."""
        self.executor.shutdown(wait=False)


class AsyncIndexer:
    """Async indexing with optimized batching."""
    
    def __init__(
        self,
        indexer,
        async_processor: Optional[AsyncDocumentProcessor] = None,
        batch_processor: Optional[BatchEmbeddingProcessor] = None
    ):
        """
        Initialize async indexer.
        
        Args:
            indexer: Base indexer instance
            async_processor: Async document processor
            batch_processor: Batch embedding processor
        """
        self.indexer = indexer
        self.async_processor = async_processor or AsyncDocumentProcessor()
        self.batch_processor = batch_processor or BatchEmbeddingProcessor(
            indexer.embedding_service
        )
    
    async def index_documents_async(
        self,
        file_paths: List[Path],
        chunk_size: int = 1200,
        overlap: int = 200
    ) -> Dict[str, Any]:
        """
        Index documents asynchronously.
        
        Args:
            file_paths: List of document paths
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            
        Returns:
            Indexing statistics
        """
        start_time = time.time()
        
        # Process documents in parallel
        process_func = lambda text: self.indexer.text_chunker.chunk_text(
            text, chunk_size, overlap
        )
        
        results = await self.async_processor.process_documents(
            file_paths, process_func
        )
        
        # Collect all chunks
        all_chunks = []
        successful_files = []
        
        for result in results:
            if result.success and result.data:
                for chunk in result.data:
                    all_chunks.append({
                        'source': str(result.file_path),
                        'chunk': chunk
                    })
                successful_files.append(result.file_path)
        
        # Generate embeddings in batches
        if all_chunks:
            texts = [c['chunk'] for c in all_chunks]
            embeddings = await self.batch_processor.generate_embeddings_batch(texts)
            
            # Add to database
            await self._add_to_database(all_chunks, embeddings)
        
        total_time = time.time() - start_time
        
        return {
            'total_files': len(file_paths),
            'successful_files': len(successful_files),
            'failed_files': len(file_paths) - len(successful_files),
            'total_chunks': len(all_chunks),
            'processing_time': total_time,
            'chunks_per_second': len(all_chunks) / total_time if total_time > 0 else 0
        }
    
    async def _add_to_database(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]]
    ):
        """
        Add chunks and embeddings to database.
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: List of embeddings
        """
        # Run database operation in executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._add_to_database_sync,
            chunks,
            embeddings
        )
    
    def _add_to_database_sync(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]]
    ):
        """Synchronous database addition."""
        rows = []
        for chunk, embedding in zip(chunks, embeddings):
            row = {
                'id': int(time.time() * 1e6),
                'source': chunk['source'],
                'chunk': chunk['chunk'],
                'vector': embedding,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            rows.append(row)
        
        self.indexer.db_manager.add_documents(rows)


class PerformanceMonitor:
    """Monitor async processing performance."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'total_processing_time': 0.0,
            'average_document_time': 0.0
        }
    
    def update_metrics(self, stats: Dict[str, Any]):
        """
        Update performance metrics.
        
        Args:
            stats: Processing statistics
        """
        self.metrics['documents_processed'] += stats.get('successful_files', 0)
        self.metrics['chunks_created'] += stats.get('total_chunks', 0)
        self.metrics['embeddings_generated'] += stats.get('total_chunks', 0)
        self.metrics['total_processing_time'] += stats.get('processing_time', 0)
        
        if self.metrics['documents_processed'] > 0:
            self.metrics['average_document_time'] = (
                self.metrics['total_processing_time'] / 
                self.metrics['documents_processed']
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'total_processing_time': 0.0,
            'average_document_time': 0.0
        }