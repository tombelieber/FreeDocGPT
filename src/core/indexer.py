import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import streamlit as st

from ..config import get_settings
from ..document_processing import DocumentReader, TextChunker, DocumentAnalyzer
from .database import DatabaseManager
from .embeddings import EmbeddingService
from .deduplication import DocumentHasher, ChunkDeduplicator, IncrementalIndexer
from .hybrid_search import HybridSearch
from .token_chunker import TokenChunker, ChunkOptimizer

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Handles document indexing into vector database."""
    
    def __init__(
        self, 
        db_manager: Optional[DatabaseManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        document_reader: Optional[DocumentReader] = None,
        text_chunker: Optional[TextChunker] = None,
        document_analyzer: Optional[DocumentAnalyzer] = None,
        use_token_chunking: bool = False
    ):
        self.settings = get_settings()
        self.db_manager = db_manager or DatabaseManager()
        self.embedding_service = embedding_service or EmbeddingService()
        self.document_reader = document_reader or DocumentReader()
        self.text_chunker = text_chunker or TextChunker()
        self.document_analyzer = document_analyzer or DocumentAnalyzer()
        
        # Initialize token-based chunker if enabled
        self.use_token_chunking = use_token_chunking or self.settings.use_token_chunking
        if self.use_token_chunking:
            self.token_chunker = TokenChunker(use_spacy=True)
            self.chunk_optimizer = ChunkOptimizer()
        else:
            self.token_chunker = None
            self.chunk_optimizer = None
        
        # Initialize deduplication and hybrid search
        self.hasher = DocumentHasher()
        self.deduplicator = ChunkDeduplicator(
            threshold=self.settings.dedup_threshold
        ) if self.settings.dedup_enabled else None
        self.incremental_indexer = IncrementalIndexer(self.db_manager)
        
        # Initialize hybrid search if enabled
        if self.settings.hybrid_search_enabled:
            try:
                self.hybrid_search = HybridSearch(
                    db_manager=self.db_manager,
                    embedding_service=self.embedding_service
                )
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid search: {e}")
                self.hybrid_search = None
        else:
            self.hybrid_search = None
    
    def scan_documents_folder(self) -> List[Path]:
        """Scan documents folder for supported files."""
        folder = self.settings.get_documents_path()
        files = []
        
        for ext in self.settings.supported_extensions:
            # Use rglob to recursively find files in subdirectories
            files.extend(folder.rglob(f"*{ext}"))
            # Also check for uppercase extensions
            files.extend(folder.rglob(f"*{ext.upper()}"))
        
        # Remove duplicates and sort
        unique_files = list(set(files))
        return sorted(unique_files)
    
    def index_documents(
        self, 
        files: List[Path], 
        chunk_chars: int = 1200, 
        overlap: int = 200, 
        auto_detect: bool = True
    ):
        """Index multiple documents into the vector database."""
        if not files:
            st.warning("No files to index")
            return
        
        # Use incremental indexer to filter files
        new_files, changed_files = self.incremental_indexer.filter_changed_files(files)
        
        # Handle changed files (reindex them)
        docs_folder = self.settings.get_documents_path()
        for changed_file in changed_files:
            try:
                rel_path = changed_file.relative_to(docs_folder)
                source_key = str(rel_path)
            except ValueError:
                source_key = changed_file.name
            
            # Remove old chunks before reindexing
            self.incremental_indexer.remove_old_chunks(source_key)
            st.info(f"♻️ Reindexing changed file: {source_key}")
        
        # Combine new and changed files for processing
        files_to_index = new_files + changed_files
        
        if not files_to_index:
            st.info("All files are up to date")
            return
        
        st.info(f"Processing {len(new_files)} new and {len(changed_files)} changed document(s)...")
        
        if auto_detect:
            st.info("🤖 Auto-detecting document types for optimal processing...")
        
        all_docs = []
        doc_stats = {"meeting": 0, "prd": 0, "technical": 0, "wiki": 0, "general": 0}
        
        for file_path in files_to_index:
            # Get relative path for display and storage
            try:
                rel_path = file_path.relative_to(docs_folder)
                display_name = str(rel_path)
            except ValueError:
                display_name = file_path.name
            
            with st.spinner(f"Processing {display_name}..."):
                text = self.document_reader.read_file(file_path)
                
                if text and text.strip():
                    if auto_detect:
                        with st.spinner(f"🤖 Analyzing {display_name} with AI..."):
                            # Detect document type and language
                            doc_type = self.document_analyzer.detect_document_type(text, display_name)
                            language = self.document_analyzer.detect_language(text)
                            
                            # Get adaptive parameters
                            adaptive_chunk, adaptive_overlap = self.document_analyzer.get_adaptive_chunk_params(
                                doc_type, language
                            )
                            doc_stats[doc_type] = doc_stats.get(doc_type, 0) + 1
                            
                            # Show detection results
                            self._show_detection_results(display_name, doc_type, language, 
                                                        adaptive_chunk, adaptive_overlap)
                            
                            # Use token-based chunking if enabled
                            if self.use_token_chunking and self.token_chunker:
                                # Get optimal chunk size for the model
                                max_tokens = self.chunk_optimizer.get_optimal_chunk_size(
                                    self.settings.generation_model
                                )
                                # Use adaptive chunking based on document type
                                chunk_results = self.token_chunker.adaptive_chunk(
                                    text, doc_type=doc_type, max_tokens=max_tokens
                                )
                                chunks = [c['text'] for c in chunk_results]
                                st.caption(f"🎯 Token-based chunking: {len(chunks)} chunks, max {max_tokens} tokens")
                            else:
                                # Use smart chunking for technical docs
                                chunks = self.text_chunker.smart_chunk_text(
                                    text, adaptive_chunk, adaptive_overlap, 
                                    preserve_code=(doc_type == "technical")
                                )
                    else:
                        doc_type = "general"
                        language = "english"
                        chunks = self.text_chunker.chunk_text(text, chunk_chars, overlap)
                    
                    # Get file metadata
                    file_hash, file_modified = self.hasher.file_hash(file_path)
                    
                    for idx, chunk in enumerate(chunks):
                        all_docs.append({
                            "source": display_name,  # Store the relative path as source
                            "chunk": chunk,
                            "timestamp": datetime.now().isoformat(),
                            "content_hash": self.hasher.fast_hash(chunk),
                            "doc_type": doc_type if auto_detect else "general",
                            "language": language if auto_detect else "english",
                            "chunk_index": idx,
                            "total_chunks": len(chunks),
                            "file_modified": file_modified
                        })
        
        if all_docs:
            # Apply content-based deduplication
            if self.settings.dedup_enabled and self.deduplicator:
                all_docs = self.deduplicator.deduplicate_by_content(all_docs)
            
            self._index_chunks(all_docs, doc_stats, auto_detect, len(files_to_index))
    
    def _show_detection_results(self, filename: str, doc_type: str, language: str, 
                               chunk_size: int, overlap: int):
        """Display document detection results."""
        type_emoji = self.settings.doc_type_params.get(doc_type, {}).get("emoji", "📄")
        
        lang_emoji = {
            "english": "🇬🇧",
            "chinese_simplified": "🇨🇳",
            "chinese_traditional": "🇹🇼",
            "mixed": "🌐",
            "unknown": "❓"
        }.get(language, "❓")
        
        st.caption(f"{type_emoji} Detected: {doc_type.upper()} document in {lang_emoji} {language}")
        st.caption(f"📊 Using adaptive settings: chunk={chunk_size}, overlap={overlap}")
    
    def _index_chunks(self, all_docs: list, doc_stats: dict, auto_detect: bool, num_files: int):
        """Index document chunks with embeddings."""
        with st.spinner(f"Generating embeddings for {len(all_docs)} chunks..."):
            embeddings = self.embedding_service.embed_texts(
                [d["chunk"] for d in all_docs]
            )
            
            if embeddings:
                # Apply similarity-based deduplication if enabled
                if self.settings.dedup_enabled and self.deduplicator:
                    import numpy as np
                    embeddings_array = np.array(embeddings, dtype=np.float32)
                    all_docs, embeddings_array = self.deduplicator.deduplicate_by_similarity(
                        all_docs, embeddings_array
                    )
                    embeddings = embeddings_array.tolist() if len(embeddings_array) > 0 else []
                
                rows = []
                for i, (doc, emb) in enumerate(zip(all_docs, embeddings)):
                    row_data = {
                        "id": int(time.time() * 1e6) + i,
                        "source": doc["source"],
                        "chunk": doc["chunk"],
                        "vector": emb,
                        "timestamp": doc["timestamp"],
                        "content_hash": doc.get("content_hash", ""),
                        "doc_type": doc.get("doc_type", "general"),
                        "language": doc.get("language", "english"),
                        "chunk_index": doc.get("chunk_index", 0),
                        "total_chunks": doc.get("total_chunks", 1),
                        "page_number": doc.get("page_number", 0),
                        "section_header": doc.get("section_header", ""),
                        "file_modified": doc.get("file_modified", datetime.now().replace(microsecond=0))
                    }
                    rows.append(row_data)
                
                # Only add documents if we have rows to add
                if rows:
                    self.db_manager.add_documents(rows)
                else:
                    logger.info("No rows to add to database")
                
                # Index in Tantivy for hybrid search
                if self.hybrid_search and rows:
                    try:
                        self.hybrid_search.index_documents(rows)
                    except Exception as e:
                        logger.warning(f"Failed to index in Tantivy: {e}")
                
                # Show statistics
                if auto_detect and sum(doc_stats.values()) > 0:
                    st.success(f"✅ Indexed {num_files} document(s) with {len(rows)} chunks")
                    stats_text = ", ".join([
                        f"{k.capitalize()} ({v})" 
                        for k, v in doc_stats.items() 
                        if v > 0
                    ])
                    st.info(f"📊 Document types detected: {stats_text}")
                else:
                    st.success(f"✅ Indexed {num_files} document(s) with {len(rows)} chunks")