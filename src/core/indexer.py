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

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Handles document indexing into vector database."""
    
    def __init__(
        self, 
        db_manager: Optional[DatabaseManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        document_reader: Optional[DocumentReader] = None,
        text_chunker: Optional[TextChunker] = None,
        document_analyzer: Optional[DocumentAnalyzer] = None
    ):
        self.settings = get_settings()
        self.db_manager = db_manager or DatabaseManager()
        self.embedding_service = embedding_service or EmbeddingService()
        self.document_reader = document_reader or DocumentReader()
        self.text_chunker = text_chunker or TextChunker()
        self.document_analyzer = document_analyzer or DocumentAnalyzer()
    
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
        
        # Get already indexed files
        existing_df = self.db_manager.get_indexed_documents()
        existing_sources = set(existing_df["Document"].values) if not existing_df.empty else set()
        
        # Get the documents folder path for relative path calculation
        docs_folder = self.settings.get_documents_path()
        
        # Filter out already indexed files - use relative path for comparison
        new_files = []
        for f in files:
            # Get relative path from documents folder
            try:
                rel_path = f.relative_to(docs_folder)
                if str(rel_path) not in existing_sources:
                    new_files.append(f)
            except ValueError:
                # File is not in documents folder, use name
                if f.name not in existing_sources:
                    new_files.append(f)
        
        if not new_files:
            st.info("All files are already indexed")
            return
        
        st.info(f"Indexing {len(new_files)} new document(s)...")
        
        if auto_detect:
            st.info("🤖 Auto-detecting document types for optimal processing...")
        
        all_docs = []
        doc_stats = {"meeting": 0, "prd": 0, "technical": 0, "wiki": 0, "general": 0}
        
        for file_path in new_files:
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
                            
                            # Use smart chunking for technical docs
                            chunks = self.text_chunker.smart_chunk_text(
                                text, adaptive_chunk, adaptive_overlap, 
                                preserve_code=(doc_type == "technical")
                            )
                    else:
                        chunks = self.text_chunker.chunk_text(text, chunk_chars, overlap)
                    
                    for chunk in chunks:
                        all_docs.append({
                            "source": display_name,  # Store the relative path as source
                            "chunk": chunk, 
                            "timestamp": datetime.now().isoformat()
                        })
        
        if all_docs:
            self._index_chunks(all_docs, doc_stats, auto_detect, len(new_files))
    
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
                rows = []
                for i, (doc, emb) in enumerate(zip(all_docs, embeddings)):
                    rows.append({
                        "id": int(time.time() * 1e6) + i,
                        "source": doc["source"],
                        "chunk": doc["chunk"],
                        "vector": emb,
                        "timestamp": doc["timestamp"],
                    })
                
                self.db_manager.add_documents(rows)
                
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