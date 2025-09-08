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
            files.extend(folder.glob(f"*{ext}"))
        
        return sorted(files)
    
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
        
        # Filter out already indexed files
        new_files = [f for f in files if f.name not in existing_sources]
        
        if not new_files:
            st.info("All files are already indexed")
            return
        
        st.info(f"Indexing {len(new_files)} new document(s)...")
        
        if auto_detect:
            st.info("🤖 Auto-detecting document types for optimal processing...")
        
        all_docs = []
        doc_stats = {"meeting": 0, "prd": 0, "technical": 0, "wiki": 0, "general": 0}
        
        for file_path in new_files:
            with st.spinner(f"Processing {file_path.name}..."):
                text = self.document_reader.read_file(file_path)
                
                if text and text.strip():
                    if auto_detect:
                        with st.spinner(f"🤖 Analyzing {file_path.name} with AI..."):
                            # Detect document type and language
                            doc_type = self.document_analyzer.detect_document_type(text, file_path.name)
                            language = self.document_analyzer.detect_language(text)
                            
                            # Get adaptive parameters
                            adaptive_chunk, adaptive_overlap = self.document_analyzer.get_adaptive_chunk_params(
                                doc_type, language
                            )
                            doc_stats[doc_type] = doc_stats.get(doc_type, 0) + 1
                            
                            # Show detection results
                            self._show_detection_results(file_path.name, doc_type, language, 
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
                            "source": file_path.name, 
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