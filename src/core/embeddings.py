import logging
from typing import List, Optional

import ollama
import streamlit as st

from ..config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles text embedding generation."""
    
    def __init__(self, model: Optional[str] = None):
        settings = get_settings()
        self.model = model or settings.embed_model
    
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """Generate embeddings for text chunks."""
        try:
            embeddings = []
            
            if show_progress:
                progress_bar = st.progress(0)
            
            for i, text in enumerate(texts):
                embedding = ollama.embeddings(
                    model=self.model, 
                    prompt=text
                )["embedding"]
                embeddings.append(embedding)
                
                if show_progress:
                    progress_bar.progress((i + 1) / len(texts))
            
            if show_progress:
                progress_bar.empty()
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            st.error(f"Embedding failed: {e}")
            return []
    
    def embed_query(self, query: str) -> Optional[List[float]]:
        """Generate embedding for a single query."""
        try:
            result = ollama.embeddings(
                model=self.model, 
                prompt=query
            )
            return result["embedding"]
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return None