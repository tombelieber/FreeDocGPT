import logging
from typing import Tuple, Optional

import ollama
import streamlit as st

from ..config import get_settings

logger = logging.getLogger(__name__)


class DocumentAnalyzer:
    """Analyzes documents to detect type and language."""
    
    def __init__(self, model: Optional[str] = None):
        settings = get_settings()
        self.model = model or settings.gen_model
        self.doc_type_params = settings.doc_type_params
        self.language_adjustments = settings.language_adjustments
    
    def detect_document_type(self, text: str, filename: str = "") -> str:
        """Use LLM to intelligently detect document type."""
        sample = text[:3000] if len(text) > 3000 else text
        
        classification_prompt = f"""
        Analyze this document sample and classify it into ONE of these categories:
        - meeting: Meeting notes, minutes, action items, decisions (会议记录, 會議記錄, 行动项, 決定事項)
        - prd: Product requirements, specifications, user stories (产品需求文档, 產品需求文檔, 规格说明, 規格說明)
        - technical: Technical documentation, API docs, code documentation (技术文档, 技術文檔, API文档, 代码说明)
        - wiki: Knowledge base articles, how-to guides, FAQs (知识库, 知識庫, 操作指南, 常见问题)
        - general: Other documents that don't fit above categories
        
        Document filename: {filename}
        
        Document sample:
        {sample}
        
        Respond with ONLY the category name (meeting/prd/technical/wiki/general), nothing else.
        """
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=classification_prompt,
                options={"temperature": 0.1, "num_predict": 10}
            )
            
            doc_type = response['response'].strip().lower()
            
            # Validate response
            valid_types = ["meeting", "prd", "technical", "wiki", "general"]
            if doc_type not in valid_types:
                # Fallback: check for code blocks as a simple heuristic
                if "```" in text or "def " in text or "function" in text:
                    return "technical"
                return "general"
            
            return doc_type
            
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            st.warning(f"Document type detection failed, using fallback: {e}")
            # Simple fallback based on code blocks
            if "```" in text:
                return "technical"
            return "general"
    
    def detect_language(self, text: str) -> str:
        """Detect the primary language of the document."""
        sample = text[:500] if len(text) > 500 else text
        
        language_prompt = f"""
        Detect the primary language of this text. Common options:
        - english
        - chinese_simplified (简体中文)
        - chinese_traditional (繁體中文)
        - mixed (multiple languages)
        
        Text: {sample}
        
        Respond with ONLY the language identifier, nothing else.
        """
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=language_prompt,
                options={"temperature": 0.1, "num_predict": 20}
            )
            return response['response'].strip().lower()
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "unknown"
    
    def get_adaptive_chunk_params(self, doc_type: str, language: str = "english") -> Tuple[int, int]:
        """Get optimal chunk parameters based on document type and language."""
        # Get base parameters from config
        params = self.doc_type_params.get(doc_type, self.doc_type_params["general"])
        chunk_size = params["chunk_size"]
        overlap = params["overlap"]
        
        # Adjust for CJK languages
        if "chinese" in language or "japanese" in language or "korean" in language:
            adjustments = self.language_adjustments["cjk"]
            chunk_size = int(chunk_size * adjustments["chunk_multiplier"])
            overlap = int(overlap * adjustments["overlap_multiplier"])
        elif language == "mixed":
            adjustments = self.language_adjustments["mixed"]
            chunk_size = int(chunk_size * adjustments["chunk_multiplier"])
        
        return chunk_size, overlap