import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Optional, Dict

import markdown
import pandas as pd
from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader
from .vision_readers import VisionDocumentReader

logger = logging.getLogger(__name__)


class DocumentReader:
    """Handles reading various document formats."""
    
    def __init__(self):
        self.vision_reader = VisionDocumentReader()
    
    def read_pdf_bytes(self, pdf_bytes: bytes, vision_enabled: bool = True) -> str:
        """Read PDF from bytes with optional vision support."""
        try:
            if vision_enabled:
                # Use vision reader for comprehensive extraction
                pdf_content = self.vision_reader.read_pdf_with_vision(pdf_bytes)
                # Store the full content for later use
                self.last_pdf_content = pdf_content
                # Return formatted text for indexing
                return self.vision_reader.format_for_indexing(pdf_content)
            else:
                # Basic text extraction only
                reader = PdfReader(io.BytesIO(pdf_bytes))
                return "\n".join([p.extract_text() or "" for p in reader.pages])
        except Exception as e:
            logger.warning(f"Skipping PDF due to read error: {e}")
            return None
    
    @staticmethod
    def read_docx_bytes(docx_bytes: bytes) -> str:
        """Read DOCX from bytes."""
        try:
            doc = Document(io.BytesIO(docx_bytes))
            return "\n".join([
                paragraph.text 
                for paragraph in doc.paragraphs 
                if paragraph.text.strip()
            ])
        except Exception as e:
            logger.warning(f"Skipping DOCX due to read error: {e}")
            return None
    
    @staticmethod
    def read_markdown(content: str) -> str:
        """Convert markdown to plain text."""
        try:
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(separator="\n", strip=True)
        except Exception as e:
            logger.error(f"Error reading markdown: {e}")
            raise
    
    @staticmethod
    def read_html(content: str) -> str:
        """Extract text from HTML."""
        try:
            soup = BeautifulSoup(content, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator="\n", strip=True)
        except Exception as e:
            logger.error(f"Error reading HTML: {e}")
            raise
    
    @staticmethod
    def read_csv_bytes(csv_bytes: bytes) -> str:
        """Read CSV from bytes."""
        try:
            df = pd.read_csv(io.BytesIO(csv_bytes))
            text_parts = []
            for _, row in df.iterrows():
                row_text = ", ".join([
                    f"{col}: {val}" 
                    for col, val in row.items() 
                    if pd.notna(val)
                ])
                if row_text:
                    text_parts.append(row_text)
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise
    
    @staticmethod
    def read_excel_bytes(excel_bytes: bytes) -> str:
        """Read Excel from bytes."""
        try:
            dfs = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=None)
            text_parts = []
            for sheet_name, df in dfs.items():
                text_parts.append(f"Sheet: {sheet_name}")
                for _, row in df.iterrows():
                    row_text = ", ".join([
                        f"{col}: {val}" 
                        for col, val in row.items() 
                        if pd.notna(val)
                    ])
                    if row_text:
                        text_parts.append(row_text)
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error reading Excel: {e}")
            raise
    
    @staticmethod
    def read_json_bytes(json_bytes: bytes) -> str:
        """Read JSON from bytes."""
        try:
            data = json.loads(json_bytes.decode("utf-8"))
            return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error reading JSON: {e}")
            raise
    
    def read_file(self, file_path: Path) -> Optional[str]:
        """Read content from various file formats."""
        ext = file_path.suffix.lower()
        
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            
            if ext == ".pdf":
                # Quick PDF validation to avoid noisy parser warnings
                if not file_bytes.startswith(b"%PDF"):
                    logger.warning(f"Invalid PDF header, skipping: {file_path.name}")
                    try:
                        import streamlit as st
                        st.warning(f"Skipping invalid PDF: {file_path.name}")
                    except Exception:
                        pass
                    return None
                return self.read_pdf_bytes(file_bytes, vision_enabled=True)
            elif ext in [".docx", ".doc"]:
                if ext == ".doc":
                    logger.warning(f"Unsupported legacy DOC format, skipping: {file_path.name}")
                    try:
                        import streamlit as st
                        st.warning(f"Skipping unsupported .doc file: {file_path.name}")
                    except Exception:
                        pass
                    return None
                # Validate DOCX is a real zip (OOXML)
                if not zipfile.is_zipfile(io.BytesIO(file_bytes)):
                    logger.warning(f"Invalid DOCX (not a zip), skipping: {file_path.name}")
                    try:
                        import streamlit as st
                        st.warning(f"Skipping invalid DOCX: {file_path.name}")
                    except Exception:
                        pass
                    return None
                return self.read_docx_bytes(file_bytes)
            elif ext in [".md", ".markdown"]:
                content = file_bytes.decode("utf-8")
                return self.read_markdown(content)
            elif ext in [".html", ".htm"]:
                content = file_bytes.decode("utf-8")
                return self.read_html(content)
            elif ext == ".csv":
                return self.read_csv_bytes(file_bytes)
            elif ext in [".xlsx", ".xls"]:
                return self.read_excel_bytes(file_bytes)
            elif ext == ".json":
                return self.read_json_bytes(file_bytes)
            else:
                # Plain text files
                return file_bytes.decode("utf-8", errors="ignore")
                
        except Exception as e:
            logger.warning(f"Skipping unreadable file {file_path.name}: {e}")
            return None
