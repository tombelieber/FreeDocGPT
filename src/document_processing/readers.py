import io
import json
import logging
from pathlib import Path
from typing import Optional

import markdown
import pandas as pd
from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class DocumentReader:
    """Handles reading various document formats."""
    
    @staticmethod
    def read_pdf_bytes(pdf_bytes: bytes) -> str:
        """Read PDF from bytes."""
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            return "\n".join([p.extract_text() or "" for p in reader.pages])
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise
    
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
            logger.error(f"Error reading DOCX: {e}")
            raise
    
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
                return self.read_pdf_bytes(file_bytes)
            elif ext in [".docx", ".doc"]:
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
            logger.error(f"Error reading {file_path.name}: {e}")
            return None