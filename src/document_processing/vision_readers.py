import io
import json
import logging
import base64
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class VisionDocumentReader:
    """Enhanced document reader with vision capabilities for PDFs."""
    
    def __init__(self):
        self.extracted_images = []
        
    def extract_images_from_pdf(self, pdf_bytes: bytes) -> List[Dict]:
        """Extract images from PDF bytes."""
        images = []
        try:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num, page in enumerate(pdf_document):
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                    else:  # CMYK: Convert to RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix.tobytes("png")
                    
                    # Convert to PIL Image
                    img_pil = Image.open(io.BytesIO(img_data))
                    
                    # Convert to base64 for storage
                    buffered = io.BytesIO()
                    img_pil.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    images.append({
                        "page": page_num + 1,
                        "index": img_index,
                        "data": img_base64,
                        "format": "png",
                        "width": img_pil.width,
                        "height": img_pil.height
                    })
                    
                    pix = None
                    
            pdf_document.close()
            
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {e}")
            
        return images
    
    def extract_tables_from_pdf(self, pdf_bytes: bytes) -> List[Dict]:
        """Extract tables from PDF using pdfplumber."""
        tables = []
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table_index, table in enumerate(page_tables):
                        if table:
                            tables.append({
                                "page": page_num + 1,
                                "index": table_index,
                                "data": table,
                                "type": "table"
                            })
        except Exception as e:
            logger.error(f"Error extracting tables from PDF: {e}")
            
        return tables
    
    def read_pdf_with_vision(self, pdf_bytes: bytes) -> Dict:
        """Read PDF with text, images, and tables extraction."""
        result = {
            "text": "",
            "images": [],
            "tables": [],
            "metadata": {}
        }
        
        try:
            # Extract text
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text_pages = []
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text:
                    text_pages.append(f"[Page {page_num + 1}]\n{page_text}")
            result["text"] = "\n\n".join(text_pages)
            
            # Extract images
            result["images"] = self.extract_images_from_pdf(pdf_bytes)
            
            # Extract tables
            result["tables"] = self.extract_tables_from_pdf(pdf_bytes)
            
            # Add metadata
            result["metadata"] = {
                "num_pages": len(reader.pages),
                "num_images": len(result["images"]),
                "num_tables": len(result["tables"])
            }
            
        except Exception as e:
            logger.error(f"Error reading PDF with vision: {e}")
            raise
            
        return result
    
    def format_for_indexing(self, pdf_content: Dict) -> str:
        """Format extracted content for text indexing."""
        parts = []
        
        # Add text content
        if pdf_content["text"]:
            parts.append(pdf_content["text"])
        
        # Add table descriptions
        for table in pdf_content["tables"]:
            parts.append(f"\n[Table on page {table['page']}]")
            if table["data"] and len(table["data"]) > 0:
                # Format first row as headers if available
                headers = table["data"][0] if len(table["data"]) > 0 else []
                if headers:
                    parts.append("Headers: " + " | ".join(str(h) for h in headers if h))
                # Add sample rows
                for row in table["data"][1:4]:  # First 3 data rows
                    if row:
                        parts.append(" | ".join(str(cell) for cell in row if cell))
        
        # Add image placeholders
        for img in pdf_content["images"]:
            parts.append(f"\n[Image on page {img['page']}: {img['width']}x{img['height']} pixels]")
        
        return "\n".join(parts)
    
    def get_image_for_vision(self, image_data: Dict) -> bytes:
        """Convert stored image data back to bytes for vision model."""
        return base64.b64decode(image_data["data"])