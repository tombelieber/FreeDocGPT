import logging
from typing import List

logger = logging.getLogger(__name__)


class TextChunker:
    """Handles text chunking with various strategies."""
    
    @staticmethod
    def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        text = " ".join(text.split())
        chunks, start, n = [], 0, len(text)
        
        while start < n:
            end = min(n, start + chunk_chars)
            chunks.append(text[start:end])
            if end == n:
                break
            start = max(0, end - overlap)
            
        return chunks
    
    @staticmethod
    def smart_chunk_text(
        text: str, 
        chunk_chars: int = 1200, 
        overlap: int = 200, 
        preserve_code: bool = True
    ) -> List[str]:
        """Smart text chunking that preserves code blocks and paragraph boundaries."""
        if not preserve_code or "```" not in text:
            # Simple chunking if no code blocks
            return TextChunker.chunk_text(text, chunk_chars, overlap)
        
        chunks = []
        parts = text.split("```")
        
        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Regular text - chunk normally
                if part.strip():
                    text_chunks = TextChunker.chunk_text(part, chunk_chars, overlap)
                    chunks.extend(text_chunks)
            else:
                # Code block - try to keep together if not too large
                code_block = f"```{part}```"
                if len(code_block) <= chunk_chars * 1.5:  # Allow 50% larger for code blocks
                    chunks.append(code_block)
                else:
                    # Code block too large, chunk it but preserve markers
                    code_chunks = TextChunker.chunk_text(part, chunk_chars, overlap)
                    for j, code_chunk in enumerate(code_chunks):
                        if j == 0:
                            chunks.append(f"```{code_chunk}")
                        elif j == len(code_chunks) - 1:
                            chunks.append(f"{code_chunk}```")
                        else:
                            chunks.append(code_chunk)
        
        return chunks