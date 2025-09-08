"""
Token-based chunking for precise token management.
Supports multiple tokenization models and respects model context windows.
"""

import tiktoken
import spacy
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TokenChunker:
    """Advanced token-based chunking with sentence boundary detection."""
    
    def __init__(self, model: str = "cl100k_base", use_spacy: bool = True):
        """
        Initialize the token chunker.
        
        Args:
            model: Tokenizer model to use (cl100k_base for GPT-4, etc.)
            use_spacy: Whether to use spacy for sentence boundary detection
        """
        self.encoder = tiktoken.get_encoding(model)
        self.use_spacy = use_spacy
        
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                # Add sentencizer if not present
                if "sentencizer" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("sentencizer")
            except Exception as e:
                logger.warning(f"Could not load spacy model: {e}. Falling back to simple chunking.")
                self.use_spacy = False
                self.nlp = None
        else:
            self.nlp = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))
    
    def chunk_by_tokens(
        self, 
        text: str, 
        max_tokens: int = 512, 
        overlap_tokens: int = 50,
        min_chunk_tokens: int = 100
    ) -> List[Tuple[str, int, int]]:
        """
        Chunk text by token count.
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
            min_chunk_tokens: Minimum tokens for a chunk to be valid
            
        Returns:
            List of tuples (chunk_text, start_token_idx, end_token_idx)
        """
        if not text or not text.strip():
            return []
        
        tokens = self.encoder.encode(text)
        
        if len(tokens) <= max_tokens:
            return [(text, 0, len(tokens))]
        
        chunks = []
        stride = max(1, max_tokens - overlap_tokens)  # Ensure stride is at least 1
        
        for i in range(0, len(tokens), stride):
            end_idx = min(i + max_tokens, len(tokens))
            chunk_tokens = tokens[i:end_idx]
            
            # Skip chunks that are too small (unless it's the last chunk)
            if len(chunk_tokens) < min_chunk_tokens and i + stride < len(tokens):
                continue
                
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append((chunk_text, i, end_idx))
            
            # If we've processed all tokens, break
            if end_idx >= len(tokens):
                break
        
        return chunks
    
    def chunk_with_boundaries(
        self,
        text: str,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        respect_sentences: bool = True
    ) -> List[dict]:
        """
        Smart chunking that respects sentence boundaries.
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap between chunks
            respect_sentences: Whether to respect sentence boundaries
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not self.use_spacy or not respect_sentences or not self.nlp:
            # Fall back to simple token chunking
            simple_chunks = self.chunk_by_tokens(text, max_tokens, overlap_tokens)
            return [
                {
                    'text': chunk,
                    'start_token': start,
                    'end_token': end,
                    'token_count': end - start,
                    'method': 'token_based'
                }
                for chunk, start, end in simple_chunks
            ]
        
        # Use spacy for sentence segmentation
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_start_token = 0
        total_tokens_processed = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.encoder.encode(sentence))
            
            # If adding this sentence would exceed max_tokens
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start_token': chunk_start_token,
                    'end_token': total_tokens_processed,
                    'token_count': current_tokens,
                    'method': 'sentence_aware'
                })
                
                # Start new chunk with overlap
                if overlap_tokens > 0 and len(current_chunk) > 0:
                    # Calculate how many sentences to keep for overlap
                    overlap_sentences = []
                    overlap_token_count = 0
                    
                    for sent in reversed(current_chunk):
                        sent_tokens = len(self.encoder.encode(sent))
                        if overlap_token_count + sent_tokens <= overlap_tokens:
                            overlap_sentences.insert(0, sent)
                            overlap_token_count += sent_tokens
                        else:
                            break
                    
                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = overlap_token_count + sentence_tokens
                    chunk_start_token = total_tokens_processed - overlap_token_count
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                    chunk_start_token = total_tokens_processed
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            
            total_tokens_processed += sentence_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start_token': chunk_start_token,
                'end_token': total_tokens_processed,
                'token_count': current_tokens,
                'method': 'sentence_aware'
            })
        
        return chunks
    
    def chunk_code(
        self,
        code: str,
        language: str = "python",
        max_tokens: int = 512,
        overlap_tokens: int = 100
    ) -> List[dict]:
        """
        Special chunking for code that tries to preserve function/class boundaries.
        
        Args:
            code: Code to chunk
            language: Programming language
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap between chunks
            
        Returns:
            List of chunk dictionaries
        """
        # For now, use line-based chunking for code
        lines = code.split('\n')
        
        chunks = []
        current_chunk_lines = []
        current_tokens = 0
        chunk_start_line = 0
        
        for i, line in enumerate(lines):
            line_tokens = len(self.encoder.encode(line + '\n'))
            
            if current_tokens + line_tokens > max_tokens and current_chunk_lines:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk_lines)
                chunks.append({
                    'text': chunk_text,
                    'start_line': chunk_start_line,
                    'end_line': i,
                    'token_count': current_tokens,
                    'method': 'code_aware',
                    'language': language
                })
                
                # Calculate overlap
                if overlap_tokens > 0:
                    overlap_lines = []
                    overlap_token_count = 0
                    
                    for line_text in reversed(current_chunk_lines):
                        line_token_count = len(self.encoder.encode(line_text + '\n'))
                        if overlap_token_count + line_token_count <= overlap_tokens:
                            overlap_lines.insert(0, line_text)
                            overlap_token_count += line_token_count
                        else:
                            break
                    
                    current_chunk_lines = overlap_lines + [line]
                    current_tokens = overlap_token_count + line_tokens
                    chunk_start_line = i - len(overlap_lines)
                else:
                    current_chunk_lines = [line]
                    current_tokens = line_tokens
                    chunk_start_line = i
            else:
                current_chunk_lines.append(line)
                current_tokens += line_tokens
        
        # Don't forget the last chunk
        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            chunks.append({
                'text': chunk_text,
                'start_line': chunk_start_line,
                'end_line': len(lines),
                'token_count': current_tokens,
                'method': 'code_aware',
                'language': language
            })
        
        return chunks
    
    def adaptive_chunk(
        self,
        text: str,
        doc_type: str = "general",
        max_tokens: int = 512
    ) -> List[dict]:
        """
        Adaptive chunking based on document type.
        
        Args:
            text: Text to chunk
            doc_type: Type of document (general, code, technical, meeting, etc.)
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of chunk dictionaries
        """
        # Adjust parameters based on document type
        if doc_type == "code":
            return self.chunk_code(text, max_tokens=max_tokens, overlap_tokens=100)
        elif doc_type == "meeting":
            # Meetings benefit from larger chunks with more overlap
            return self.chunk_with_boundaries(text, max_tokens=max_tokens, 
                                             overlap_tokens=100, respect_sentences=True)
        elif doc_type == "technical":
            # Technical docs need good overlap for context
            return self.chunk_with_boundaries(text, max_tokens=max_tokens, 
                                             overlap_tokens=75, respect_sentences=True)
        else:
            # General documents
            return self.chunk_with_boundaries(text, max_tokens=max_tokens, 
                                             overlap_tokens=50, respect_sentences=True)


class ChunkOptimizer:
    """Optimize chunks for specific models and use cases."""
    
    # Model context windows (approximate)
    MODEL_CONTEXTS = {
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "claude-2": 100000,
        "claude-instant": 100000,
        "llama-2-7b": 4096,
        "llama-2-13b": 4096,
        "llama-2-70b": 4096,
        "mistral-7b": 8192,
        "gpt-oss:20b": 4096,  # Your model
        "embeddinggemma:300m": 2048,  # Your embedding model
    }
    
    @classmethod
    def get_optimal_chunk_size(
        cls,
        model_name: str,
        reserve_for_prompt: int = 1000,
        reserve_for_response: int = 1000
    ) -> int:
        """
        Get optimal chunk size for a model.
        
        Args:
            model_name: Name of the model
            reserve_for_prompt: Tokens to reserve for prompt
            reserve_for_response: Tokens to reserve for response
            
        Returns:
            Optimal chunk size in tokens
        """
        context_size = cls.MODEL_CONTEXTS.get(model_name, 4096)
        available = context_size - reserve_for_prompt - reserve_for_response
        
        # Use 25% of available context for each chunk to allow multiple chunks
        return min(512, available // 4)
    
    @classmethod
    def validate_chunks(
        cls,
        chunks: List[dict],
        model_name: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate chunks for a specific model.
        
        Args:
            chunks: List of chunk dictionaries
            model_name: Target model name
            
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        max_chunk_size = cls.get_optimal_chunk_size(model_name)
        
        for i, chunk in enumerate(chunks):
            token_count = chunk.get('token_count', 0)
            if token_count > max_chunk_size:
                warnings.append(
                    f"Chunk {i} has {token_count} tokens, "
                    f"exceeds recommended {max_chunk_size} for {model_name}"
                )
        
        return len(warnings) == 0, warnings