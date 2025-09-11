"""
Query analysis module for intelligent search routing.
Analyzes user queries to determine relevant document types and keywords.
"""

import logging
from typing import Dict, List, Optional, Set
import ollama

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyzes queries to optimize search targeting."""
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize the query analyzer.
        
        Args:
            model: LLM model to use for analysis
        """
        from ..config import get_settings
        settings = get_settings()
        self.model = model or settings.gen_model
        
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract key terms from text using LLM.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        # Take a reasonable sample for keyword extraction
        sample = text[:2000] if len(text) > 2000 else text
        
        prompt = f"""Extract the {max_keywords} most important keywords or key phrases from this text.
Return ONLY the keywords, separated by commas, no explanations.
Focus on nouns, technical terms, and specific concepts.

Text: {sample}

Keywords:"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 100}
            )
            
            # Parse keywords from response
            keywords_raw = response['response'].strip()
            keywords = [k.strip().lower() for k in keywords_raw.split(',')]
            keywords = [k for k in keywords if k and len(k) > 2][:max_keywords]
            
            return keywords
            
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            # Fallback: simple extraction of capitalized words
            words = sample.split()
            keywords = []
            for word in words:
                if word[0].isupper() and len(word) > 3:
                    keywords.append(word.lower())
                if len(keywords) >= max_keywords:
                    break
            return keywords
    
    def analyze_query(self, query: str) -> Dict:
        """
        Analyze a query to determine search parameters.
        
        Args:
            query: User's search query
            
        Returns:
            Dictionary with analysis results including:
            - keywords: Extracted query keywords for boosting
            - search_scope: Recommended search scope (focused/broad)
            - doc_type_hint: Hint about likely document type
            - temporal_hint: Whether query is looking for recent documents
        """
        query_lower = query.lower()
        
        # Detect document type hints from query
        doc_type_hint = None
        if any(word in query_lower for word in ['meeting', 'standup', 'retro', 'retrospective', 'minutes']):
            doc_type_hint = 'meeting'
        elif any(word in query_lower for word in ['prd', 'product requirement', 'specification', 'spec']):
            doc_type_hint = 'prd'
        elif any(word in query_lower for word in ['technical', 'architecture', 'implementation', 'code', 'api']):
            doc_type_hint = 'technical'
        elif any(word in query_lower for word in ['wiki', 'documentation', 'guide', 'manual']):
            doc_type_hint = 'wiki'
        
        # Detect temporal hints
        temporal_hint = 'recent' if any(word in query_lower for word in ['last', 'latest', 'recent', 'newest']) else None
        
        # Extract keywords from query for potential boosting
        keywords = self.extract_keywords(query, max_keywords=5)
        
        # Add document type keywords if detected
        if doc_type_hint == 'meeting':
            keywords.extend(['meeting', 'agenda', 'discussion'])
        elif doc_type_hint == 'prd':
            keywords.extend(['requirement', 'product', 'feature'])
        
        # Remove duplicates while preserving order
        seen = set()
        keywords = [k for k in keywords if not (k in seen or seen.add(k))][:8]
        
        # Simple heuristic for search scope based on query length and specificity
        query_words = query.split()
        if len(query_words) > 5 or any(word in query_lower for word in ['specific', 'exact', 'particular']):
            search_scope = "focused"
        else:
            search_scope = "broad"
        
        return {
            "keywords": keywords,
            "search_scope": search_scope,
            "doc_type_hint": doc_type_hint,
            "temporal_hint": temporal_hint
        }
    
    def should_expand_search(self, initial_results_quality: float, threshold: float = 0.5) -> bool:
        """
        Determine if search should be expanded based on initial results.
        
        Args:
            initial_results_quality: Quality score of initial results (0-1)
            threshold: Quality threshold below which to expand
            
        Returns:
            True if search should be expanded
        """
        return initial_results_quality < threshold