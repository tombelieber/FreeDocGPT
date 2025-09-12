"""
Query expansion for improved search recall.
Uses multiple techniques to expand and improve queries.
"""

import logging
from typing import List, Set, Dict, Optional
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExpandedQuery:
    """Result of query expansion."""
    original: str
    expanded_terms: List[str]
    synonyms: List[str]
    related_terms: List[str]
    variations: List[str]
    combined_query: str


class QueryExpander:
    """Expand queries for better search recall."""
    
    def __init__(self):
        """Initialize query expander with common patterns and synonyms."""
        # Common technical synonyms
        self.tech_synonyms = {
            "bug": ["issue", "problem", "error", "defect", "fault"],
            "fix": ["repair", "resolve", "patch", "solution", "correction"],
            "code": ["source", "program", "script", "implementation"],
            "function": ["method", "procedure", "routine", "operation"],
            "variable": ["parameter", "argument", "value", "field"],
            "database": ["db", "datastore", "storage", "repository"],
            "api": ["interface", "endpoint", "service", "REST"],
            "test": ["check", "verify", "validate", "QA", "testing"],
            "deploy": ["release", "publish", "launch", "rollout"],
            "config": ["configuration", "settings", "preferences", "setup"],
            "doc": ["documentation", "docs", "manual", "guide"],
            "auth": ["authentication", "authorization", "login", "security"],
            "user": ["customer", "client", "person", "account"],
            "log": ["logging", "logs", "trace", "debug", "record"],
            "error": ["exception", "failure", "crash", "fault"],
            "performance": ["speed", "efficiency", "optimization", "perf"],
            "cache": ["caching", "buffer", "store", "memory"],
            "search": ["query", "find", "lookup", "retrieve"],
            "index": ["indexing", "catalog", "directory"],
            "async": ["asynchronous", "concurrent", "parallel", "non-blocking"]
        }
        
        # Common abbreviations and their expansions
        self.abbreviations = {
            "db": "database",
            "api": "application programming interface",
            "ui": "user interface",
            "ux": "user experience",
            "qa": "quality assurance",
            "ci": "continuous integration",
            "cd": "continuous deployment",
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "nlp": "natural language processing",
            "rag": "retrieval augmented generation",
            "llm": "large language model",
            "gpu": "graphics processing unit",
            "cpu": "central processing unit",
            "ram": "random access memory",
            "ssd": "solid state drive",
            "http": "hypertext transfer protocol",
            "sql": "structured query language",
            "json": "javascript object notation",
            "xml": "extensible markup language",
            "csv": "comma separated values",
            "pdf": "portable document format",
            "env": "environment",
            "config": "configuration",
            "auth": "authentication",
            "admin": "administrator",
            "dev": "development",
            "prod": "production",
            "perf": "performance"
        }
        
        # Common word variations
        self.word_variations = {
            "running": ["run", "runs", "ran", "runner"],
            "writing": ["write", "writes", "wrote", "written", "writer"],
            "reading": ["read", "reads", "reader"],
            "processing": ["process", "processes", "processed", "processor"],
            "indexing": ["index", "indexes", "indexed", "indexer"],
            "searching": ["search", "searches", "searched", "searcher"],
            "caching": ["cache", "caches", "cached"],
            "testing": ["test", "tests", "tested", "tester"],
            "debugging": ["debug", "debugs", "debugged", "debugger"],
            "deploying": ["deploy", "deploys", "deployed", "deployment"]
        }
    
    def expand_query(self, query: str) -> ExpandedQuery:
        """
        Expand a query with synonyms, variations, and related terms.
        
        Args:
            query: Original query string
            
        Returns:
            ExpandedQuery object with expansion results
        """
        query_lower = query.lower()
        words = query_lower.split()
        
        # Collect expansions
        synonyms = []
        related_terms = []
        variations = []
        
        for word in words:
            # Get synonyms
            if word in self.tech_synonyms:
                synonyms.extend(self.tech_synonyms[word])
            
            # Expand abbreviations
            if word in self.abbreviations:
                related_terms.append(self.abbreviations[word])
            
            # Check if word is an expanded form
            for abbr, expanded in self.abbreviations.items():
                if word in expanded.lower():
                    related_terms.append(abbr)
            
            # Get word variations
            for base, vars in self.word_variations.items():
                if word == base or word in vars:
                    variations.extend([v for v in vars if v != word])
        
        # Remove duplicates
        synonyms = list(set(synonyms))
        related_terms = list(set(related_terms))
        variations = list(set(variations))
        
        # Create expanded terms list
        expanded_terms = list(set(synonyms + related_terms + variations))
        
        # Create combined query
        all_terms = [query] + expanded_terms[:5]  # Limit expansion to avoid query explosion
        combined_query = " OR ".join(f'"{term}"' if " " in term else term for term in all_terms)
        
        return ExpandedQuery(
            original=query,
            expanded_terms=expanded_terms,
            synonyms=synonyms,
            related_terms=related_terms,
            variations=variations,
            combined_query=combined_query
        )
    
    def extract_key_phrases(self, query: str) -> List[str]:
        """
        Extract key phrases from query.
        
        Args:
            query: Query string
            
        Returns:
            List of key phrases
        """
        # Simple phrase extraction using quotes
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        
        # Extract potential multi-word phrases (2-3 words)
        words = query.lower().split()
        phrases = []
        
        for i in range(len(words) - 1):
            # Two-word phrases
            phrase = f"{words[i]} {words[i+1]}"
            if len(words[i]) > 2 and len(words[i+1]) > 2:  # Skip short words
                phrases.append(phrase)
            
            # Three-word phrases
            if i < len(words) - 2:
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                if len(words[i]) > 2 and len(words[i+2]) > 2:
                    phrases.append(phrase)
        
        return quoted_phrases + phrases
    
    def generate_query_variations(self, query: str) -> List[str]:
        """
        Generate variations of the query.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations
        """
        variations = [query]  # Include original
        
        # Add question forms
        if not query.endswith("?"):
            variations.append(f"{query}?")
            variations.append(f"What is {query}?")
            variations.append(f"How to {query}?")
            variations.append(f"Where is {query}?")
        
        # Add statement forms
        if query.startswith(("what", "how", "where", "when", "why", "who")):
            # Remove question words
            statement = re.sub(r"^(what|how|where|when|why|who)\s+(is|are|to|does|do)\s+", "", query.lower())
            variations.append(statement)
        
        # Add common prefixes
        if not any(query.lower().startswith(p) for p in ["find", "search", "get", "show"]):
            variations.append(f"find {query}")
            variations.append(f"search for {query}")
        
        return list(set(variations))[:5]  # Limit to 5 variations
    
    def apply_spell_correction(self, query: str) -> Optional[str]:
        """
        Apply basic spell correction (placeholder for more advanced implementation).
        
        Args:
            query: Query string
            
        Returns:
            Corrected query or None if no corrections
        """
        # Common typos in technical terms
        corrections = {
            "databse": "database",
            "fucntion": "function",
            "vairable": "variable",
            "authentification": "authentication",
            "authorisation": "authorization",
            "deployement": "deployment",
            "configuraiton": "configuration",
            "documetnation": "documentation",
            "performace": "performance",
            "asyncronous": "asynchronous",
            "indxing": "indexing",
            "serach": "search",
            "cahce": "cache",
            "errir": "error",
            "debig": "debug",
            "tets": "test",
            "pyton": "python",
            "javascrpt": "javascript",
            "typscript": "typescript"
        }
        
        corrected = query
        changed = False
        
        for typo, correct in corrections.items():
            if typo in query.lower():
                corrected = re.sub(typo, correct, corrected, flags=re.IGNORECASE)
                changed = True
        
        return corrected if changed else None


class SmartQueryProcessor:
    """Advanced query processing with context awareness."""
    
    def __init__(self, query_expander: Optional[QueryExpander] = None):
        """
        Initialize smart query processor.
        
        Args:
            query_expander: Query expander instance
        """
        self.expander = query_expander or QueryExpander()
    
    def process_query(
        self,
        query: str,
        expand: bool = True,
        correct_spelling: bool = True,
        extract_phrases: bool = True
    ) -> Dict[str, any]:
        """
        Process query with multiple techniques.
        
        Args:
            query: Original query
            expand: Whether to expand query
            correct_spelling: Whether to correct spelling
            extract_phrases: Whether to extract key phrases
            
        Returns:
            Dictionary with processed query information
        """
        result = {
            "original": query,
            "processed": query,
            "expansions": [],
            "phrases": [],
            "corrections": None,
            "variations": []
        }
        
        # Spell correction
        if correct_spelling:
            corrected = self.expander.apply_spell_correction(query)
            if corrected:
                result["corrections"] = corrected
                query = corrected  # Use corrected version for further processing
        
        # Query expansion
        if expand:
            expanded = self.expander.expand_query(query)
            result["expansions"] = expanded.expanded_terms
            result["processed"] = expanded.combined_query
        
        # Extract key phrases
        if extract_phrases:
            result["phrases"] = self.expander.extract_key_phrases(query)
        
        # Generate variations
        result["variations"] = self.expander.generate_query_variations(query)
        
        return result
    
    def optimize_for_search_type(
        self,
        query: str,
        search_type: str = "hybrid"
    ) -> str:
        """
        Optimize query for specific search type.
        
        Args:
            query: Original query
            search_type: Type of search (vector, keyword, hybrid)
            
        Returns:
            Optimized query
        """
        if search_type == "keyword":
            # For keyword search, use exact terms and phrases
            phrases = self.expander.extract_key_phrases(query)
            if phrases:
                return " AND ".join(f'"{p}"' for p in phrases[:3])
            return query
        
        elif search_type == "vector":
            # For vector search, expand with synonyms
            expanded = self.expander.expand_query(query)
            return f"{query} {' '.join(expanded.synonyms[:3])}"
        
        else:  # hybrid
            # Balanced approach
            expanded = self.expander.expand_query(query)
            return expanded.combined_query
    
    def create_multi_query(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Create multiple query variations for ensemble search.
        
        Args:
            query: Original query
            num_queries: Number of queries to generate
            
        Returns:
            List of query variations
        """
        queries = [query]  # Start with original
        
        # Add expanded version
        expanded = self.expander.expand_query(query)
        if expanded.expanded_terms:
            queries.append(f"{query} {' '.join(expanded.expanded_terms[:2])}")
        
        # Add question variation
        variations = self.expander.generate_query_variations(query)
        queries.extend(variations[:num_queries - len(queries)])
        
        # Ensure we have the requested number
        while len(queries) < num_queries:
            queries.append(query)
        
        return queries[:num_queries]
    
    def reformulate_with_context(
        self, 
        query: str, 
        conversation_history: List[Dict]
    ) -> str:
        """
        Reformulate query using conversation context to resolve pronouns and references.
        
        Args:
            query: Current user query that may contain pronouns/references
            conversation_history: Previous conversation messages
            
        Returns:
            Reformulated query with resolved references
        """
        # Check if query contains pronouns or references that need context
        pronouns = ["it", "its", "that", "this", "these", "those", "them", "they", "their"]
        query_lower = query.lower()
        
        # Check if query likely needs context
        needs_context = False
        for pronoun in pronouns:
            if re.search(r'\b' + pronoun + r'\b', query_lower):
                needs_context = True
                break
        
        # Also check for follow-up patterns
        follow_up_patterns = [
            r'^(what|how|why|when|where|who) about',
            r'^(tell|show|explain|describe) (me )?more',
            r'^(and|also|but|however)',
            r'^(can|could|would|should) (you|it|that)',
        ]
        
        for pattern in follow_up_patterns:
            if re.match(pattern, query_lower):
                needs_context = True
                break
        
        if not needs_context or not conversation_history:
            return query
        
        # Extract context from recent messages
        recent_context = []
        for msg in conversation_history[-4:]:  # Look at last 2 exchanges
            if msg["role"] == "user":
                recent_context.append(f"User asked: {msg['content']}")
            elif msg["role"] == "assistant":
                # Extract key topics from assistant response (first 200 chars)
                content = msg["content"][:200] if len(msg["content"]) > 200 else msg["content"]
                recent_context.append(f"Assistant discussed: {content}")
        
        # Build reformulated query
        if recent_context:
            # Simple heuristic reformulation
            context_summary = " ".join(recent_context[-2:])  # Last exchange
            
            # Try to extract the main topic from recent context
            # Look for nouns in the last user message
            last_user_msg = None
            for msg in reversed(conversation_history):
                if msg["role"] == "user":
                    last_user_msg = msg["content"]
                    break
            
            if last_user_msg:
                # Simple noun extraction (words that are likely topics)
                words = last_user_msg.split()
                potential_topics = []
                for word in words:
                    # Skip common words and keep potential topic words
                    if len(word) > 3 and word.lower() not in [
                        "what", "when", "where", "how", "why", "which",
                        "about", "with", "from", "that", "this", "these",
                        "those", "have", "been", "were", "will", "would",
                        "could", "should", "does", "doing", "done"
                    ]:
                        potential_topics.append(word)
                
                if potential_topics:
                    # Add context to query
                    topic_context = " ".join(potential_topics[:3])
                    
                    # Replace pronouns with likely topics
                    reformulated = query
                    if "it" in query_lower or "its" in query_lower:
                        reformulated = f"{query} (referring to {topic_context})"
                    elif "that" in query_lower or "this" in query_lower:
                        reformulated = f"{query} (about {topic_context})"
                    else:
                        reformulated = f"{query} (in context of {topic_context})"
                    
                    logger.info(f"Query reformulated: '{query}' -> '{reformulated}'")
                    return reformulated
        
        return query