"""
Handle meta-queries about the document index itself.
Provides direct answers for questions about indexed documents.
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
from .database import DatabaseManager

logger = logging.getLogger(__name__)


class MetaQueryHandler:
    """Handles queries about the indexed documents themselves."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize the meta query handler."""
        self.db_manager = db_manager or DatabaseManager()
        
        # Keywords that indicate meta-queries
        self.meta_keywords = {
            'count': ['how many', 'number of', 'count', 'total'],
            'list': ['what documents', 'which documents', 'list all', 'show all', 'what files', 'which files', 'indexed documents', 'available documents'],
            'stats': ['statistics', 'summary', 'overview', 'index info']
        }
    
    def is_meta_query(self, query: str) -> Optional[str]:
        """
        Check if a query is asking about the index itself.
        
        Returns:
            Query type ('count', 'list', 'stats') or None if not a meta query
        """
        query_lower = query.lower()
        
        # Check for document/file/index related terms
        if not any(term in query_lower for term in ['document', 'file', 'index', 'available', 'stored']):
            return None
        
        # Check for meta query patterns
        for query_type, keywords in self.meta_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return query_type
        
        return None
    
    def handle_meta_query(self, query: str, query_type: str) -> Dict[str, Any]:
        """
        Handle a meta query and return formatted response.
        
        Args:
            query: The user's query
            query_type: Type of meta query ('count', 'list', 'stats')
            
        Returns:
            Dictionary with response data
        """
        try:
            # Get indexed documents information
            table = self.db_manager.get_table()
            df = table.to_pandas()
            
            if df.empty:
                return {
                    'success': True,
                    'type': query_type,
                    'message': "No documents are currently indexed.",
                    'data': {}
                }
            
            # Get unique documents and their stats
            doc_stats = df.groupby('source').agg({
                'id': 'count',
                'timestamp': 'first',
                'doc_type': 'first',
                'file_modified': 'first'
            }).reset_index()
            doc_stats.columns = ['document', 'chunks', 'indexed_at', 'type', 'modified']
            doc_stats = doc_stats.sort_values('indexed_at', ascending=False)
            
            if query_type == 'count':
                return {
                    'success': True,
                    'type': 'count',
                    'message': f"📊 **{len(doc_stats)} documents** are currently indexed with a total of **{df.shape[0]} chunks**.",
                    'data': {
                        'document_count': len(doc_stats),
                        'total_chunks': df.shape[0],
                        'avg_chunks_per_doc': df.shape[0] // len(doc_stats) if len(doc_stats) > 0 else 0
                    }
                }
            
            elif query_type == 'list':
                # Format document list
                doc_list = []
                for _, row in doc_stats.iterrows():
                    doc_info = f"• **{row['document']}**"
                    if pd.notna(row['type']):
                        doc_info += f" ({row['type']})"
                    doc_info += f" - {row['chunks']} chunks"
                    doc_list.append(doc_info)
                
                return {
                    'success': True,
                    'type': 'list',
                    'message': f"📚 **{len(doc_stats)} documents indexed:**\n\n" + "\n".join(doc_list),
                    'data': {
                        'documents': doc_stats.to_dict('records')
                    }
                }
            
            elif query_type == 'stats':
                # Calculate statistics
                doc_types = df['doc_type'].value_counts().to_dict() if 'doc_type' in df.columns else {}
                
                stats_msg = f"""📊 **Index Statistics:**

• **Total Documents:** {len(doc_stats)}
• **Total Chunks:** {df.shape[0]}
• **Average Chunks per Document:** {df.shape[0] // len(doc_stats) if len(doc_stats) > 0 else 0}

**Document Types:**
"""
                for doc_type, count in doc_types.items():
                    stats_msg += f"• {doc_type}: {count} chunks\n"
                
                return {
                    'success': True,
                    'type': 'stats',
                    'message': stats_msg,
                    'data': {
                        'document_count': len(doc_stats),
                        'total_chunks': df.shape[0],
                        'document_types': doc_types
                    }
                }
            
            return {
                'success': False,
                'message': "Unknown meta query type"
            }
            
        except Exception as e:
            logger.error(f"Failed to handle meta query: {e}")
            return {
                'success': False,
                'message': f"Error retrieving index information: {str(e)}"
            }
    
    def get_quick_stats(self) -> str:
        """Get a quick one-line summary of indexed documents."""
        try:
            table = self.db_manager.get_table()
            df = table.to_pandas()
            
            if df.empty:
                return "No documents indexed"
            
            doc_count = df['source'].nunique()
            chunk_count = len(df)
            
            return f"{doc_count} documents, {chunk_count} chunks indexed"
            
        except Exception:
            return "Index information unavailable"