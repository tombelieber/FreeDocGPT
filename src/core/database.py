import logging
from typing import Optional

import lancedb
import pandas as pd
import pyarrow as pa

from ..config import get_settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages LanceDB database operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.db = None
        self.table = None
        self._connect()
    
    def _connect(self):
        """Connect to the database."""
        try:
            self.db = lancedb.connect(self.settings.db_dir)
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def get_table(self):
        """Get or create the documents table."""
        if self.table is not None:
            return self.table
            
        if self.settings.table_name not in self.db.table_names():
            schema = pa.schema([
                pa.field("id", pa.int64()),
                pa.field("source", pa.string()),
                pa.field("chunk", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), 768)),
                pa.field("timestamp", pa.string()),
                # Enhanced metadata fields
                pa.field("content_hash", pa.string()),  # SHA-256 or xxHash
                pa.field("doc_type", pa.string()),  # meeting, prd, technical, wiki, general
                pa.field("language", pa.string()),  # english, chinese_simplified, etc.
                pa.field("chunk_index", pa.int32()),  # Position in document
                pa.field("total_chunks", pa.int32()),  # Total chunks in document
                pa.field("page_number", pa.int32()),  # For PDFs
                pa.field("section_header", pa.string()),  # Section/chapter title
                pa.field("file_modified", pa.timestamp('ms')),  # File modification time
            ])
            self.table = self.db.create_table(
                self.settings.table_name, 
                data=[], 
                schema=schema
            )
        else:
            self.table = self.db.open_table(self.settings.table_name)
        
        return self.table
    
    def get_indexed_documents(self) -> pd.DataFrame:
        """Get list of indexed documents with statistics."""
        try:
            table = self.get_table()
            df = table.to_pandas()
            
            if df.empty:
                return pd.DataFrame()
            
            # Group by source to get document statistics
            stats = df.groupby("source").agg({
                "id": "count", 
                "timestamp": "first"
            }).reset_index()
            
            stats.columns = ["Document", "Chunks", "Indexed At"]
            stats = stats.sort_values("Indexed At", ascending=False)
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get indexed documents: {e}")
            return pd.DataFrame()
    
    def add_documents(self, rows: list):
        """Add documents to the table."""
        try:
            table = self.get_table()
            table.add(rows)
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def search(self, embedding: list, limit: int = 5) -> Optional[pd.DataFrame]:
        """Search for similar documents."""
        try:
            table = self.get_table()
            results = table.search(embedding).limit(limit).to_pandas()
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return None
    
    def clear_index(self):
        """Clear all indexed documents."""
        try:
            if self.settings.table_name in self.db.table_names():
                self.db.drop_table(self.settings.table_name)
                self.table = None
                return True
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False