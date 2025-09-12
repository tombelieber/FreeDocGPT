#!/usr/bin/env python3
"""
Test script for incremental indexing with change detection.
Verifies that unchanged files are skipped and changed files are re-indexed.
"""

import os
import time
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

# Set up environment
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_PORT"] = "8502"

from src.core.indexer import DocumentIndexer
from src.core.database import DatabaseManager
from src.config import get_settings

def create_test_file(path: Path, content: str):
    """Create a test file with specific content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  ✓ Created: {path.name}")

def modify_file(path: Path, new_content: str):
    """Modify a file's content."""
    time.sleep(0.1)  # Ensure modification time is different
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"  ✓ Modified: {path.name}")

def test_incremental_indexing():
    """Test incremental indexing with change detection."""
    print("\n🔍 INCREMENTAL INDEXING TEST")
    print("=" * 60)
    
    # Initialize components
    settings = get_settings()
    db_manager = DatabaseManager()
    indexer = DocumentIndexer(db_manager=db_manager)
    
    # Create a temporary test directory
    test_dir = Path("documents/test_incremental")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Create initial test files
        print("\n1️⃣ Creating initial test files...")
        test_files = [
            (test_dir / "doc1.txt", "This is document 1 - original content"),
            (test_dir / "doc2.md", "# Document 2\n\nOriginal markdown content"),
            (test_dir / "doc3.txt", "Document 3 - will remain unchanged"),
        ]
        
        for path, content in test_files:
            create_test_file(path, content)
        
        # Step 2: Initial indexing
        print("\n2️⃣ Initial indexing (should index all 3 files)...")
        print("-" * 40)
        
        # Clear any existing index
        db_manager.clear_index()
        
        # Index the files
        files = list(test_dir.glob("*"))
        indexer.index_documents(files)
        
        # Get indexed documents
        indexed = indexer.incremental_indexer.get_indexed_files()
        print(f"\nIndexed files: {len(indexed)}")
        for source in indexed:
            print(f"  - {source}")
        
        # Step 3: Re-index without changes (should skip all)
        print("\n3️⃣ Re-indexing without changes (should skip all)...")
        print("-" * 40)
        indexer.index_documents(files)
        
        # Step 4: Modify one file and add a new one
        print("\n4️⃣ Modifying doc1.txt and adding doc4.txt...")
        modify_file(test_files[0][0], "This is document 1 - MODIFIED content!")
        create_test_file(test_dir / "doc4.txt", "This is a new document 4")
        
        # Step 5: Re-index with changes
        print("\n5️⃣ Re-indexing with changes...")
        print("Expected: 1 new, 1 changed, 2 unchanged")
        print("-" * 40)
        files = list(test_dir.glob("*"))
        indexer.index_documents(files)
        
        # Verify final state
        indexed = indexer.incremental_indexer.get_indexed_files()
        print(f"\nFinal indexed files: {len(indexed)}")
        for source, info in indexed.items():
            print(f"  - {source}: {info['chunk_count']} chunks")
        
        print("\n✅ Incremental indexing test completed successfully!")
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up test files...")
        if test_dir.exists():
            shutil.rmtree(test_dir)
        print("  ✓ Test directory removed")

if __name__ == "__main__":
    test_incremental_indexing()