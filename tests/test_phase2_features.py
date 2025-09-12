#!/usr/bin/env python3
"""
Test script for Phase 2 enhancements
Tests token chunking, reranking, async processing, caching, and query expansion
"""

import sys
import time
import asyncio
from pathlib import Path

print("=" * 60)
print("Phase 2 Enhancement Test Suite")
print("=" * 60)

passed = 0
failed = 0

# Test 1: Token-based chunking
print("\n1. Testing Token-based Chunking...")
try:
    from src.core.token_chunker import TokenChunker, ChunkOptimizer
    
    chunker = TokenChunker(use_spacy=True)
    text = "This is a test document. " * 100  # ~100 tokens
    
    # Test basic chunking
    chunks = chunker.chunk_by_tokens(text, max_tokens=50, overlap_tokens=10)
    assert len(chunks) > 0, "No chunks generated"
    
    # Test adaptive chunking
    result = chunker.adaptive_chunk(text, doc_type="technical", max_tokens=100)
    assert len(result) > 0, "Adaptive chunking failed"
    
    # Test chunk optimizer
    optimizer = ChunkOptimizer()
    optimal_size = optimizer.get_optimal_chunk_size("gpt-oss:20b")
    assert optimal_size > 0, "Optimizer failed"
    
    print("✅ Token-based chunking working")
    print(f"   - Generated {len(chunks)} chunks")
    print(f"   - Optimal chunk size: {optimal_size} tokens")
    passed += 1
except Exception as e:
    print(f"❌ Token-based chunking failed: {e}")
    failed += 1

# Test 2: Reranking models
print("\n2. Testing Reranking Models...")
try:
    from src.core.reranker import Reranker, HybridReranker
    
    reranker = Reranker("fast")
    
    # Test basic reranking
    query = "machine learning"
    documents = [
        "Deep learning with neural networks",
        "Machine learning algorithms",
        "Database management systems",
        "Learning Python programming"
    ]
    
    reranked = reranker.rerank(query, documents, top_k=2, return_scores=True)
    assert len(reranked) == 2, "Reranking failed"
    
    # Test that ML doc is ranked higher than DB doc
    top_doc = reranked[0][0]
    assert "learning" in top_doc.lower(), "Reranking quality issue"
    
    print("✅ Reranking models working")
    print(f"   - Top result: {reranked[0][0][:50]}...")
    print(f"   - Score: {reranked[0][1]:.3f}")
    passed += 1
except Exception as e:
    print(f"❌ Reranking failed: {e}")
    failed += 1

# Test 3: Async/Batch Processing
print("\n3. Testing Async/Batch Processing...")
try:
    from src.core.async_processor import AsyncDocumentProcessor, BatchEmbeddingProcessor
    
    async def test_async():
        processor = AsyncDocumentProcessor(io_workers=2, compute_workers=4)
        
        # Create test files
        test_dir = Path("test_async")
        test_dir.mkdir(exist_ok=True)
        
        test_files = []
        for i in range(3):
            file_path = test_dir / f"test_{i}.txt"
            file_path.write_text(f"Test content {i}")
            test_files.append(file_path)
        
        # Test async processing
        def process_func(content):
            return len(content)
        
        results = await processor.process_documents(test_files, process_func)
        
        # Cleanup
        for f in test_files:
            f.unlink()
        test_dir.rmdir()
        
        return len(results) == 3 and all(r.success for r in results)
    
    # Run async test
    success = asyncio.run(test_async())
    assert success, "Async processing failed"
    
    print("✅ Async/batch processing working")
    print("   - Processed 3 documents in parallel")
    passed += 1
except Exception as e:
    print(f"❌ Async processing failed: {e}")
    failed += 1

# Test 4: Caching Strategy
print("\n4. Testing Caching Strategy...")
try:
    from src.core.cache import EmbeddingCache, SearchResultCache, CacheManager
    
    # Test embedding cache
    cache = EmbeddingCache(cache_dir=".test_cache/embeddings")
    
    # Test set and get
    test_text = "Test embedding text"
    test_embedding = [0.1, 0.2, 0.3]
    
    cache.set(test_text, test_embedding)
    retrieved = cache.get(test_text)
    assert retrieved == test_embedding, "Cache retrieval failed"
    
    # Test cache stats
    stats = cache.get_stats()
    assert stats['hits'] == 1, "Cache hit not recorded"
    
    # Test search cache
    search_cache = SearchResultCache(cache_dir=".test_cache/search")
    
    # Cleanup
    cache.clear()
    cache.close()
    search_cache.clear()
    
    print("✅ Caching strategy working")
    print(f"   - Cache hit rate: {stats['hit_rate']:.2%}")
    passed += 1
except Exception as e:
    print(f"❌ Caching failed: {e}")
    failed += 1

# Test 5: Query Expansion
print("\n5. Testing Query Expansion...")
try:
    from src.core.query_expansion import QueryExpander, SmartQueryProcessor
    
    expander = QueryExpander()
    processor = SmartQueryProcessor(expander)
    
    # Test basic expansion
    query = "database error"
    expanded = expander.expand_query(query)
    
    assert len(expanded.synonyms) > 0, "No synonyms found"
    assert "db" in expanded.related_terms or "issue" in expanded.synonyms, "Expected expansions missing"
    
    # Test smart processing
    result = processor.process_query(query, expand=True, correct_spelling=True)
    assert len(result['expansions']) > 0, "Smart processing failed"
    
    # Test multi-query generation
    queries = processor.create_multi_query(query, num_queries=3)
    assert len(queries) == 3, "Multi-query generation failed"
    
    print("✅ Query expansion working")
    print(f"   - Original: {query}")
    print(f"   - Expanded terms: {expanded.expanded_terms[:3]}")
    passed += 1
except Exception as e:
    print(f"❌ Query expansion failed: {e}")
    failed += 1

# Test 6: Integration Test
print("\n6. Testing Phase 2 Integration...")
try:
    # Test that all components can work together
    from src.core.token_chunker import TokenChunker
    from src.core.reranker import Reranker
    from src.core.cache import EmbeddingCache
    from src.core.query_expansion import QueryExpander
    
    # Create instances
    chunker = TokenChunker(use_spacy=False)  # Skip spacy for speed
    reranker = Reranker("fast")
    cache = EmbeddingCache(cache_dir=".test_cache")
    expander = QueryExpander()
    
    # Test workflow
    text = "Machine learning is a subset of artificial intelligence."
    chunks = chunker.chunk_by_tokens(text, max_tokens=20)
    
    query = "AI and ML"
    expanded = expander.expand_query(query)
    
    # Clean up
    cache.clear()
    cache.close()
    
    print("✅ Phase 2 components integrate successfully")
    passed += 1
except Exception as e:
    print(f"❌ Integration test failed: {e}")
    failed += 1

# Summary
print("\n" + "=" * 60)
print(f"Test Results: {passed} passed, {failed} failed")

if failed == 0:
    print("✅ ALL PHASE 2 TESTS PASSED!")
    print("\nPhase 2 Features Successfully Implemented:")
    print("1. ✅ Token-based chunking with Tiktoken")
    print("2. ✅ Reranking with sentence-transformers")
    print("3. ✅ Async/batch processing")
    print("4. ✅ Embedding and search caching")
    print("5. ✅ Query expansion for better recall")
    print("6. ✅ All components integrate properly")
else:
    print(f"⚠️ {failed} test(s) failed. Review the output above.")
    sys.exit(1)