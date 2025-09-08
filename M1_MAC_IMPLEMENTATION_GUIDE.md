# Mac M1 Max Local Implementation Guide for RAG Improvements

## System Requirements & Advantages
- **Mac M1 Max**: 32GB unified memory, Neural Engine, Metal acceleration
- **Key Advantages**: Native ARM64 support, excellent for ML workloads, unified memory architecture

## Phase 1: Critical Improvements (Immediate)

### 1. Hybrid Search Implementation

#### **Recommended: Tantivy (Rust-based) + LanceDB**
```python
# pip install tantivy-py
```
- **Why**: Native ARM64 support, 10x faster than pure Python BM25
- **Memory efficient**: Uses 50% less RAM than Elasticsearch
- **No Java/JVM required** (unlike Elasticsearch)

**Alternative Options Evaluated:**
- ❌ Elasticsearch: Too heavy, requires JVM, 2GB+ RAM overhead
- ❌ Whoosh: Pure Python, too slow for production
- ✅ **Tantivy**: Best performance on M1, minimal memory footprint

#### Implementation:
```python
from tantivy import Index, Document, SchemaBuilder

class HybridSearch:
    def __init__(self):
        # Tantivy for BM25
        self.tantivy_index = self._create_tantivy_index()
        # Keep existing LanceDB for vectors
        self.vector_db = lancedb.connect(".lancedb")
        
    def search(self, query: str, k: int = 10):
        # Parallel search
        bm25_results = self.tantivy_search(query, k*2)
        vector_results = self.vector_search(query, k*2)
        # Reciprocal Rank Fusion (RRF)
        return self.rrf_combine(bm25_results, vector_results, k)
```

### 2. Enhanced Metadata Schema

#### **Recommended: Apache Arrow + Parquet**
Already using PyArrow with LanceDB - extend the schema:

```python
import pyarrow as pa
import hashlib

enhanced_schema = pa.schema([
    # Existing
    pa.field("id", pa.int64()),
    pa.field("source", pa.string()),
    pa.field("chunk", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 768)),
    pa.field("timestamp", pa.string()),
    
    # New metadata fields
    pa.field("content_hash", pa.string()),  # SHA-256
    pa.field("doc_type", pa.string()),
    pa.field("language", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("total_chunks", pa.int32()),
    pa.field("page_number", pa.int32()),
    pa.field("section_header", pa.string()),
    pa.field("file_modified", pa.timestamp('ms')),
])
```

### 3. Content-Based Change Detection

#### **Recommended: xxHash + SHA-256**
```python
# pip install xxhash
```
- **xxHash**: Ultra-fast for quick comparisons (5GB/s on M1)
- **SHA-256**: For cryptographic verification when needed

```python
import xxhash
import hashlib

class DocumentHasher:
    @staticmethod
    def fast_hash(content: bytes) -> str:
        """Quick hash for change detection"""
        return xxhash.xxh64(content).hexdigest()
    
    @staticmethod
    def secure_hash(content: bytes) -> str:
        """Cryptographic hash for verification"""
        return hashlib.sha256(content).hexdigest()
```

### 4. Deduplication

#### **Recommended: Faiss-CPU (ARM64 optimized)**
```bash
# Install ARM64 optimized version
pip install faiss-cpu
```

```python
import faiss
import numpy as np

class ChunkDeduplicator:
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.index = faiss.IndexFlatIP(768)  # Inner product
        
    def deduplicate(self, embeddings: np.ndarray) -> List[int]:
        """Remove near-duplicate chunks"""
        # Uses NEON SIMD on M1 for acceleration
        unique_indices = []
        for i, emb in enumerate(embeddings):
            D, I = self.index.search(emb.reshape(1, -1), 1)
            if D[0][0] < self.threshold:
                unique_indices.append(i)
                self.index.add(emb.reshape(1, -1))
        return unique_indices
```

## Phase 2: Enhanced Capabilities (1-2 Months)

### 1. Token-Based Chunking

#### **Recommended: Tiktoken (OpenAI's tokenizer)**
```python
# pip install tiktoken
```
- Native Rust implementation with Python bindings
- ARM64 optimized
- Supports multiple encodings

```python
import tiktoken

class TokenChunker:
    def __init__(self, model="cl100k_base"):
        self.encoder = tiktoken.get_encoding(model)
        
    def chunk_by_tokens(self, text: str, max_tokens: int = 512, overlap: int = 50):
        tokens = self.encoder.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i:i + max_tokens]
            chunks.append(self.encoder.decode(chunk_tokens))
        return chunks
```

### 2. Reranking Models

#### **Recommended: Sentence-Transformers with ONNX Runtime**
```python
# pip install sentence-transformers onnxruntime
```

**Best Models for M1:**
1. **ms-marco-MiniLM-L-6-v2** (Fast, 22M params)
2. **bge-reranker-base** (Accurate, 110M params)

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        # Automatically uses CoreML on M1 if available
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def rerank(self, query: str, documents: List[str], top_k: int = 5):
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        # Sort by score and return top_k
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
```

### 3. Async/Batch Processing

#### **Recommended: asyncio + aiofiles + ThreadPoolExecutor**
- No additional dependencies needed
- Leverages M1's efficiency cores for I/O
- Performance cores for compute

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles

class AsyncIndexer:
    def __init__(self):
        # Use efficiency cores for I/O
        self.io_executor = ThreadPoolExecutor(max_workers=4)
        # Use performance cores for compute
        self.compute_executor = ThreadPoolExecutor(max_workers=8)
        
    async def process_documents(self, file_paths: List[Path]):
        # Parallel document reading
        tasks = [self.read_document(fp) for fp in file_paths]
        documents = await asyncio.gather(*tasks)
        
        # Batch processing
        batch_size = 10
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            await self.process_batch(batch)
```

### 4. Performance Monitoring

#### **Recommended: Prometheus + Grafana (Lightweight)**
```python
# pip install prometheus-client
```

```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
search_latency = Histogram('search_latency_seconds', 'Search latency')
index_counter = Counter('documents_indexed_total', 'Total indexed documents')
embedding_cache_hits = Gauge('embedding_cache_hit_ratio', 'Cache hit ratio')

@search_latency.time()
def search(query: str):
    # Your search logic
    pass
```

## Technology Stack Summary

### Phase 1 Stack
| Component | Technology | Why for M1 Max |
|-----------|------------|----------------|
| BM25 Search | Tantivy | Native ARM64, Rust performance |
| Metadata | PyArrow/Parquet | Already integrated, efficient |
| Hashing | xxHash + SHA-256 | NEON SIMD acceleration |
| Deduplication | Faiss-CPU | ARM64 optimized |

### Phase 2 Stack
| Component | Technology | Why for M1 Max |
|-----------|------------|----------------|
| Tokenization | Tiktoken | Rust + ARM64 native |
| Reranking | ONNX Runtime | CoreML acceleration |
| Async | asyncio | Efficient core utilization |
| Monitoring | Prometheus | Minimal overhead |

## Memory & Performance Considerations

### Memory Usage (32GB M1 Max)
- Base system: ~4GB
- LanceDB: ~2GB
- Tantivy index: ~1GB
- Model loading: ~2GB
- Processing buffer: ~4GB
- **Total: ~13GB** (40% utilization)

### Performance Targets
- Document indexing: 100 docs/minute
- Search latency: <100ms (hybrid)
- Reranking: <50ms for 20 docs
- Embedding generation: 50 docs/minute

## Installation Script

```bash
#!/bin/bash
# M1 Mac optimized installation

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Phase 1 dependencies
pip install --upgrade pip
pip install tantivy-py xxhash faiss-cpu

# Install Phase 2 dependencies  
pip install tiktoken sentence-transformers
pip install onnxruntime prometheus-client
pip install aiofiles

# Verify ARM64 installation
python -c "import platform; print(f'Architecture: {platform.machine()}')"
```

## Development Workflow

1. **Use Rosetta sparingly**: Most packages now have ARM64 support
2. **Leverage Metal**: For embedding models when possible
3. **Monitor thermals**: M1 Max can throttle under sustained load
4. **Use Activity Monitor**: Track efficiency vs performance core usage

## Testing Performance

```python
# benchmark.py
import time
import numpy as np

def benchmark_search():
    """Test hybrid search performance"""
    start = time.time()
    # Your search implementation
    results = hybrid_search.search("test query", k=10)
    latency = (time.time() - start) * 1000
    print(f"Search latency: {latency:.2f}ms")
    assert latency < 100, "Search too slow"

def benchmark_indexing():
    """Test indexing throughput"""
    docs = ["sample doc"] * 100
    start = time.time()
    indexer.index_documents(docs)
    throughput = 100 / (time.time() - start) * 60
    print(f"Indexing: {throughput:.0f} docs/min")
    assert throughput > 100, "Indexing too slow"
```

## Missing Critical Elements & Solutions

### 1. **Integration with Existing Ollama Models**
```python
# How to integrate with your current setup
class IntegratedRAGSystem:
    def __init__(self):
        # Existing Ollama models
        self.embed_model = "embeddinggemma:300m"
        self.gen_model = "gpt-oss:20b"
        self.vision_model = "llava:7b"
        
        # New components
        self.tantivy_index = TantivyIndex()
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
```

### 2. **Streamlit UI Updates**
```python
# Update sidebar for hybrid search toggle
with st.sidebar:
    search_mode = st.radio(
        "Search Mode",
        ["Vector Only", "Keyword Only", "Hybrid (Recommended)"],
        index=2
    )
    
    if search_mode == "Hybrid":
        alpha = st.slider("Vector vs Keyword Weight", 0.0, 1.0, 0.5)
```

### 3. **Error Handling & Fallbacks**
```python
class RobustHybridSearch:
    def search(self, query: str, k: int = 10):
        try:
            # Try hybrid search
            return self._hybrid_search(query, k)
        except TantivyError:
            # Fallback to vector-only
            logger.warning("BM25 index unavailable, using vector search")
            return self._vector_search(query, k)
```

### 4. **Data Migration Strategy**
```python
# Migrate existing indexed documents
class DataMigrator:
    def migrate_to_enhanced_schema(self):
        """Add missing metadata to existing documents"""
        existing_docs = self.db_manager.get_all_documents()
        
        for doc in existing_docs:
            # Add missing fields
            doc['content_hash'] = xxhash.xxh64(doc['chunk']).hexdigest()
            doc['doc_type'] = self.analyzer.detect_document_type(doc['chunk'])
            doc['language'] = self.analyzer.detect_language(doc['chunk'])
            
        self.db_manager.update_documents(existing_docs)
```

### 5. **Caching Strategy**
```python
# pip install diskcache
from diskcache import Cache

class CachedEmbeddings:
    def __init__(self, cache_dir=".cache/embeddings"):
        self.cache = Cache(cache_dir, size_limit=2**30)  # 1GB cache
        
    def get_embedding(self, text: str):
        cache_key = xxhash.xxh64(text).hexdigest()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        embedding = ollama.embeddings(model="embeddinggemma:300m", prompt=text)
        self.cache[cache_key] = embedding
        return embedding
```

### 6. **Sentence Boundary Detection**
```python
# pip install spacy
# python -m spacy download en_core_web_sm

import spacy

class SmartChunker:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
    def chunk_with_boundaries(self, text: str, max_tokens: int = 512):
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.encoder.encode(sentence))
            if current_tokens + sentence_tokens > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
                
        return chunks
```

### 7. **Query Expansion**
```python
class QueryExpander:
    def expand_query(self, query: str) -> List[str]:
        """Generate query variations for better recall"""
        expanded = [query]
        
        # Add synonyms using WordNet or simple rules
        # Add stemmed version
        # Add common typo corrections
        
        return expanded
```

### 8. **Testing Suite**
```python
# test_hybrid_search.py
import pytest

def test_tantivy_installation():
    """Verify Tantivy works on M1"""
    from tantivy import Index
    assert Index is not None

def test_hybrid_search_performance():
    """Ensure hybrid search meets latency targets"""
    search = HybridSearch()
    start = time.time()
    results = search.search("test query")
    assert (time.time() - start) < 0.1  # <100ms

def test_fallback_mechanism():
    """Test graceful degradation"""
    # Simulate Tantivy failure
    # Verify vector search still works
```

### 9. **Configuration Management**
```python
# config.yaml
hybrid_search:
  enabled: true
  vector_weight: 0.5
  keyword_weight: 0.5
  reranking_enabled: true
  reranker_model: "ms-marco-MiniLM-L-6-v2"
  
performance:
  batch_size: 10
  max_workers: 8
  cache_size_mb: 1024
  
ollama:
  embed_model: "embeddinggemma:300m"
  gen_model: "gpt-oss:20b"
  vision_model: "llava:7b"
```

### 10. **Deployment Checklist**
- [ ] Verify all dependencies install on ARM64
- [ ] Test with existing Ollama models
- [ ] Migrate existing indexed documents
- [ ] Update Streamlit UI components
- [ ] Configure caching
- [ ] Set up monitoring
- [ ] Run performance benchmarks
- [ ] Test fallback mechanisms
- [ ] Document API changes

## Conclusion

This technology stack is optimized for Mac M1 Max local development:
- **Native ARM64** packages wherever possible
- **Minimal memory footprint** (<15GB total)
- **No heavy dependencies** (no Elasticsearch, no Docker required)
- **Leverages M1 advantages**: Unified memory, Neural Engine, efficiency cores
- **Fully integrated** with existing Ollama models
- **Robust fallbacks** for production reliability

The selected technologies provide production-grade performance while maintaining simplicity for local development and demonstration purposes.