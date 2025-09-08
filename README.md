# 🚀 Enterprise Document Intelligence System
**AI-Powered Document Search & Q&A for Corporate Knowledge Management**

Transform your company's scattered documentation into an intelligent, searchable knowledge base. Ask questions in natural language and get instant, accurate answers from your documents.

> 🎯 **Demo Application** | Ready for enterprise deployment as a 常駐 service

## 🎯 What This Does

This system creates an intelligent layer over your company's documentation (Notion, Lark, PDFs, etc.) enabling:
- **Instant Knowledge Discovery**: Find information across thousands of documents in seconds
- **Natural Language Q&A**: Ask questions like "What were the Q3 action items?" or "How do I configure the API?"
- **Smart Document Processing**: Automatically adapts to different document types (meeting notes, PRDs, technical docs)
- **Source Attribution**: Always know where answers come from with direct citations

## 💡 Business Value & Potential

When deployed as a company-wide 常駐 service, this tool can:
- **Boost Productivity**: Reduce time spent searching for documentation by 80%
- **Accelerate Onboarding**: New employees can instantly access institutional knowledge
- **Enable Feature Development**: Developers quickly locate specs, requirements, and technical docs
- **Improve Decision Making**: Surface relevant meeting notes, decisions, and action items instantly
- **Break Down Silos**: Make knowledge accessible across teams and departments

## ✨ Key Features (v2.0)

### 🤖 Intelligent Document Processing
- **Auto-Detection**: Automatically identifies document types (meeting notes, PRDs, technical docs, wikis)
- **Adaptive Chunking**: Optimizes processing based on document type for better search results
- **Smart Context Preservation**: Maintains code blocks, tables, and formatting integrity

### 📊 Enterprise-Ready Features
- **Multi-Format Support**: 15+ file formats including PDF, Word, Markdown, HTML, CSV, Excel, JSON
- **Notion/Lark Integration**: Optimized for enterprise documentation exports
- **Configurable Processing**: Fine-tune chunk sizes, overlap, and search parameters
- **Document Type Presets**: One-click optimization for different content types

### 🔍 Advanced Search & Analytics
- **Vector Similarity Search**: Find semantically related content, not just keywords
- **Real-time Performance Metrics**: Token generation speed, response times, and throughput
- **Source Tracking**: Complete audit trail of information sources
- **Document Statistics**: Track indexed content, chunk distribution, and coverage

## 🚀 Quick Start Testing Guide

### Step 1: Install Prerequisites (macOS)

```bash
# Install Ollama (local LLM runtime)
brew install ollama

# Start Ollama service
ollama serve &

# Pull required models (this may take 10-15 minutes)
ollama pull gpt-oss:20b        # Main language model (20B parameters)
ollama pull embeddinggemma:300m # Embedding model for search
```

### Step 2: Setup Python Environment

```bash
# Clone the repository
git clone <repository-url>
cd agentic_rag_gptoss_mac

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Prepare Test Documents

```bash
# Create documents folder
mkdir -p documents

# Add sample documents (any of these formats):
# - Export from Notion/Lark as Markdown
# - Add PDFs, Word docs, or text files
# - Include meeting notes, PRDs, or technical docs
```

### Step 4: Run the Application

```bash
# Start the Streamlit app
streamlit run app.py

# Open browser to http://localhost:8501
```

### Step 5: Test the System

1. **Index Documents**
   - Click "🔄 Index New Documents" in sidebar
   - Watch as system auto-detects document types
   - View indexing statistics

2. **Ask Questions**
   - Try: "What are the main features?"
   - Try: "What decisions were made in the last meeting?"
   - Try: "How do I implement the authentication module?"

3. **Optimize for Your Content**
   - Use presets: Meeting Notes, PRD/Specs, Tech Docs, Wiki
   - Fine-tune chunk sizes and overlap
   - Adjust search depth (Top-K results)

## 📁 Supported Document Formats

| Category | Formats | Best For |
|----------|---------|----------|
| **Documents** | PDF, DOCX, DOC | Reports, specifications, manuals |
| **Knowledge Base** | MD, Markdown | Notion/Lark exports, wikis |
| **Web Content** | HTML, HTM | Web documentation, archived pages |
| **Structured Data** | CSV, XLSX, XLS | Data tables, inventories |
| **Configuration** | JSON, XML, YAML | Config files, API specs |
| **Plain Text** | TXT, LOG, RTF | Meeting notes, logs |

## ⚙️ Configuration & Optimization

### Environment Variables (.env)
```env
DB_DIR=.lancedb                    # Vector database location
TABLE_NAME=docs                    # Database table name
EMBED_MODEL=embeddinggemma:300m    # Embedding model for search
GEN_MODEL=gpt-oss:20b              # Generation model for answers
DOCUMENTS_FOLDER=documents         # Document storage folder
```

### Document Processing Presets

| Document Type | Chunk Size | Overlap | Use Case |
|--------------|------------|---------|----------|
| **Meeting Notes** | 800 | 100 | Capture individual decisions & action items |
| **PRD/Specs** | 1500 | 300 | Keep requirements & features together |
| **Technical Docs** | 1800 | 400 | Preserve code blocks & examples |
| **Wiki/KB** | 1200 | 200 | Balance detail & searchability |

## 🎯 Testing Checklist

- [ ] **Document Ingestion**: Add 5-10 diverse documents to `documents/` folder
- [ ] **Auto-Detection**: Enable auto-detect and verify document type identification
- [ ] **Search Accuracy**: Ask 10 different questions and verify answer relevance
- [ ] **Performance**: Monitor token generation speed (should be >20 tokens/sec)
- [ ] **Source Attribution**: Verify citations point to correct documents
- [ ] **Different Formats**: Test with PDFs, Markdown, Word docs, and CSVs
- [ ] **Preset Optimization**: Try different presets for various document types
- [ ] **Clear & Re-index**: Test index management capabilities

## 📈 Performance Metrics

The system provides real-time metrics:
- **Response Time**: Total time to generate answer
- **First Token Latency**: Time to start streaming response
- **Token Generation**: Speed in tokens per second
- **Search Performance**: Document retrieval time

## 🚀 Future Extensibility

If approved for production deployment:
- **Multi-tenant Support**: Separate knowledge bases per team/department
- **Access Control**: Role-based document access
- **API Integration**: RESTful API for programmatic access
- **Slack/Teams Integration**: Query documents directly from chat
- **Continuous Learning**: Feedback loop for answer improvement
- **Multi-language Support**: Process documents in multiple languages
- **Cloud Deployment**: Scale to handle enterprise-wide usage

## 🔒 Privacy & Security

- **100% Local Processing**: All data stays on your infrastructure
- **No External APIs**: Uses local Ollama models, no data leaves your network
- **Audit Trail**: Complete tracking of document access and queries
- **Configurable Retention**: Control how long indexed data is retained

## 💬 Support & Feedback

This is a demonstration application showcasing the potential of AI-powered document intelligence for enterprise use. For questions, feature requests, or production deployment discussions, please contact the development team.

---

**Note**: This demo requires ~15GB disk space for models and runs optimally with 16GB+ RAM.