# Claude Development Guidelines

## Integration Testing

### After Bug Fixes - Indexing Feature Verification

**IMPORTANT**: After any bug fix or code modification related to document processing, indexing, or search functionality, you MUST run the following integration test to verify the indexing feature works correctly:

#### Integration Test Command:
```bash
source .venv/bin/activate && python test_indexing.py
```

#### What This Test Verifies:
1. **Document Scanning**: Ensures all document types (.docx, .pdf, .md, .txt) are properly detected
2. **Recursive Directory Scanning**: Verifies subdirectories are scanned correctly
3. **Document Name Preservation**: Confirms original document names and paths are preserved
4. **Index Clearing**: Tests the ability to clear and rebuild the index

#### If Test Fails:
1. Read the error output carefully
2. Identify which specific test case failed
3. Fix the issue in the relevant source file:
   - Document scanning issues → `src/core/indexer.py`
   - Database issues → `src/core/database.py`
   - Configuration issues → `src/config/settings.py`
4. Re-run the test until it passes
5. Document the fix in your commit message

#### Additional Testing for Specific Components:

If you modify **hybrid search** functionality:
```bash
python test_hybrid_search.py
```

If you modify **UI components**:
```bash
streamlit run src/app.py --server.headless true &
sleep 5
curl http://localhost:8501 || echo "UI test failed"
pkill -f streamlit
```

#### Pre-Commit Checklist:
- [ ] Run `python test_indexing.py` - must pass
- [ ] Run type checking if available: `mypy src/` (if configured)
- [ ] Run linting if available: `flake8 src/` or `ruff check src/`
- [ ] Verify no sensitive information in code (API keys, passwords)
- [ ] Update this CLAUDE.md if new test procedures are needed

## Development Commands Reference

### Virtual Environment:
```bash
source .venv/bin/activate  # Activate virtual environment
```

### Running the Application:
```bash
streamlit run src/app.py
```

### Running Ollama (for embeddings):
```bash
ollama serve  # Start Ollama server
ollama pull nomic-embed-text  # Pull embedding model if not present
```

## Project Structure:
- `src/core/`: Core functionality (indexing, search, database)
- `src/ui/`: Streamlit UI components
- `src/config/`: Configuration and settings
- `documents/`: Default document storage location
- `data/`: Database and vector storage

## Important Notes:
- Always run integration tests after modifying core functionality
- The indexing test creates test documents in the documents folder
- Tests should complete within 30 seconds
- If Ollama is not running, some tests may skip embedding generation but should still verify file scanning