import streamlit as st

from ..config import get_settings
from ..core import DatabaseManager, DocumentIndexer


def render_sidebar(db_manager: DatabaseManager, indexer: DocumentIndexer):
    """Render the sidebar with document management."""
    settings = get_settings()
    
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Display documents folder path
        st.info(f"📂 Documents folder: `./{settings.documents_folder}/`")
        
        # Scan for documents
        available_files = indexer.scan_documents_folder()
        
        if available_files:
            st.success(f"Found {len(available_files)} document(s)")
            
            # Auto-detect toggle
            auto_detect = st.checkbox(
                "🤖 AI-Powered Auto-Detection",
                value=True,
                help="Uses LLM to intelligently detect document types and languages. Works with any language!"
            )
            
            if auto_detect:
                st.caption("✨ AI will analyze each document to determine optimal processing settings")
                st.caption("🌐 Supports: English, 简体中文, 繁體中文, and mixed languages")
            
            # Index button
            if st.button("🔄 Index New Documents", type="primary", use_container_width=True):
                # Use session state values if available
                chunk_size = st.session_state.get('chunk_size', 1200)
                overlap_size = st.session_state.get('overlap_size', 200)
                indexer.index_documents(
                    available_files, 
                    chunk_chars=chunk_size, 
                    overlap=overlap_size, 
                    auto_detect=auto_detect
                )
                st.rerun()
        else:
            st.warning(f"No documents found in `./{settings.documents_folder}/`")
            st.markdown("**Supported formats:**")
            st.markdown("PDF, Word, Markdown, HTML, CSV, Excel, JSON, TXT, etc.")
            st.markdown("**🎨 Vision Support:**")
            st.markdown("✅ PDF images, charts, and diagrams via LLaVA")
        
        st.divider()
        
        # Show indexed documents
        st.header("📊 Indexed Documents")
        indexed_docs = db_manager.get_indexed_documents()
        
        if not indexed_docs.empty:
            st.dataframe(indexed_docs, use_container_width=True, hide_index=True)
            
            total_chunks = indexed_docs["Chunks"].sum()
            st.metric("Total Chunks", total_chunks)
            
            # Clear index button
            if st.button("🗑️ Clear All Index", use_container_width=True):
                if db_manager.clear_index():
                    st.success("Index cleared!")
                    st.rerun()
        else:
            st.info("No documents indexed yet")
        
        st.divider()
        
        # Hybrid Search Controls
        st.header("🔍 Search Settings")
        
        # Search mode selector
        search_mode = st.radio(
            "Search Mode",
            ["hybrid", "vector", "keyword"],
            index=["hybrid", "vector", "keyword"].index(settings.default_search_mode),
            help="Choose search strategy: Hybrid combines keyword and vector search"
        )
        st.session_state['search_mode'] = search_mode
        
        # Hybrid search weight slider (only show for hybrid mode)
        if search_mode == "hybrid":
            alpha = st.slider(
                "Vector vs Keyword Weight",
                min_value=0.0,
                max_value=1.0,
                value=settings.hybrid_alpha,
                step=0.1,
                help="0 = Pure keyword search, 1 = Pure vector search, 0.5 = Balanced"
            )
            st.session_state['hybrid_alpha'] = alpha
            
            # Show weight distribution
            keyword_weight = int((1 - alpha) * 100)
            vector_weight = int(alpha * 100)
            st.caption(f"📝 Keyword: {keyword_weight}% | 🎯 Vector: {vector_weight}%")
        
        # Results limit
        top_k = st.number_input(
            "Number of Results",
            min_value=1,
            max_value=20,
            value=settings.search_result_limit,
            help="Number of document chunks to retrieve"
        )
        st.session_state['top_k'] = top_k