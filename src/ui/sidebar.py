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