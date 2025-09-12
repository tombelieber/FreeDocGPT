import streamlit as st

from ..config import get_settings
from ..core import DatabaseManager, DocumentIndexer
from .i18n import t, get_locale, set_locale
from .locale_persistence import save_locale


def render_sidebar(db_manager: DatabaseManager, indexer: DocumentIndexer, search_service=None):
    """Render the sidebar with tabbed navigation."""
    settings = get_settings()
    
    with st.sidebar:
        # Tabbed navigation for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Documents", "🔍 Search", "🤖 Models", "⚙️ Settings"])
        
        with tab1:  # Documents
            _render_documents_tab(db_manager, indexer, settings)
            
        with tab2:  # Search
            _render_search_tab(settings, search_service)
            
        with tab3:  # Models
            _render_models_tab(settings, search_service)
            
        with tab4:  # Settings
            _render_settings_tab(settings)


def _render_documents_tab(db_manager: DatabaseManager, indexer: DocumentIndexer, settings):
    """Documents tab - file management and indexing."""
    st.markdown("### 📄 Documents")
    
    # Folder info
    st.info(f"📂 Folder: `./{settings.documents_folder}/`")
    
    # Scan for documents
    available_files = indexer.scan_documents_folder()
    
    if available_files:
        st.success(f"Found {len(available_files)} documents")
        
        # AI Auto-detect
        auto_detect = st.checkbox(
            "🤖 AI Auto-detect", 
            value=True,
            help="Auto-detect document types and languages"
        )
        
        # Index button
        if st.button("🔄 Index Documents", type="primary", use_container_width=True):
            chunk_size = st.session_state.get('chunk_size', 1200)
            overlap_size = st.session_state.get('overlap_size', 200)
            indexer.index_documents(
                available_files, 
                chunk_chars=chunk_size, 
                overlap=overlap_size, 
                auto_detect=auto_detect
            )
            st.rerun()
        
        # Document list
        with st.expander("View documents", expanded=False):
            for file in available_files[:15]:
                st.caption(f"📄 {file.name}")
            if len(available_files) > 15:
                st.caption(f"... and {len(available_files) - 15} more")
    else:
        st.warning("No documents found")
        st.markdown("**Supported:** PDF, Word, Markdown, TXT, HTML, CSV, Excel, JSON")
    
    # Index status
    st.divider()
    indexed_docs = db_manager.get_indexed_documents()
    if not indexed_docs.empty:
        total_chunks = indexed_docs["Chunks"].sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Indexed", len(indexed_docs))
        with col2:
            st.metric("Chunks", total_chunks)
            
        if st.button("🧹 Reset Index", use_container_width=True):
            _clear_index(db_manager, indexer)
            st.rerun()
            
        # Detailed view
        with st.expander("View indexed", expanded=True):
            st.dataframe(indexed_docs, use_container_width=True, hide_index=True)
    else:
        st.info("No documents indexed")
        if st.button("🧹 Reset Index", use_container_width=True):
            _clear_index(db_manager, indexer)
            st.rerun()


def _render_search_tab(settings, search_service):
    """Search tab - search configuration and controls."""
    st.markdown("### 🔍 Search")
    
    # Search mode
    search_mode = st.radio(
        "Search Mode",
        ["hybrid", "vector", "keyword"],
        index=["hybrid", "vector", "keyword"].index(settings.default_search_mode),
        help="Search strategy"
    )
    st.session_state['search_mode'] = search_mode
    
    # Hybrid settings
    if search_mode == "hybrid":
        alpha = st.slider(
            "Vector/Keyword Balance",
            0.0, 1.0,
            settings.hybrid_alpha,
            step=0.1,
            help="0=Keyword only, 1=Vector only"
        )
        st.session_state['hybrid_alpha'] = alpha
        
        kw_pct = int((1-alpha)*100)
        vec_pct = int(alpha*100)
        st.caption(f"📝 Keyword: {kw_pct}% | 🎯 Vector: {vec_pct}%")
    
    # Results limit
    top_k = st.slider(
        "Results Count", 
        1, 15, 
        settings.search_result_limit, 
        help="Number of document chunks"
    )
    st.session_state['top_k'] = top_k
    
    # Processing settings
    st.divider()
    st.markdown("**Processing**")
    
    # Initialize session state for processing settings
    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = 1200
    if 'overlap_size' not in st.session_state:
        st.session_state.overlap_size = 200
    
    chunk_size = st.slider(
        "Chunk Size", 
        500, 2000, 
        st.session_state.chunk_size, 
        step=100,
        help="Characters per chunk"
    )
    st.session_state.chunk_size = chunk_size
    
    overlap_size = st.slider(
        "Overlap Size", 
        0, 400, 
        st.session_state.overlap_size, 
        step=50,
        help="Shared characters between chunks"
    )
    st.session_state.overlap_size = overlap_size
    
    # AI features
    st.divider()
    thinking_mode = st.checkbox(
        "🤔 AI Thinking Mode", 
        value=st.session_state.get('thinking_mode', False),
        help="Show AI reasoning process"
    )
    st.session_state['thinking_mode'] = thinking_mode


def _render_models_tab(settings, search_service):
    """Models tab - AI model configuration."""
    st.markdown("### 🤖 Models")
    
    # Model status
    st.markdown("**Current Models**")
    st.caption(f"🔤 Embedding: {settings.embed_model}")
    st.caption(f"💬 Generation: {settings.gen_model}")
    
    # Ollama setup info
    with st.expander("Setup Guide", expanded=False):
        st.markdown("""
        **Install Ollama:**
        ```bash
        brew install ollama
        ollama serve
        ```
        
        **Pull Models:**
        ```bash
        ollama pull embeddinggemma:300m
        ollama pull gpt-oss:20b
        ```
        """)
    
    # Model configuration
    st.divider()
    embed_model = st.text_input(
        "Embedding Model", 
        value=settings.embed_model,
        help="Model for document embeddings"
    )
    
    gen_model = st.text_input(
        "Generation Model", 
        value=settings.gen_model,
        help="Model for chat responses"
    )
    
    if st.button("🔍 Check Ollama Status", use_container_width=True):
        from ..utils import display_ollama_status
        display_ollama_status()
    
    # System prompt
    st.divider()
    st.markdown("**System Prompt**")
    
    from pathlib import Path
    
    prompt_path = settings.system_prompt_path
    candidate = Path(prompt_path)
    if not candidate.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        candidate = repo_root / candidate
    
    if 'custom_system_prompt' not in st.session_state:
        try:
            if candidate.exists():
                st.session_state.custom_system_prompt = candidate.read_text(encoding="utf-8")
            else:
                st.session_state.custom_system_prompt = ""
        except Exception:
            st.session_state.custom_system_prompt = ""
    
    with st.expander("Edit System Prompt", expanded=False):
        edited_prompt = st.text_area(
            "Prompt Content",
            value=st.session_state.custom_system_prompt,
            height=200,
            help="Customize AI behavior"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save", type="primary", use_container_width=True):
                try:
                    candidate.write_text(edited_prompt, encoding="utf-8")
                    st.session_state.custom_system_prompt = edited_prompt
                    if search_service:
                        search_service.reload_system_prompt()
                    st.success("✅ Saved!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                try:
                    if candidate.exists():
                        st.session_state.custom_system_prompt = candidate.read_text(encoding="utf-8")
                        st.success("✅ Reset!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")


def _render_settings_tab(settings):
    """Settings tab - general application settings."""
    st.markdown("### ⚙️ Settings")
    
    # Language
    current = get_locale()
    lang_options = {
        "en": "🇺🇸 English", 
        "zh-Hant": "🇹🇼 繁體中文",
        "zh-Hans": "🇨🇳 简体中文", 
        "es": "🇪🇸 Español",
        "ja": "🇯🇵 日本語"
    }
    
    choice = st.selectbox(
        "Language", 
        options=list(lang_options.keys()),
        index=list(lang_options.keys()).index(current) if current in lang_options else 0,
        format_func=lambda x: lang_options.get(x, x)
    )
    
    if choice != current:
        set_locale(choice)
        try:
            st.query_params["locale"] = choice
            save_locale(choice)
        except Exception:
            pass
        st.rerun()
    
    # Interface settings
    st.divider()
    st.markdown("**Interface**")
    
    completion_sound = st.checkbox(
        "🔊 Completion Sound",
        value=st.session_state.get('enable_completion_sound', settings.enable_completion_sound),
        help="Play sound when response completes"
    )
    st.session_state['enable_completion_sound'] = completion_sound
    
    # Quick presets
    st.divider()
    st.markdown("**Quick Presets**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Meeting Notes", use_container_width=True):
            st.session_state.chunk_size = 800
            st.session_state.overlap_size = 100
            st.session_state.top_k = 3
            st.success("Applied Meeting Notes preset")
            st.rerun()
            
        if st.button("💻 Tech Docs", use_container_width=True):
            st.session_state.chunk_size = 1800
            st.session_state.overlap_size = 400
            st.session_state.top_k = 5
            st.success("Applied Tech Docs preset")
            st.rerun()
    
    with col2:
        if st.button("📋 PRD/Specs", use_container_width=True):
            st.session_state.chunk_size = 1500
            st.session_state.overlap_size = 300
            st.session_state.top_k = 7
            st.success("Applied PRD/Specs preset")
            st.rerun()
            
        if st.button("📚 Wiki/KB", use_container_width=True):
            st.session_state.chunk_size = 1200
            st.session_state.overlap_size = 200
            st.session_state.top_k = 5
            st.success("Applied Wiki/KB preset")
            st.rerun()
    
    # Current config
    st.divider()
    st.markdown("**Current Config**")
    config_cols = st.columns(3)
    
    with config_cols[0]:
        st.metric("Chunk", f"{st.session_state.get('chunk_size', 1200)}")
    with config_cols[1]:
        st.metric("Overlap", f"{st.session_state.get('overlap_size', 200)}")
    with config_cols[2]:
        st.metric("Results", f"{st.session_state.get('top_k', 5)}")


def _clear_index(db_manager: DatabaseManager, indexer: DocumentIndexer):
    """Helper function to clear the index."""
    ok = db_manager.clear_index()
    try:
        if indexer.hybrid_search:
            indexer.hybrid_search.clear_index()
    except Exception as e:
        st.warning(f"Tantivy clear warning: {e}")
    
    if ok:
        st.success("✅ Index reset completed")
    else:
        st.error("❌ Failed to reset index")