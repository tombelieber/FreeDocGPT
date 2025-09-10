import streamlit as st

from ..config import get_settings
from ..core import DatabaseManager, DocumentIndexer
from .i18n import t, get_locale, set_locale
from .locale_persistence import save_locale


def render_sidebar(db_manager: DatabaseManager, indexer: DocumentIndexer):
    """Render the sidebar with document management."""
    settings = get_settings()
    
    with st.sidebar:
        st.header(t("sidebar.document_management", "📁 Document Management"))
        
        # Display documents folder path
        st.info(t("sidebar.documents_folder", "📂 Documents folder: `./{folder}/`", folder=settings.documents_folder))
        
        # Scan for documents
        available_files = indexer.scan_documents_folder()
        
        if available_files:
            st.success(t("sidebar.found_docs", "Found {count} document(s)", count=len(available_files)))
            
            # Auto-detect toggle
            auto_detect = st.checkbox(
                t("sidebar.auto_detect", "🤖 AI-Powered Auto-Detection"),
                value=True,
                help=t("sidebar.auto_detect_help", "Uses LLM to intelligently detect document types and languages.")
            )
            
            if auto_detect:
                st.caption("✨ AI will analyze each document to determine optimal processing settings")
                st.caption("🌐 Supports: English, 简体中文, 繁體中文, and mixed languages")
            
            # Index button
            if st.button(t("sidebar.index_new_docs", "🔄 Index New Documents"), type="primary", use_container_width=True):
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
            st.warning(t("sidebar.no_docs_found", "No documents found in `./{folder}/`", folder=settings.documents_folder))
            st.markdown(f"**{t('sidebar.supported_formats_title', 'Supported formats:')}**")
            st.markdown(t("sidebar.supported_formats_list", "PDF, Word, Markdown, HTML, CSV, Excel, JSON, TXT, etc."))
            st.markdown(f"**{t('sidebar.vision_support_title', '🎨 Vision Support:')}**")
            st.markdown(t("sidebar.vision_support_detail", "✅ PDF images, charts, and diagrams via LLaVA"))
        
        st.divider()
        
        # Show indexed documents
        st.header(t("sidebar.indexed_docs", "📊 Indexed Documents"))
        indexed_docs = db_manager.get_indexed_documents()
        
        if not indexed_docs.empty:
            st.dataframe(indexed_docs, use_container_width=True, hide_index=True)
            
            total_chunks = indexed_docs["Chunks"].sum()
            st.metric(t("sidebar.total_chunks", "Total Chunks"), total_chunks)
            
            # Reset index button (always visible path below too)
            if st.button(t("sidebar.reset_index", "🧹 Reset Index"), use_container_width=True, help=t("sidebar.reset_index_help", "Drop LanceDB table and clear Tantivy index")):
                ok = db_manager.clear_index()
                try:
                    if indexer.hybrid_search:
                        indexer.hybrid_search.clear_index()
                except Exception as e:
                    st.warning(t("sidebar.tantivy_warn", "Tantivy clear warning: {err}", err=e))
                if ok:
                    st.success(t("sidebar.reset_ok", "Index reset completed."))
                else:
                    st.error(t("sidebar.reset_fail", "Failed to reset index. You can manually delete the .lancedb folder and reload."))
                st.rerun()
        else:
            st.info(t("sidebar.no_index_yet", "No documents indexed yet"))
            # Offer reset even when stats fail (corruption may hide the table)
            if st.button(t("sidebar.reset_index", "🧹 Reset Index"), use_container_width=True, help=t("sidebar.reset_index_help", "Drop LanceDB table and clear Tantivy index")):
                ok = db_manager.clear_index()
                try:
                    if indexer.hybrid_search:
                        indexer.hybrid_search.clear_index()
                except Exception as e:
                    st.warning(t("sidebar.tantivy_warn", "Tantivy clear warning: {err}", err=e))
                if ok:
                    st.success(t("sidebar.reset_ok", "Index reset completed."))
                else:
                    st.error(t("sidebar.reset_fail", "Failed to reset index. You can manually delete the .lancedb folder and reload."))
                st.rerun()
        
        st.divider()
        
        # Hybrid Search Controls
        st.header(t("search.header", "🔍 Search Settings"))
        
        # Search mode selector
        search_mode = st.radio(
            t("search.mode", "Search Mode"),
            ["hybrid", "vector", "keyword"],
            index=["hybrid", "vector", "keyword"].index(settings.default_search_mode),
            help=t("search.mode_help", "Choose search strategy: Hybrid combines keyword and vector search")
        )
        st.session_state['search_mode'] = search_mode
        
        # Hybrid search weight slider (only show for hybrid mode)
        if search_mode == "hybrid":
            alpha = st.slider(
                t("search.alpha_label", "Vector vs Keyword Weight"),
                min_value=0.0,
                max_value=1.0,
                value=settings.hybrid_alpha,
                step=0.1,
                help=t("search.alpha_help", "0 = Pure keyword search, 1 = Pure vector search, 0.5 = Balanced")
            )
            st.session_state['hybrid_alpha'] = alpha
            
            # Show weight distribution
            keyword_weight = int((1 - alpha) * 100)
            vector_weight = int(alpha * 100)
            st.caption(t("search.weight_caption", "📝 Keyword: {kw}% | 🎯 Vector: {vec}%", kw=keyword_weight, vec=vector_weight))
        
        # Results limit
        top_k = st.number_input(
            t("search.results_label", "Number of Results"),
            min_value=1,
            max_value=20,
            value=settings.search_result_limit,
            help=t("search.results_help", "Number of document chunks to retrieve")
        )
        st.session_state['top_k'] = top_k
        
        # Thinking Mode Toggle
        st.divider()
        thinking_mode = st.checkbox(
            "🤔 **AI Thinking Mode**",
            value=st.session_state.get('thinking_mode', False),
            help="Show AI's reasoning process before the final answer (like ChatGPT thinking)"
        )
        st.session_state['thinking_mode'] = thinking_mode
        
        if thinking_mode:
            st.caption("✨ AI will show its thought process and reasoning")
        else:
            st.caption("💨 Faster responses without thinking display")

        st.divider()
        # Language selector
        current = get_locale()
        label = t("lang.selector", "🌐 Language / 語言 / 语言")
        display = {
            "en": t("lang.en", "English"),
            "zh-Hant": t("lang.zh_hant", "繁體中文"),
            "zh-Hans": t("lang.zh_hans", "简体中文"),
            "es": t("lang.es", "Español"),
            "ja": t("lang.ja", "日本語"),
        }
        options = ["en", "zh-Hant", "zh-Hans", "es", "ja"]
        idx = options.index(current) if current in options else 0
        choice = st.selectbox(label, options=options, index=idx, format_func=lambda x: display.get(x, x))
        if choice != current:
            set_locale(choice)
            try:
                # Use modern query params API
                st.query_params["locale"] = choice
            except Exception:
                pass
            try:
                save_locale(choice)
            except Exception:
                pass
            st.rerun()
        
        # Model Configuration Info (shown once)
        st.divider()
        st.header("🤖 AI Models")
        st.caption(f"**Embedding:** {settings.embed_model}")
        st.caption(f"**Generation:** {settings.gen_model}")