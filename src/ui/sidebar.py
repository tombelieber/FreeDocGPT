import streamlit as st
import subprocess
import platform
from pathlib import Path

from ..config import get_settings
from ..core import DatabaseManager, DocumentIndexer
from .i18n import t, get_locale, set_locale
from .locale_persistence import save_locale


def render_sidebar(db_manager: DatabaseManager, indexer: DocumentIndexer, search_service=None):
    """Render the sidebar with tabbed navigation."""
    settings = get_settings()
    
    with st.sidebar:
        # Tabbed navigation for better organization
        tab1, tab2, tab3, tab4 = st.tabs([t("ui.documents", "📄 Documents"), t("ui.search", "🔍 Search"), t("ui.models", "🤖 Models"), t("ui.settings", "⚙️ Settings")])
        
        with tab1:  # Documents
            _render_documents_tab(db_manager, indexer, settings)
            
        with tab2:  # Search
            _render_search_tab(settings, search_service)
            
        with tab3:  # Models
            _render_models_tab(settings, search_service)
            
        with tab4:  # Settings
            _render_settings_tab(settings, search_service)


def _render_documents_tab(db_manager: DatabaseManager, indexer: DocumentIndexer, settings):
    """Documents tab - file management and indexing."""
    st.markdown(f"### {t('ui.documents', '📄 Documents')}")
    
    # Upload section
    st.markdown(t("sidebar.upload_documents", "**Upload Documents**"))
    uploaded_files = st.file_uploader(
        t("sidebar.choose_files", "Choose files"),
        type=list(ext.lstrip('.') for ext in settings.supported_extensions),
        accept_multiple_files=True,
        help=t("sidebar.upload_help", "Upload documents to add them to your collection")
    )
    
    if uploaded_files:
        if st.button(t("sidebar.upload_button", "📤 Upload Files"), type="primary", use_container_width=True):
            _handle_file_uploads(uploaded_files, settings)
    
    st.divider()
    
    # Folder info
    st.info(t("sidebar.folder", "📂 Folder: `./{folder}/`", folder=settings.documents_folder))
    
    # Scan for documents
    available_files = indexer.scan_documents_folder()
    
    if available_files:
        st.success(t("sidebar.found_documents", "Found {count} documents", count=len(available_files)))
        
        # AI Auto-detect
        auto_detect = st.checkbox(
            t("sidebar.ai_auto_detect", "🤖 AI Auto-detect"), 
            value=True,
            help=t("sidebar.auto_detect_help", "Auto-detect document types and languages")
        )
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("sidebar.index_documents", "🔄 Index Documents"), type="primary", use_container_width=True):
                chunk_size = st.session_state.get('chunk_size', 1200)
                overlap_size = st.session_state.get('overlap_size', 200)
                indexer.index_documents(
                    available_files, 
                    chunk_chars=chunk_size, 
                    overlap=overlap_size, 
                    auto_detect=auto_detect
                )
                st.rerun()
        
        with col2:
            if st.button(t("sidebar.remove_all", "🗑️ Remove All"), use_container_width=True):
                _show_remove_all_dialog(available_files, settings)
        
        # Document list with multi-select functionality
        with st.expander(t("sidebar.view_documents", "View documents"), expanded=False):
            # Initialize session state for selected files
            if "selected_files" not in st.session_state:
                st.session_state.selected_files = set()
            
            # Batch selection controls
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(t("sidebar.select_all_visible", "☑️ All"), key="select_all_visible", use_container_width=True):
                    files_to_show = available_files[:15]
                    st.session_state.selected_files.update(str(f) for f in files_to_show)
                    st.rerun()
            
            with col2:
                if st.button(t("sidebar.select_none", "☐ None"), key="select_none_visible", use_container_width=True):
                    st.session_state.selected_files.clear()
                    st.rerun()
            
            with col3:
                selected_count = len(st.session_state.selected_files)
                if selected_count > 0:
                    if st.button(f"🗑️ ({selected_count})", key="remove_selected", help=t("sidebar.remove_selected", "Remove selected files"), use_container_width=True):
                        _show_batch_remove_dialog(st.session_state.selected_files, settings)
            
            st.divider()
            
            # File list with checkboxes
            files_to_show = available_files[:15]
            for i, file in enumerate(files_to_show):
                file_str = str(file)
                is_selected = file_str in st.session_state.selected_files
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    # Checkbox for file selection
                    if st.checkbox(f"📄 {file.name}", value=is_selected, key=f"select_file_{i}"):
                        st.session_state.selected_files.add(file_str)
                    elif file_str in st.session_state.selected_files:
                        st.session_state.selected_files.remove(file_str)
                
                with col2:
                    # Individual remove button
                    if st.button("🗑️", key=f"remove_single_{i}", help=t("sidebar.remove_file", "Remove file"), use_container_width=False):
                        _show_remove_file_dialog(file, settings)
            
            if len(available_files) > 15:
                st.caption(t("sidebar.and_more", "... and {count} more", count=len(available_files) - 15))
                
                # Show all files button if there are many
                if st.button(t("sidebar.manage_all_files", "📁 Manage All Files"), use_container_width=True):
                    _show_file_manager_dialog(available_files, settings)
    else:
        st.warning(t("sidebar.no_documents_found", "No documents found"))
        st.markdown(t("sidebar.supported_formats_basic", "**Supported:** PDF, Word, Markdown, TXT, HTML, CSV, Excel, JSON"))
    
    # Index status
    st.divider()
    indexed_docs = db_manager.get_indexed_documents()
    if not indexed_docs.empty:
        total_chunks = indexed_docs["Chunks"].sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(t("sidebar.indexed_metric", "Indexed"), len(indexed_docs))
        with col2:
            st.metric(t("sidebar.chunks_metric", "Chunks"), total_chunks)
            
        if st.button(t("sidebar.reset_index_button", "🧹 Reset Index"), use_container_width=True):
            _show_reset_confirmation_dialog(db_manager, indexer)
            
        # Detailed view
        with st.expander(t("sidebar.view_indexed", "View indexed"), expanded=True):
            st.dataframe(indexed_docs, use_container_width=True, hide_index=True)
    else:
        st.info(t("sidebar.no_documents_indexed", "No documents indexed"))
        if st.button(t("sidebar.reset_index_button", "🧹 Reset Index"), use_container_width=True):
            _show_reset_confirmation_dialog(db_manager, indexer)


def _render_search_tab(settings, search_service):
    """Search tab - search configuration and controls."""
    st.markdown(t("sidebar.search_title", "### 🔍 Search"))
    
    # Ensure top_k is initialized for consistency with Settings tab
    if 'top_k' not in st.session_state:
        st.session_state.top_k = settings.search_result_limit
    
    # Search mode
    search_mode = st.radio(
        t("sidebar.search_mode", "Search Mode"),
        ["hybrid", "vector", "keyword"],
        index=["hybrid", "vector", "keyword"].index(settings.default_search_mode),
        help=t("sidebar.search_strategy_help", "Search strategy")
    )
    st.session_state['search_mode'] = search_mode
    
    # Hybrid settings
    if search_mode == "hybrid":
        alpha = st.slider(
            t("sidebar.vector_keyword_balance", "Vector/Keyword Balance"),
            0.0, 1.0,
            settings.hybrid_alpha,
            step=0.1,
            help=t("sidebar.hybrid_balance_help", "0=Keyword only, 1=Vector only")
        )
        st.session_state['hybrid_alpha'] = alpha
        
        kw_pct = int((1-alpha)*100)
        vec_pct = int(alpha*100)
        st.caption(t("sidebar.keyword_percent", "📝 Keyword: {percent}%", percent=kw_pct) + " | " + t("sidebar.vector_percent", "🎯 Vector: {percent}%", percent=vec_pct))
    
    
    # AI features - thinking mode moved to chat footer for better accessibility


def _render_models_tab(settings, search_service):
    """Models tab - AI model configuration."""
    st.markdown(t("sidebar.models_title", "### 🤖 Models"))
    
    # Model status
    st.markdown(t("sidebar.current_models", "**Current Models**"))
    st.caption(f"🔤 Embedding: {settings.embed_model}")
    st.caption(f"💬 Generation: {settings.gen_model}")
    
    # Ollama setup info
    with st.expander(t("sidebar.setup_guide", "Setup Guide"), expanded=False):
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
        t("sidebar.embedding_model_input", "Embedding Model"), 
        value=settings.embed_model,
        help=t("sidebar.embedding_model_help", "Model for document embeddings")
    )
    
    gen_model = st.text_input(
        t("sidebar.generation_model_input", "Generation Model"), 
        value=settings.gen_model,
        help=t("sidebar.generation_model_help", "Model for chat responses")
    )
    
    if st.button(t("sidebar.check_ollama_status", "🔍 Check Ollama Status"), use_container_width=True):
        from ..utils import display_ollama_status
        display_ollama_status()
    
    # System prompt
    st.divider()
    st.markdown(t("sidebar.system_prompt", "**System Prompt**"))
    
    from pathlib import Path
    
    prompt_path = settings.system_prompt_path
    candidate = Path(prompt_path)
    if not candidate.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        candidate = repo_root / candidate
    
    if 'custom_system_prompt' not in st.session_state:
        try:
            # Load language-specific prompt based on current locale
            current_locale = get_locale()
            
            # Map locales to prompt file names
            prompt_files = {
                "en": "rag_prompt_en.md",
                "zh-Hant": "rag_prompt_zh_hant.md", 
                "zh-Hans": "rag_prompt_zh_hans.md",
                "es": "rag_prompt_es.md",
                "ja": "rag_prompt_ja.md"
            }
            
            # Get the appropriate prompt file for current language
            prompt_filename = prompt_files.get(current_locale, "rag_prompt_en.md")
            
            # Try language-specific prompt first
            repo_root = Path(__file__).resolve().parents[2]
            language_candidate = repo_root / prompt_filename
            
            if language_candidate.exists() and language_candidate.is_file():
                content = language_candidate.read_text(encoding="utf-8").strip()
                if content:
                    st.session_state.custom_system_prompt = content
                else:
                    # Try default file as fallback
                    if candidate.exists():
                        st.session_state.custom_system_prompt = candidate.read_text(encoding="utf-8")
                    else:
                        st.session_state.custom_system_prompt = ""
            else:
                # Fallback to configured path (backward compatibility)
                if candidate.exists():
                    st.session_state.custom_system_prompt = candidate.read_text(encoding="utf-8")
                else:
                    st.session_state.custom_system_prompt = ""
        except Exception:
            st.session_state.custom_system_prompt = ""
    
    with st.expander(t("sidebar.edit_system_prompt", "Edit System Prompt"), expanded=False):
        edited_prompt = st.text_area(
            t("sidebar.prompt_content", "Prompt Content"),
            value=st.session_state.custom_system_prompt,
            height=200,
            help=t("sidebar.prompt_help", "Customize AI behavior")
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("sidebar.save", "💾 Save"), type="primary", use_container_width=True):
                try:
                    candidate.write_text(edited_prompt, encoding="utf-8")
                    st.session_state.custom_system_prompt = edited_prompt
                    if search_service:
                        search_service.reload_system_prompt()
                    st.success(t("common.saved", "✅ Saved!"))
                except Exception as e:
                    st.error(t("sidebar.error", "❌ Error: {error}", error=str(e)))
        
        with col2:
            if st.button(t("common.reset", "🔄 Reset"), use_container_width=True):
                try:
                    # Load language-specific prompt based on current locale
                    current_locale = get_locale()
                    
                    # Map locales to prompt file names
                    prompt_files = {
                        "en": "rag_prompt_en.md",
                        "zh-Hant": "rag_prompt_zh_hant.md", 
                        "zh-Hans": "rag_prompt_zh_hans.md",
                        "es": "rag_prompt_es.md",
                        "ja": "rag_prompt_ja.md"
                    }
                    
                    # Get the appropriate prompt file for current language
                    prompt_filename = prompt_files.get(current_locale, "rag_prompt_en.md")
                    
                    # Try language-specific prompt first
                    repo_root = Path(__file__).resolve().parents[2]
                    language_candidate = repo_root / prompt_filename
                    
                    prompt_content = ""
                    
                    if language_candidate.exists() and language_candidate.is_file():
                        prompt_content = language_candidate.read_text(encoding="utf-8").strip()
                        if prompt_content:
                            st.session_state.custom_system_prompt = prompt_content
                            st.success(t("common.reset_success", f"✅ Reset to {current_locale} prompt!"))
                        else:
                            raise ValueError(f"Language-specific prompt file is empty: {language_candidate}")
                    else:
                        # Fallback to configured path (backward compatibility)
                        if candidate.exists():
                            prompt_content = candidate.read_text(encoding="utf-8").strip()
                            if prompt_content:
                                st.session_state.custom_system_prompt = prompt_content
                                st.success(t("common.reset_success", "✅ Reset to default prompt!"))
                            else:
                                raise ValueError(f"Default prompt file is empty: {candidate}")
                        else:
                            raise ValueError(f"No prompt file found for language {current_locale}")
                    
                    # Also reload search service if available
                    if search_service is not None:
                        search_service.reload_system_prompt()
                    
                    st.rerun()
                except Exception as e:
                    st.error(t("sidebar.error", "❌ Error: {error}", error=str(e)))


def _render_settings_tab(settings, search_service=None):
    """Settings tab - general application settings."""
    st.markdown(t("sidebar.settings_title", "### ⚙️ Settings"))
    
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
        t("sidebar.language", "Language"), 
        options=list(lang_options.keys()),
        index=list(lang_options.keys()).index(current) if current in lang_options else 0,
        format_func=lambda x: lang_options.get(x, x)
    )
    
    if choice != current:
        # Force locale change immediately
        set_locale(choice)
        # Force session state update by explicitly setting it again
        st.session_state["locale"] = choice
        
        try:
            st.query_params["locale"] = choice
            save_locale(choice)
            
            # Clear all cached prompt-related session state first
            if 'custom_system_prompt' in st.session_state:
                del st.session_state.custom_system_prompt
            if 'last_prompt_locale' in st.session_state:
                st.session_state.last_prompt_locale = choice
            
            # Force prompt cache invalidation and reload
            if search_service is not None:
                try:
                    # Reload system prompt (this now internally invalidates cache first)
                    search_service.reload_system_prompt()
                    st.success(t("sidebar.language_success", "✅ Language changed and system prompt reloaded!"))
                except Exception as e:
                    st.warning(t("sidebar.language_changed_warning", "⚠️ Language changed but failed to reload prompt: {error}", error=str(e)))
            else:
                st.warning(t("sidebar.search_service_unavailable", "⚠️ Search service not available, prompt will be reloaded on next query"))
                
        except Exception as e:
            st.error(f"❌ Error during language change: {e}")
        
        st.rerun()
    
    # Interface settings
    st.divider()
    st.markdown(t("sidebar.interface", "**Interface**"))
    
    completion_sound = st.checkbox(
        t("sidebar.completion_sound", "🔊 Completion Sound"),
        value=st.session_state.get('enable_completion_sound', settings.enable_completion_sound),
        help=t("sidebar.completion_sound_help", "Play sound when response completes")
    )
    st.session_state['enable_completion_sound'] = completion_sound
    
    # Document Processing Settings
    st.divider()
    st.markdown(t("sidebar.document_processing", "**Document Processing**"))
    
    # Initialize session state for processing settings
    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = 1200
    if 'overlap_size' not in st.session_state:
        st.session_state.overlap_size = 200
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 5
    
    # Chunk Size with comprehensive tooltip
    chunk_size = st.slider(
        t("sidebar.chunk_size_slider", "Chunk Size (characters)"), 
        500, 3000, 
        st.session_state.chunk_size, 
        step=100,
        help="""**Chunk Size** determines how much text is processed together as one unit.

**Smaller chunks (500-1000):**
• Better for Q&A with specific facts
• More precise but may lose context
• Good for: FAQs, definitions, short articles

**Medium chunks (1000-2000):** 
• Balanced approach for most documents
• Preserves moderate context while staying focused
• Good for: Technical docs, reports, general content

**Larger chunks (2000-3000):**
• Better for complex topics needing context
• May be less precise but preserves relationships
• Good for: Academic papers, detailed explanations, narratives"""
    )
    st.session_state.chunk_size = chunk_size
    
    # Overlap Size with comprehensive tooltip  
    overlap_size = st.slider(
        t("sidebar.overlap_size_slider", "Overlap Size (characters)"), 
        0, 500, 
        st.session_state.overlap_size, 
        step=25,
        help="""**Overlap Size** controls how much text is shared between adjacent chunks.

**No overlap (0):**
• Fastest processing, no duplication
• Risk of breaking related concepts across chunks
• Use when: Documents have clear section breaks

**Small overlap (50-150):**
• Minimal context preservation with good efficiency  
• Helps maintain continuity between chunks
• Use when: Well-structured documents

**Medium overlap (150-300):**
• Good balance of context and efficiency
• Recommended for most document types
• Helps AI understand relationships across boundaries

**Large overlap (300-500):**
• Maximum context preservation
• More processing time and storage
• Use when: Complex documents with interconnected concepts"""
    )
    st.session_state.overlap_size = overlap_size
    
    # Results Count with comprehensive tooltip
    top_k = st.slider(
        t("sidebar.results_count", "Results Count"), 
        1, 20, 
        st.session_state.top_k,
        help="""**Results Count** sets how many relevant document chunks are retrieved for each query.

**Few results (1-3):**
• Faster responses, lower token usage
• More focused but may miss relevant information  
• Good for: Simple questions, specific fact lookup

**Medium results (4-8):**
• Balanced approach for most queries
• Good coverage without overwhelming the AI
• Recommended for: General Q&A, document exploration

**Many results (9-20):**
• Comprehensive information gathering
• Slower responses, higher token usage
• Good for: Complex questions, research tasks, summarization

Note: More results = better coverage but higher costs and slower responses"""
    )
    st.session_state.top_k = top_k
    
    # Current Configuration Display
    st.divider()
    st.markdown(t("sidebar.current_config", "**Current Configuration**"))
    config_cols = st.columns(3)
    
    with config_cols[0]:
        st.metric(t("sidebar.chunk_size_metric", "Chunk Size"), f"{chunk_size:,} chars", help=t("sidebar.chunk_size_help_short", "Characters per document chunk"))
    with config_cols[1]:
        st.metric(t("sidebar.overlap_metric", "Overlap"), f"{overlap_size} chars", help=t("sidebar.overlap_help_short", "Shared characters between chunks"))  
    with config_cols[2]:
        st.metric(t("sidebar.results_metric", "Results"), f"{top_k}", help=t("sidebar.results_help_short", "Chunks retrieved per query"))


@st.dialog(t("dialog.reset_index_confirmation", "Reset Index Confirmation"))
def _show_reset_confirmation_dialog(db_manager: DatabaseManager, indexer: DocumentIndexer):
    """Show confirmation dialog for resetting the index."""
    st.warning(t("sidebar.warning_cannot_undo", "⚠️ **Warning**: This action cannot be undone!"))
    st.markdown(t("sidebar.will_delete", "This will permanently delete:"))
    st.markdown(t("sidebar.delete_indexed_docs", "- All indexed documents and their embeddings"))
    st.markdown(t("sidebar.delete_vector_db", "- Vector database entries"))
    st.markdown(t("sidebar.delete_search_index", "- Search index data"))
    st.markdown("")
    st.markdown(t("sidebar.need_reindex", "You will need to re-index your documents to restore search functionality."))
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(t("sidebar.cancel", "❌ Cancel"), use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button(t("sidebar.confirm_reset", "🗑️ Confirm Reset"), type="primary", use_container_width=True):
            _clear_index(db_manager, indexer)
            st.rerun()


def _clear_index(db_manager: DatabaseManager, indexer: DocumentIndexer):
    """Helper function to clear the index."""
    ok = db_manager.clear_index()
    try:
        if indexer.hybrid_search:
            indexer.hybrid_search.clear_index()
    except Exception as e:
        st.warning(f"Tantivy clear warning: {e}")
    
    if ok:
        st.success(t("sidebar.index_reset_completed", "✅ Index reset completed"))
    else:
        st.error(t("sidebar.index_reset_failed", "❌ Failed to reset index"))


def _handle_file_uploads(uploaded_files, settings):
    """Handle file uploads and save them to the documents folder."""
    documents_path = Path(settings.documents_folder)
    documents_path.mkdir(exist_ok=True)
    
    success_count = 0
    error_count = 0
    
    for uploaded_file in uploaded_files:
        try:
            # Get the file path
            file_path = documents_path / uploaded_file.name
            
            # Check if file already exists
            if file_path.exists():
                st.warning(t("sidebar.file_exists", "File '{filename}' already exists and will be overwritten", filename=uploaded_file.name))
            
            # Save the file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            success_count += 1
            st.success(t("sidebar.file_uploaded", "✅ Uploaded: {filename}", filename=uploaded_file.name))
            
        except Exception as e:
            error_count += 1
            st.error(t("sidebar.upload_error", "❌ Failed to upload {filename}: {error}", filename=uploaded_file.name, error=str(e)))
    
    # Show summary
    if success_count > 0:
        st.info(t("sidebar.upload_summary", "📋 Upload complete: {success} successful, {error} failed", success=success_count, error=error_count))
        
        # Auto-refresh to show new files
        if error_count == 0:
            st.rerun()


def _move_to_trash(file_path: Path) -> bool:
    """Move a file to trash using system-appropriate method."""
    try:
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            # Use native trash command
            subprocess.run(["trash", str(file_path)], check=True)
        elif system == "linux":
            # Try trash-cli first, fallback to gio
            try:
                subprocess.run(["trash-put", str(file_path)], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                try:
                    subprocess.run(["gio", "trash", str(file_path)], check=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Fallback: move to a .trash folder
                    trash_dir = file_path.parent / ".trash"
                    trash_dir.mkdir(exist_ok=True)
                    trash_file = trash_dir / file_path.name
                    
                    # Handle name conflicts
                    counter = 1
                    while trash_file.exists():
                        stem = file_path.stem
                        suffix = file_path.suffix
                        trash_file = trash_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    file_path.rename(trash_file)
        else:  # Windows or other
            # For Windows, move to Recycle Bin would require additional dependency
            # For now, move to .trash folder as fallback
            trash_dir = file_path.parent / ".trash"
            trash_dir.mkdir(exist_ok=True)
            trash_file = trash_dir / file_path.name
            
            # Handle name conflicts
            counter = 1
            while trash_file.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                trash_file = trash_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            file_path.rename(trash_file)
        
        return True
        
    except Exception as e:
        st.error(t("sidebar.trash_error", "❌ Failed to move file to trash: {error}", error=str(e)))
        return False


@st.dialog("Remove File Confirmation")
def _show_remove_file_dialog(file_path: Path, settings):
    """Show confirmation dialog for removing a single file."""
    st.warning(t("sidebar.remove_warning", "⚠️ **Warning**: This will move the file to trash!"))
    st.markdown(t("sidebar.file_to_remove", "File to remove:"))
    st.code(str(file_path))
    st.markdown(t("sidebar.trash_note", "📁 The file will be moved to your system's trash/recycle bin."))
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(t("sidebar.cancel", "❌ Cancel"), use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button(t("sidebar.confirm_remove", "🗑️ Remove"), type="primary", use_container_width=True):
            if _move_to_trash(file_path):
                st.success(t("sidebar.file_removed", "✅ File moved to trash: {filename}", filename=file_path.name))
                st.rerun()


@st.dialog("Remove Selected Files Confirmation")
def _show_batch_remove_dialog(selected_files, settings):
    """Show confirmation dialog for removing selected files."""
    selected_paths = [Path(f) for f in selected_files]
    
    st.warning(t("sidebar.batch_remove_warning", "⚠️ **Warning**: This will move selected files to trash!"))
    st.markdown(t("sidebar.files_to_remove_batch", "Files to remove:"))
    st.markdown(f"**{len(selected_paths)}** {t('sidebar.files_selected', 'files selected')}")
    
    # Show files to be removed
    st.markdown(t("sidebar.preview_files", "**Files to be removed:**"))
    for path in selected_paths:
        st.caption(f"🗑️ {path.name}")
    
    st.markdown(t("sidebar.trash_note", "📁 All files will be moved to your system's trash/recycle bin."))
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(t("sidebar.cancel", "❌ Cancel"), use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button(t("sidebar.confirm_remove_selected", "🗑️ Remove Selected"), type="primary", use_container_width=True):
            _handle_bulk_file_removal(selected_files)
            # Clear selection after removal
            if "selected_files" in st.session_state:
                st.session_state.selected_files.clear()
            st.rerun()


@st.dialog("Remove All Files Confirmation")
def _show_remove_all_dialog(available_files, settings):
    """Show confirmation dialog for removing all files."""
    st.error(t("sidebar.remove_all_warning", "⚠️ **DANGER**: This will move ALL documents to trash!"))
    st.markdown(t("sidebar.files_to_remove_all", "Files to remove:"))
    st.markdown(f"**{len(available_files)}** {t('sidebar.documents_found', 'documents found')}")
    
    # Show first few files as preview
    st.markdown(t("sidebar.preview_files", "**Preview of files to be removed:**"))
    for file in available_files[:10]:
        st.caption(f"🗑️ {file.name}")
    
    if len(available_files) > 10:
        st.caption(f"... {t('sidebar.and_more_files', 'and {count} more files', count=len(available_files) - 10)}")
    
    st.markdown(t("sidebar.trash_note", "📁 All files will be moved to your system's trash/recycle bin."))
    st.markdown(t("sidebar.action_irreversible", "⚠️ This action cannot be undone from within the app."))
    
    # Double confirmation
    confirm_text = st.text_input(
        t("sidebar.type_confirm", "Type 'REMOVE ALL' to confirm:"),
        placeholder="REMOVE ALL"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(t("sidebar.cancel", "❌ Cancel"), use_container_width=True):
            st.rerun()
    
    with col2:
        is_confirmed = confirm_text.strip().upper() == "REMOVE ALL"
        if st.button(
            t("sidebar.confirm_remove_all", "🗑️ Remove All Files"), 
            type="primary" if is_confirmed else "secondary",
            disabled=not is_confirmed,
            use_container_width=True
        ):
            if is_confirmed:
                _handle_remove_all_files(available_files)
                st.rerun()


@st.dialog("File Manager")
def _show_file_manager_dialog(available_files, settings):
    """Show dialog for managing all files."""
    st.markdown(f"### {t('sidebar.file_manager', '📁 File Manager')}")
    st.markdown(t("sidebar.total_files", "Total files: **{count}**", count=len(available_files)))
    
    # Create a list of files to remove
    if "files_to_remove" not in st.session_state:
        st.session_state.files_to_remove = set()
    
    # File selection
    st.markdown(t("sidebar.select_files_remove", "**Select files to remove:**"))
    
    # Select all / none buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button(t("sidebar.select_all", "☑️ Select All"), use_container_width=True):
            st.session_state.files_to_remove = set(str(f) for f in available_files)
            st.rerun()
    with col2:
        if st.button(t("sidebar.select_none", "☐ Select None"), use_container_width=True):
            st.session_state.files_to_remove.clear()
            st.rerun()
    
    # File checkboxes
    for file in available_files:
        file_str = str(file)
        is_selected = file_str in st.session_state.files_to_remove
        
        if st.checkbox(f"📄 {file.name}", value=is_selected, key=f"select_{file_str}"):
            st.session_state.files_to_remove.add(file_str)
        elif file_str in st.session_state.files_to_remove:
            st.session_state.files_to_remove.remove(file_str)
    
    st.divider()
    
    # Action buttons
    selected_count = len(st.session_state.files_to_remove)
    if selected_count > 0:
        st.warning(t("sidebar.files_selected", "⚠️ {count} files selected for removal", count=selected_count))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("sidebar.cancel", "❌ Cancel"), use_container_width=True):
                st.session_state.files_to_remove.clear()
                st.rerun()
        
        with col2:
            if st.button(t("sidebar.remove_selected", "🗑️ Remove Selected"), type="primary", use_container_width=True):
                _handle_bulk_file_removal(st.session_state.files_to_remove)
                st.session_state.files_to_remove.clear()
                st.rerun()
    else:
        if st.button(t("sidebar.close", "✅ Close"), use_container_width=True):
            st.rerun()


def _handle_bulk_file_removal(files_to_remove):
    """Handle removal of multiple files."""
    success_count = 0
    error_count = 0
    
    for file_str in files_to_remove:
        file_path = Path(file_str)
        if _move_to_trash(file_path):
            success_count += 1
        else:
            error_count += 1
    
    # Show summary
    if success_count > 0:
        st.success(t("sidebar.bulk_remove_success", "✅ Moved {count} files to trash", count=success_count))
    
    if error_count > 0:
        st.error(t("sidebar.bulk_remove_error", "❌ Failed to remove {count} files", count=error_count))
    
    st.info(t("sidebar.bulk_remove_summary", "📋 Removal complete: {success} successful, {error} failed", success=success_count, error=error_count))


def _handle_remove_all_files(available_files):
    """Handle removal of all files."""
    success_count = 0
    error_count = 0
    
    for file_path in available_files:
        if _move_to_trash(file_path):
            success_count += 1
        else:
            error_count += 1
    
    # Show summary
    if success_count > 0:
        st.success(t("sidebar.remove_all_success", "✅ Moved {count} files to trash", count=success_count))
    
    if error_count > 0:
        st.error(t("sidebar.remove_all_error", "❌ Failed to remove {count} files", count=error_count))
    
    st.info(t("sidebar.remove_all_summary", "📋 Remove all complete: {success} successful, {error} failed", success=success_count, error=error_count))