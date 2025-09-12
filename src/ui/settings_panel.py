import streamlit as st

from ..config import get_settings
from ..utils import display_ollama_status
from .i18n import t


def render_settings_panel(search_service=None):
    """Render the settings panel with improved UIUX organization."""
    settings = get_settings()
    
    with st.expander(t("settings.title", "⚙️ Settings"), expanded=False):
        # Initialize session state
        if 'chunk_size' not in st.session_state:
            st.session_state.chunk_size = 1200
        if 'overlap_size' not in st.session_state:
            st.session_state.overlap_size = 200
        if 'top_k' not in st.session_state:
            st.session_state.top_k = 5
        
        # Organized tabs for better UIUX
        tab1, tab2, tab3, tab4 = st.tabs([
            t("settings.tab_quick", "⚡ Quick Setup"),
            t("settings.tab_processing", "🔧 Processing"), 
            t("settings.tab_models", "🤖 Models"),
            t("settings.tab_interface", "🎨 Interface")
        ])
        
        with tab1:
            _render_quick_setup_tab()
            
        with tab2:
            _render_processing_tab()
            
        with tab3:
            _render_models_tab(settings, search_service)
            
        with tab4:
            _render_interface_tab(settings)


def _render_quick_setup_tab():
    """Render the quick setup tab with essential controls."""
    
    # Sub-group 1: Document Type Presets
    with st.container():
        st.markdown("#### 🎯 Document Type Presets")
        st.caption("One-click optimization for common document types")
        
        col1, col2 = st.columns(2)
        
        with col1:
            meeting_preset = st.button(
                "📝 Meeting Notes", 
                use_container_width=True,
                help="Optimized for meeting notes and conversations"
            )
            prd_preset = st.button(
                "📋 PRD/Specs", 
                use_container_width=True,
                help="Best for product requirements and specifications"
            )
        
        with col2:
            tech_preset = st.button(
                "💻 Tech Docs", 
                use_container_width=True,
                help="Ideal for code documentation and technical guides"
            )
            wiki_preset = st.button(
                "📚 Wiki/KB", 
                use_container_width=True,
                help="Perfect for knowledge base articles"
            )
        
        # Apply presets
        if meeting_preset:
            st.session_state.chunk_size = 800
            st.session_state.overlap_size = 100
            st.session_state.top_k = 3
            st.success("📝 Applied Meeting Notes preset")
            st.rerun()
        elif prd_preset:
            st.session_state.chunk_size = 1500
            st.session_state.overlap_size = 300
            st.session_state.top_k = 7
            st.success("📋 Applied PRD/Specs preset")
            st.rerun()
        elif tech_preset:
            st.session_state.chunk_size = 1800
            st.session_state.overlap_size = 400
            st.session_state.top_k = 5
            st.success("💻 Applied Tech Docs preset")
            st.rerun()
        elif wiki_preset:
            st.session_state.chunk_size = 1200
            st.session_state.overlap_size = 200
            st.session_state.top_k = 5
            st.success("📚 Applied Wiki/KB preset")
            st.rerun()
    
    st.divider()
    
    # Sub-group 2: Current Configuration Overview
    with st.container():
        st.markdown("#### 📊 Current Configuration")
        st.caption("Real-time overview of your active settings")
        
        config_cols = st.columns(3)
        
        chunk_size = st.session_state.chunk_size
        overlap_size = st.session_state.overlap_size
        top_k = st.session_state.top_k
        
        with config_cols[0]:
            st.metric("Chunk Size", f"{chunk_size} chars")
            if chunk_size < 1000:
                st.caption("⚠️ Small: Good for Q&A")
            elif chunk_size > 1500:
                st.caption("📦 Large: Good for code")
            else:
                st.caption("✅ Balanced")
        
        with config_cols[1]:
            st.metric("Overlap", f"{overlap_size} chars")
            overlap_pct = (overlap_size / chunk_size * 100) if chunk_size > 0 else 0
            st.caption(f"{overlap_pct:.0f}% of chunk size")
        
        with config_cols[2]:
            st.metric("Results", f"{top_k} chunks")
            if top_k <= 3:
                st.caption("🎯 Focused search")
            elif top_k >= 7:
                st.caption("🌐 Comprehensive")
            else:
                st.caption("⚖️ Balanced")
    
    st.divider()
    
    # Sub-group 3: Quick Tips
    with st.expander("💡 Quick Tips & Best Practices", expanded=False):
        st.markdown("""
        **🚀 Getting Started:**
        - New user? Try **Meeting Notes** preset for most documents
        - Working with code? Use **Tech Docs** preset
        - Processing PDFs? **PRD/Specs** handles structured content well
        
        **🎯 Performance Tips:**
        - Smaller chunks = faster search, less context
        - Larger chunks = slower search, more context
        - Higher overlap = better coherence, more storage
        """)


def _render_processing_tab():
    """Render the document processing configuration tab."""
    
    # Sub-group 1: Core Processing Settings
    with st.container():
        st.markdown("#### 🔧 Core Processing Settings")
        st.caption("Fine-tune how documents are chunked and processed")
        
        chunk_size = st.slider(
            "📏 Chunk Size (characters per segment)",
            500, 2000,
            st.session_state.chunk_size,
            step=100,
            key="chunk_slider",
            help="For mixed content documents, 1500-1800 is recommended."
        )
        st.session_state.chunk_size = chunk_size
        
        overlap_size = st.slider(
            "🔄 Overlap Size (shared text between chunks)",
            0, 400,
            st.session_state.overlap_size,
            step=50,
            key="overlap_slider",
            help="For documents with code blocks, use 300-400."
        )
        st.session_state.overlap_size = overlap_size
    
    st.divider()
    
    # Sub-group 2: Search Configuration
    with st.container():
        st.markdown("#### 🔍 Search Configuration")
        st.caption("Control how many results are returned")
        
        top_k = st.slider(
            "Search Results (number of relevant segments)",
            1, 10,
            st.session_state.top_k,
            key="topk_slider",
            help="For technical queries use 5-7, for simple Q&A use 3-5."
        )
        st.session_state.top_k = top_k
        
        # Visual indicator for search scope
        scope_cols = st.columns(3)
        with scope_cols[0]:
            if top_k <= 3:
                st.info("🎯 **Focused Search**\nFaster, more precise results")
        with scope_cols[1]:
            if 4 <= top_k <= 6:
                st.info("⚖️ **Balanced Search**\nGood mix of speed and coverage")
        with scope_cols[2]:
            if top_k >= 7:
                st.info("🌐 **Comprehensive Search**\nMore context, slower processing")
    
    st.divider()
    
    # Sub-group 3: Document Type Guidelines
    with st.expander("📋 Document Type Guidelines", expanded=False):
        st.markdown("#### 🎯 Optimized Settings by Document Type")
        
        doc_cols = st.columns(2)
        
        with doc_cols[0]:
            st.markdown("""
            **📝 Text-Heavy Documents:**
            - Meeting Notes: 600-800 chunk, 100 overlap
            - Articles/Blogs: 800-1200 chunk, 150 overlap
            - Books/Long Form: 1200-1500 chunk, 200 overlap
            """)
            
        with doc_cols[1]:
            st.markdown("""
            **💻 Structured Documents:**
            - Technical Docs: 1500-1800 chunk, 300-400 overlap
            - Code Documentation: 1800-2000 chunk, 400 overlap
            - API References: 1000-1500 chunk, 200-300 overlap
            """)
    
    # Sub-group 4: Export Format Tips
    with st.expander("📤 Export Format Recommendations", expanded=False):
        st.markdown("""
        **📌 For Best Results:**
        
        **Notion/Lark Export:**
        - ✅ Export as Markdown
        - ✅ Include images and tables
        - ⚠️ Avoid HTML export (complex formatting)
        
        **PDF Processing:**
        - ✅ Text-based PDFs work best
        - ✅ Use higher chunk sizes (1500+)
        - ⚠️ Scanned PDFs may need OCR
        
        **Code Documentation:**
        - ✅ Markdown with code blocks
        - ✅ High overlap for code continuity
        - ✅ Include file structure context
        """)


def _render_models_tab(settings, search_service):
    """Render the models configuration tab."""
    
    # Sub-group 1: Setup & Prerequisites
    with st.expander("⚙️ Prerequisites & Setup Guide", expanded=False):
        st.markdown("#### 🛠️ Required Setup Steps")
        setup_cols = st.columns(2)
        
        with setup_cols[0]:
            st.markdown("""
            **1. Install Ollama:**
            ```bash
            brew install ollama
            ```
            
            **2. Start Ollama Service:**
            ```bash
            ollama serve
            ```
            """)
            
        with setup_cols[1]:
            st.markdown("""
            **3. Pull Required Models:**
            ```bash
            ollama pull embeddinggemma:300m
            ollama pull gpt-oss:20b
            ```
            """)
        
        st.info("💡 **Tip:** Keep Ollama running in the background for best performance")
    
    st.divider()
    
    # Sub-group 2: Model Configuration
    with st.container():
        st.markdown("#### 🤖 Model Configuration")
        st.caption("Configure which models to use for embeddings and generation")
        
        model_cols = st.columns(2)
        
        with model_cols[0]:
            embed_model = st.text_input(
                "🔤 Embedding Model", 
                value=settings.embed_model,
                help="Model used for document embeddings and search"
            )
            
        with model_cols[1]:
            gen_model = st.text_input(
                "💬 Generation Model", 
                value=settings.gen_model,
                help="Model used for response generation and chat"
            )
        
        # Check Ollama connection
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Check Ollama Status", use_container_width=True):
                display_ollama_status()
        with col2:
            if st.button("📋 List Available Models", use_container_width=True):
                st.info("Use `ollama list` in terminal to see installed models")
    
    st.divider()
    
    # Sub-group 3: System Prompt Configuration
    with st.container():
        st.markdown("#### 📝 System Prompt Configuration")
        st.caption("Customize how the AI responds to your queries")
        
        from pathlib import Path
        
        # Resolve prompt path similarly to SearchService
        prompt_path = settings.system_prompt_path
        candidate = Path(prompt_path)
        if not candidate.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            candidate = repo_root / candidate
        
        # Initialize session state for custom prompt
        if 'custom_system_prompt' not in st.session_state:
            try:
                if candidate.exists():
                    st.session_state.custom_system_prompt = candidate.read_text(encoding="utf-8")
                else:
                    st.session_state.custom_system_prompt = ""
            except Exception:
                st.session_state.custom_system_prompt = ""
        
        # Tab interface for prompt customization
        prompt_tab1, prompt_tab2 = st.tabs([
            "✏️ Edit Prompt",
            "ℹ️ File Info"
        ])
        
        with prompt_tab1:
            st.markdown("**Customize AI Behavior:**")
            
            # Text editor for system prompt
            edited_prompt = st.text_area(
                "System Prompt Content",
                value=st.session_state.custom_system_prompt,
                height=250,
                help="Edit the system prompt that guides AI responses and behavior."
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💾 Save Changes", type="primary", use_container_width=True):
                    try:
                        # Write the edited prompt to file
                        candidate.write_text(edited_prompt, encoding="utf-8")
                        st.session_state.custom_system_prompt = edited_prompt
                        
                        # Reload the search service if available
                        if search_service is not None:
                            search_service.reload_system_prompt()
                        
                        st.success("✅ System prompt saved and reloaded!")
                    except Exception as e:
                        st.error(f"❌ Error saving prompt: {str(e)}")
            
            with col2:
                if st.button("🔄 Reset", use_container_width=True):
                    try:
                        if candidate.exists():
                            st.session_state.custom_system_prompt = candidate.read_text(encoding="utf-8")
                            st.success("✅ Prompt reset to file content")
                        else:
                            st.warning("⚠️ Prompt file not found")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error resetting: {str(e)}")
            
            with col3:
                if st.button("🔁 Reload", use_container_width=True):
                    if search_service is not None:
                        try:
                            search_service.reload_system_prompt()
                            st.success("✅ System prompt service reloaded")
                        except Exception as e:
                            st.error(f"❌ Failed to reload: {str(e)}")
                    else:
                        st.info("⚠️ Search service not available")
        
        with prompt_tab2:
            st.markdown("**File Information:**")
            st.code(str(candidate), language="text")
            
            # File status and preview
            try:
                if candidate.exists():
                    file_size = candidate.stat().st_size
                    info_cols = st.columns(2)
                    
                    with info_cols[0]:
                        st.metric("File Size", f"{file_size} bytes")
                    
                    with info_cols[1]:
                        char_count = len(st.session_state.custom_system_prompt)
                        st.metric("Content Length", f"{char_count} chars")
                    
                    # Preview toggle
                    show_preview = st.checkbox("Show content preview", value=False)
                    if show_preview:
                        st.markdown("**Preview (first 300 characters):**")
                        preview = st.session_state.custom_system_prompt[:300]
                        if len(st.session_state.custom_system_prompt) > 300:
                            preview += "..."
                        st.text(preview)
                else:
                    st.warning("⚠️ Prompt file not found; using default built-in prompt.")
            except Exception as e:
                st.warning(f"⚠️ Could not read prompt file: {e}")
    
    # Sub-group 4: Model Performance Tips
    with st.expander("🚀 Performance & Optimization Tips", expanded=False):
        st.markdown("""
        #### 💡 Model Selection Tips
        
        **Embedding Models:**
        - `nomic-embed-text`: Fast, good for general documents
        - `embeddinggemma:300m`: Lightweight, mobile-friendly
        - `mxbai-embed-large`: Higher quality, more memory
        
        **Generation Models:**
        - `gpt-oss:20b`: Balanced performance and speed
        - `llama3:8b`: Faster, good for simple queries
        - `llama3:70b`: Higher quality, requires more resources
        
        #### ⚡ Performance Optimization
        - Keep Ollama service running continuously
        - Use smaller models for faster responses
        - Monitor system resources during heavy usage
        """)


def _render_interface_tab(settings):
    """Render the interface customization tab."""
    
    # Sub-group 1: Notification Settings
    with st.container():
        st.markdown("#### 🔔 Notifications & Feedback")
        st.caption("Configure audio and visual feedback")
        
        # Sound notification toggle
        completion_sound = st.checkbox(
            "🔊 Play sound when response completes",
            value=st.session_state.get('enable_completion_sound', settings.enable_completion_sound),
            help="Play a notification sound when AI finishes generating a response"
        )
        
        # Update session state
        st.session_state['enable_completion_sound'] = completion_sound
        
        # Visual feedback settings (placeholder for future)
        st.checkbox(
            "✨ Show visual completion indicator", 
            value=True, 
            disabled=True,
            help="Coming soon: Visual indicators for response completion"
        )
    
    st.divider()
    
    # Sub-group 2: Display Preferences
    with st.container():
        st.markdown("#### 🖥️ Display & Layout")
        st.caption("Customize the appearance and layout")
        
        # Placeholder controls for future display settings
        display_cols = st.columns(2)
        
        with display_cols[0]:
            st.selectbox(
                "🎨 Color Scheme", 
                ["Auto (System)", "Light", "Dark"],
                index=0,
                disabled=True,
                help="Coming soon: Theme customization"
            )
            
        with display_cols[1]:
            st.selectbox(
                "📏 Text Size",
                ["Small", "Medium", "Large"],
                index=1,
                disabled=True,
                help="Coming soon: Adjustable text sizing"
            )
    
    st.divider()
    
    # Sub-group 3: Behavior Settings
    with st.container():
        st.markdown("#### ⚡ Behavior & Performance")
        st.caption("Control how the interface behaves")
        
        behavior_cols = st.columns(2)
        
        with behavior_cols[0]:
            st.checkbox(
                "🏃 Auto-scroll to latest message",
                value=True,
                disabled=True,
                help="Coming soon: Auto-scroll behavior control"
            )
            
        with behavior_cols[1]:
            st.checkbox(
                "💾 Auto-save chat history",
                value=True,
                disabled=True,
                help="Coming soon: Automatic chat history saving"
            )
    
    st.divider()
    
    # Sub-group 4: Advanced Interface Options
    with st.expander("🚀 Coming Soon - Advanced Options", expanded=False):
        st.markdown("#### 🎯 Planned Interface Features")
        
        feature_cols = st.columns(2)
        
        with feature_cols[0]:
            st.markdown("""
            **🎨 Visual Customization:**
            - 🌙 Dark/Light theme toggle
            - 🎨 Custom color schemes
            - 📱 Mobile-responsive layouts
            - 🖼️ Background customization
            """)
            
        with feature_cols[1]:
            st.markdown("""
            **⚡ Interaction Features:**
            - ⌨️ Keyboard shortcuts
            - 📋 Quick action buttons  
            - 🔍 Enhanced search UI
            - 📊 Usage statistics display
            """)
        
        st.info("💡 **Want a specific feature?** These will be added based on user feedback and usage patterns.")
    
    # Sub-group 5: Current Session Info
    with st.expander("📊 Current Session Information", expanded=False):
        st.markdown("#### 🔍 Session Details")
        
        session_cols = st.columns(3)
        
        with session_cols[0]:
            st.metric(
                "Active Settings",
                f"Chunk: {st.session_state.get('chunk_size', 1200)}"
            )
            
        with session_cols[1]:
            st.metric(
                "Search Mode", 
                st.session_state.get('search_mode', 'hybrid').title()
            )
            
        with session_cols[2]:
            st.metric(
                "Results Limit",
                st.session_state.get('top_k', 5)
            )
