import streamlit as st

from ..config import get_settings
from ..utils import display_ollama_status


def render_settings_panel(search_service=None):
    """Render the settings panel in the sidebar."""
    settings = get_settings()
    
    with st.expander("⚙️ Settings"):
        st.markdown("### 📊 Document Processing Settings")
        
        # Document Type Presets
        st.markdown("#### 🎯 Quick Presets")
        st.caption("Optimized for common docs: Notion/Lark exports, PDFs, tech docs")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            meeting_preset = st.button(
                "📝 Meeting Notes", 
                use_container_width=True,
                help="Meeting minutes, decisions, action items"
            )
        with col2:
            prd_preset = st.button(
                "📋 PRD/Specs", 
                use_container_width=True,
                help="Product requirements, design specs"
            )
        with col3:
            tech_preset = st.button(
                "💻 Tech Docs", 
                use_container_width=True,
                help="API docs, code documentation"
            )
        with col4:
            wiki_preset = st.button(
                "📚 Wiki/KB", 
                use_container_width=True,
                help="Knowledge base, how-to guides"
            )
        
        # Initialize session state
        if 'chunk_size' not in st.session_state:
            st.session_state.chunk_size = 1200
        if 'overlap_size' not in st.session_state:
            st.session_state.overlap_size = 200
        if 'top_k' not in st.session_state:
            st.session_state.top_k = 5
        
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
        
        # Manual controls
        chunk_size = st.slider(
            "📏 **Chunk Size** (characters per segment)",
            500, 2000,
            st.session_state.chunk_size,
            step=100,
            key="chunk_slider",
            help="For Notion/Lark docs with mixed content, 1500-1800 is recommended."
        )
        st.session_state.chunk_size = chunk_size
        
        overlap_size = st.slider(
            "🔄 **Overlap Size** (shared text between chunks)",
            0, 400,
            st.session_state.overlap_size,
            step=50,
            key="overlap_slider",
            help="For documents with code blocks, use 300-400."
        )
        st.session_state.overlap_size = overlap_size
        
        top_k = st.slider(
            "🔍 **Search Results** (number of relevant segments)",
            1, 10,
            st.session_state.top_k,
            key="topk_slider",
            help="For technical queries use 5-7, for simple Q&A use 3-5."
        )
        st.session_state.top_k = top_k
        
        # Show current configuration
        st.markdown("#### 📊 Current Configuration")
        config_cols = st.columns(3)
        
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
        
        # Best practices
        st.divider()
        st.markdown("#### 💡 Best Practices")
        st.markdown("""
        **📌 For Notion/Lark Export:**
        - Export as **Markdown (.md)** for best results
        - Notion: Settings → Export → Markdown & CSV
        - Lark/Feishu: More → Export → Markdown
        
        **📊 Recommended Settings:**
        - Meeting Notes: 600-800 chunk, 100 overlap
        - PRD/Specs: 1500 chunk, 300 overlap
        - Tech Docs: 1800 chunk, 400 overlap
        - Wiki/How-to: 1200 chunk, 200 overlap
        """)
        
        st.subheader("Prompt")
        from pathlib import Path
        # Resolve prompt path similarly to SearchService
        prompt_path = settings.system_prompt_path
        candidate = Path(prompt_path)
        if not candidate.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            candidate = repo_root / candidate
        st.caption("Active system prompt file:")
        st.code(str(candidate))
        # Optional preview (avoid nested expanders per Streamlit limitations)
        try:
            if candidate.exists():
                preview = candidate.read_text(encoding="utf-8")[:300]
                show_preview = st.checkbox("Preview first 300 chars", value=False)
                if show_preview:
                    st.text(preview)
            else:
                st.warning("Prompt file not found; using default built-in prompt.")
        except Exception as e:
            st.warning(f"Could not read prompt file: {e}")
        # Reload button
        if st.button("🔁 Reload System Prompt", help="Re-read prompt file without restarting"):
            if search_service is not None:
                try:
                    search_service.reload_system_prompt()
                    st.success("System prompt reloaded.")
                except Exception as e:
                    st.error(f"Failed to reload prompt: {e}")
            else:
                st.info("Search service not available to reload prompt.")

        st.subheader("Models")
        
        # Ollama prerequisite warning
        st.warning("""
        ⚠️ **Prerequisites Required:**
        1. Install Ollama: `brew install ollama`
        2. Start Ollama: `ollama serve`
        3. Pull required models:
           - `ollama pull embeddinggemma:300m`
           - `ollama pull gpt-oss:20b`
        """)
        
        embed_model = st.text_input(
            "Embedding Model", 
            value=settings.embed_model,
            help="Make sure this model is installed in Ollama"
        )
        
        gen_model = st.text_input(
            "Generation Model", 
            value=settings.gen_model,
            help="Make sure this model is installed in Ollama"
        )
        
        # Check Ollama connection
        if st.button("🔍 Check Ollama Status"):
            display_ollama_status()
