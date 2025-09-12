import streamlit as st

from ..config import get_settings
from ..utils import display_ollama_status
from .i18n import t


def render_settings_panel(search_service=None):
    """Render the settings panel in the sidebar."""
    settings = get_settings()
    
    with st.expander(t("settings.title", "⚙️ Settings")):
        st.markdown(f"### {t('settings.doc_processing', '📊 Document Processing Settings')}")
        
        # Document Type Presets
        st.markdown(f"#### {t('settings.quick_presets', '🎯 Quick Presets')}")
        st.caption(t("settings.quick_presets_caption", "Optimized for common docs: Notion/Lark exports, PDFs, tech docs"))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            meeting_preset = st.button(
                t("settings.preset_meeting", "📝 Meeting Notes"), 
                use_container_width=True,
                help=""
            )
        with col2:
            prd_preset = st.button(
                t("settings.preset_prd", "📋 PRD/Specs"), 
                use_container_width=True,
                help=""
            )
        with col3:
            tech_preset = st.button(
                t("settings.preset_tech", "💻 Tech Docs"), 
                use_container_width=True,
                help=""
            )
        with col4:
            wiki_preset = st.button(
                t("settings.preset_wiki", "📚 Wiki/KB"), 
                use_container_width=True,
                help=""
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
            st.success(t("settings.preset_applied_meeting", "📝 Applied Meeting Notes preset"))
            st.rerun()
        elif prd_preset:
            st.session_state.chunk_size = 1500
            st.session_state.overlap_size = 300
            st.session_state.top_k = 7
            st.success(t("settings.preset_applied_prd", "📋 Applied PRD/Specs preset"))
            st.rerun()
        elif tech_preset:
            st.session_state.chunk_size = 1800
            st.session_state.overlap_size = 400
            st.session_state.top_k = 5
            st.success(t("settings.preset_applied_tech", "💻 Applied Tech Docs preset"))
            st.rerun()
        elif wiki_preset:
            st.session_state.chunk_size = 1200
            st.session_state.overlap_size = 200
            st.session_state.top_k = 5
            st.success(t("settings.preset_applied_wiki", "📚 Applied Wiki/KB preset"))
            st.rerun()
        
        st.divider()
        
        # Manual controls
        chunk_size = st.slider(
            t("settings.chunk_size", "📏 Chunk Size (characters per segment)"),
            500, 2000,
            st.session_state.chunk_size,
            step=100,
            key="chunk_slider",
            help=t("settings.chunk_help", "For Notion/Lark docs with mixed content, 1500-1800 is recommended.")
        )
        st.session_state.chunk_size = chunk_size
        
        overlap_size = st.slider(
            t("settings.overlap_size", "🔄 Overlap Size (shared text between chunks)"),
            0, 400,
            st.session_state.overlap_size,
            step=50,
            key="overlap_slider",
            help=t("settings.overlap_help", "For documents with code blocks, use 300-400.")
        )
        st.session_state.overlap_size = overlap_size
        
        top_k = st.slider(
            t("settings.results", "🔍 Search Results (number of relevant segments)"),
            1, 10,
            st.session_state.top_k,
            key="topk_slider",
            help=t("settings.results_help", "For technical queries use 5-7, for simple Q&A use 3-5.")
        )
        st.session_state.top_k = top_k
        
        # Show current configuration
        st.markdown(f"#### {t('settings.current_config', '📊 Current Configuration')}")
        config_cols = st.columns(3)
        
        with config_cols[0]:
            st.metric(t("settings.metric_chunk", "Chunk Size"), f"{chunk_size} chars")
            if chunk_size < 1000:
                st.caption(t("settings.small_good", "⚠️ Small: Good for Q&A"))
            elif chunk_size > 1500:
                st.caption(t("settings.large_good", "📦 Large: Good for code"))
            else:
                st.caption(t("settings.balanced", "✅ Balanced"))
        
        with config_cols[1]:
            st.metric(t("settings.metric_overlap", "Overlap"), f"{overlap_size} chars")
            overlap_pct = (overlap_size / chunk_size * 100) if chunk_size > 0 else 0
            st.caption(t("settings.overlap_pct", "{pct:.0f}% of chunk size", pct=overlap_pct))
        
        with config_cols[2]:
            st.metric(t("settings.metric_results", "Results"), f"{top_k} chunks")
            if top_k <= 3:
                st.caption(t("settings.focused", "🎯 Focused search"))
            elif top_k >= 7:
                st.caption(t("settings.comprehensive", "🌐 Comprehensive"))
            else:
                st.caption(t("settings.balanced", "⚖️ Balanced"))
        
        # Best practices
        st.divider()
        st.markdown(f"#### {t('settings.best_practices', '💡 Best Practices')}")
        st.markdown(f"**{t('settings.bp_notion_title', '📌 For Notion/Lark Export:')}**\n{t('settings.bp_notion_list', '- Export as Markdown')}\n\n**{t('settings.bp_reco_title', '📊 Recommended Settings:')}**\n{t('settings.bp_reco_list', '- Meeting Notes: 600-800 chunk, 100 overlap')}")
        
        st.subheader(t("settings.prompt", "System Prompt"))
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
            t("settings.prompt_tab_edit", "✏️ Edit Prompt"),
            t("settings.prompt_tab_info", "ℹ️ File Info")
        ])
        
        with prompt_tab1:
            st.markdown(t("settings.prompt_edit_desc", "**Edit the system prompt directly:**"))
            
            # Text editor for system prompt
            edited_prompt = st.text_area(
                t("settings.prompt_editor", "System Prompt Content"),
                value=st.session_state.custom_system_prompt,
                height=300,
                help=t("settings.prompt_editor_help", "Edit the system prompt that guides AI responses. Changes are saved when you click 'Save Changes'.")
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(t("settings.save_prompt", "💾 Save Changes"), type="primary"):
                    try:
                        # Write the edited prompt to file
                        candidate.write_text(edited_prompt, encoding="utf-8")
                        st.session_state.custom_system_prompt = edited_prompt
                        
                        # Reload the search service if available
                        if search_service is not None:
                            search_service.reload_system_prompt()
                        
                        st.success(t("settings.prompt_saved", "✅ System prompt saved and reloaded!"))
                    except Exception as e:
                        st.error(t("settings.prompt_save_error", "❌ Error saving prompt: {err}", err=str(e)))
            
            with col2:
                if st.button(t("settings.reset_prompt", "🔄 Reset to File")):
                    try:
                        if candidate.exists():
                            st.session_state.custom_system_prompt = candidate.read_text(encoding="utf-8")
                            st.success(t("settings.prompt_reset", "✅ Prompt reset to file content"))
                        else:
                            st.warning(t("settings.prompt_file_missing", "⚠️ Prompt file not found"))
                        st.rerun()
                    except Exception as e:
                        st.error(t("settings.prompt_reset_error", "❌ Error resetting: {err}", err=str(e)))
            
            with col3:
                if st.button(t("settings.reload_prompt", "🔁 Reload Service")):
                    if search_service is not None:
                        try:
                            search_service.reload_system_prompt()
                            st.success(t("settings.reload_ok", "✅ System prompt service reloaded"))
                        except Exception as e:
                            st.error(t("settings.reload_fail", "❌ Failed to reload: {err}", err=str(e)))
                    else:
                        st.info(t("settings.reload_na", "⚠️ Search service not available"))
        
        with prompt_tab2:
            st.caption(t("settings.prompt_active", "**Active system prompt file:**"))
            st.code(str(candidate), language="text")
            
            # File status and preview
            try:
                if candidate.exists():
                    file_size = candidate.stat().st_size
                    st.caption(f"📄 File size: {file_size} bytes")
                    
                    # Show character count of current content
                    char_count = len(st.session_state.custom_system_prompt)
                    st.caption(f"✏️ Current content: {char_count} characters")
                    
                    # Preview toggle
                    show_preview = st.checkbox(t("settings.preview_first300", "Show preview (first 300 chars)"), value=False)
                    if show_preview:
                        preview = st.session_state.custom_system_prompt[:300]
                        if len(st.session_state.custom_system_prompt) > 300:
                            preview += "..."
                        st.text(preview)
                else:
                    st.warning(t("settings.prompt_not_found", "⚠️ Prompt file not found; using default built-in prompt."))
            except Exception as e:
                st.warning(t("settings.prompt_could_not_read", "⚠️ Could not read prompt file: {err}", err=e))

        st.subheader(t("settings.models", "Models"))
        
        # Ollama prerequisite warning
        st.warning(t("settings.ollama_prereq", "⚠️ Prerequisites Required:\n1. Install Ollama: `brew install ollama`\n2. Start Ollama: `ollama serve`\n3. Pull required models:\n   - `ollama pull embeddinggemma:300m`\n   - `ollama pull gpt-oss:20b`"))
        
        embed_model = st.text_input(
            t("settings.embed_model", "Embedding Model"), 
            value=settings.embed_model,
            help=t("settings.model_help", "Make sure this model is installed in Ollama")
        )
        
        gen_model = st.text_input(
            t("settings.gen_model", "Generation Model"), 
            value=settings.gen_model,
            help=t("settings.model_help", "Make sure this model is installed in Ollama")
        )
        
        # Check Ollama connection
        if st.button(t("settings.check_ollama", "🔍 Check Ollama Status")):
            display_ollama_status()

        st.subheader(t("settings.ui", "UI Settings"))
        
        # Sound notification toggle
        completion_sound = st.checkbox(
            t("settings.completion_sound", "🔊 Play sound when response completes"),
            value=st.session_state.get('enable_completion_sound', settings.enable_completion_sound),
            help=t("settings.completion_sound_help", "Play a notification sound when AI finishes generating a response")
        )
        
        # Update session state
        st.session_state['enable_completion_sound'] = completion_sound
