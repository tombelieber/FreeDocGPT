import streamlit as st
from datetime import datetime
from typing import Optional, List, Dict

from ..config import get_settings
from ..core import ChatHistoryManager
from .i18n import t


def render_chat_history_panel(history_manager: ChatHistoryManager):
    """Render the chat history management panel in the sidebar."""
    settings = get_settings()
    
    # Skip if chat history is disabled
    if not settings.enable_chat_history:
        return
    
    with st.sidebar:
        st.header(t("chat_history.header", "💬 Chat History"))
        
        # Search conversations
        search_query = st.text_input(
            t("chat_history.search_placeholder", "🔍 Search conversations..."),
            placeholder=t("chat_history.search_placeholder", "Search conversations..."),
            key="chat_history_search"
        )
        
        # Load conversations
        if search_query:
            conversations = history_manager.search_conversations(search_query, limit=20)
        else:
            conversations = history_manager.list_conversations(limit=20)
        
        if conversations:
            st.caption(t("chat_history.found_count", "Found {count} conversation(s)", count=len(conversations)))
            
            # Conversation list
            for conv in conversations:
                with st.container():
                    # Create columns for conversation info and actions
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Conversation name (truncated)
                        conv_name = conv["name"]
                        if len(conv_name) > 30:
                            display_name = conv_name[:27] + "..."
                        else:
                            display_name = conv_name
                        
                        # Load conversation button
                        if st.button(
                            f"📄 {display_name}",
                            key=f"load_{conv['id']}",
                            help=t("chat_history.load_tooltip", "Click to load this conversation"),
                            use_container_width=True
                        ):
                            # Load conversation into session state
                            conversation_data = history_manager.load_conversation(conv["id"])
                            if conversation_data:
                                st.session_state.messages = conversation_data.get("messages", [])
                                st.session_state.current_conversation_id = conv["id"]
                                st.session_state.conversation_name = conv["name"]
                                st.session_state.conversation_saved = True
                                st.success(t("chat_history.loaded", "Loaded: {name}", name=conv["name"]))
                                st.rerun()
                        
                        # Show conversation info
                        created_date = datetime.fromisoformat(conv["created_at"]).strftime("%m/%d %H:%M")
                        st.caption(f"📅 {created_date} • {conv['message_count']} msgs")
                        
                        # Show keywords if available
                        if conv.get("keywords"):
                            keywords = ", ".join(conv["keywords"][:3])
                            st.caption(f"🏷️ {keywords}")
                    
                    with col2:
                        # Action menu
                        action_key = f"action_{conv['id']}"
                        action = st.selectbox(
                            "⚙️",
                            ["", t("chat_history.rename", "Rename"), t("chat_history.delete", "Delete"), t("chat_history.export", "Export")],
                            key=action_key,
                            label_visibility="collapsed"
                        )
                        
                        # Handle actions
                        if action == t("chat_history.rename", "Rename"):
                            _handle_rename_conversation(history_manager, conv)
                        elif action == t("chat_history.delete", "Delete"):
                            _handle_delete_conversation(history_manager, conv)
                        elif action == t("chat_history.export", "Export"):
                            _handle_export_conversation(history_manager, conv)
                    
                    st.divider()
        else:
            if search_query:
                st.info(t("chat_history.no_search_results", "No conversations found matching '{query}'", query=search_query))
            else:
                st.info(t("chat_history.no_conversations", "No saved conversations yet"))
        
        # New conversation button
        if st.button(
            t("chat_history.new_conversation", "🆕 New Conversation"),
            use_container_width=True,
            type="secondary"
        ):
            # Clear current conversation
            st.session_state.messages = []
            st.session_state.response_stats = []
            st.session_state.current_conversation_id = None
            st.session_state.conversation_name = None
            st.session_state.conversation_saved = False
            st.rerun()
        
        # Current conversation info
        if st.session_state.get("current_conversation_id"):
            st.divider()
            st.subheader(t("chat_history.current_conversation", "Current Conversation"))
            
            current_name = st.session_state.get("conversation_name", "Untitled")
            is_saved = st.session_state.get("conversation_saved", False)
            
            # Show current conversation name with edit option
            new_name = st.text_input(
                t("chat_history.conversation_name", "Conversation Name"),
                value=current_name,
                key="current_conversation_name"
            )
            
            # Update name if changed
            if new_name != current_name and new_name.strip():
                st.session_state.conversation_name = new_name
                if st.session_state.get("current_conversation_id"):
                    history_manager.rename_conversation(
                        st.session_state.current_conversation_id,
                        new_name
                    )
                    st.session_state.conversation_saved = True
                    st.success(t("chat_history.renamed", "Conversation renamed"))
                    st.rerun()
            
            # Save status
            if is_saved:
                st.success(t("chat_history.status_saved", "✅ Saved"))
            else:
                st.warning(t("chat_history.status_unsaved", "⚠️ Unsaved changes"))
                
            # Manual save button
            if st.button(
                t("chat_history.save_conversation", "💾 Save Conversation"),
                use_container_width=True,
                disabled=len(st.session_state.get("messages", [])) == 0
            ):
                _save_current_conversation(history_manager)
        
        # Settings section
        st.divider()
        with st.expander(t("chat_history.settings", "⚙️ History Settings"), expanded=False):
            # Auto-save toggle
            auto_save = st.checkbox(
                t("chat_history.auto_save", "Auto-save conversations"),
                value=st.session_state.get("auto_save_enabled", settings.auto_save_conversations),
                help=t("chat_history.auto_save_help", "Automatically save conversations every few messages")
            )
            st.session_state.auto_save_enabled = auto_save
            
            if auto_save:
                interval = st.slider(
                    t("chat_history.save_interval", "Save every N messages"),
                    min_value=1,
                    max_value=20,
                    value=st.session_state.get("auto_save_interval", settings.auto_save_interval),
                    help=t("chat_history.interval_help", "How often to auto-save during conversation")
                )
                st.session_state.auto_save_interval = interval
            
            # Cleanup button
            if st.button(t("chat_history.cleanup", "🧹 Cleanup Old Conversations")):
                cleanup_days = settings.auto_cleanup_days
                deleted_count = history_manager.cleanup_old_conversations(cleanup_days)
                if deleted_count > 0:
                    st.success(t("chat_history.cleaned_up", "Deleted {count} old conversations", count=deleted_count))
                    st.rerun()
                else:
                    st.info(t("chat_history.no_cleanup_needed", "No old conversations to clean up"))


def _handle_rename_conversation(history_manager: ChatHistoryManager, conv: Dict):
    """Handle conversation renaming."""
    # Create a unique key for this rename dialog
    rename_key = f"rename_input_{conv['id']}"
    
    if rename_key not in st.session_state:
        st.session_state[rename_key] = conv["name"]
    
    # Show rename dialog
    with st.form(f"rename_form_{conv['id']}"):
        new_name = st.text_input(
            t("chat_history.new_name", "New name"),
            value=st.session_state[rename_key],
            key=f"new_name_{conv['id']}"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button(t("chat_history.save", "Save")):
                if new_name.strip() and new_name != conv["name"]:
                    if history_manager.rename_conversation(conv["id"], new_name.strip()):
                        st.success(t("chat_history.rename_success", "Conversation renamed"))
                        # Clear the session state to close the dialog
                        del st.session_state[rename_key]
                        st.rerun()
                    else:
                        st.error(t("chat_history.rename_error", "Failed to rename conversation"))
        
        with col2:
            if st.form_submit_button(t("chat_history.cancel", "Cancel")):
                # Clear the session state to close the dialog
                if rename_key in st.session_state:
                    del st.session_state[rename_key]
                st.rerun()


def _handle_delete_conversation(history_manager: ChatHistoryManager, conv: Dict):
    """Handle conversation deletion."""
    # Show confirmation dialog
    confirm_key = f"delete_confirm_{conv['id']}"
    
    if st.session_state.get(confirm_key, False):
        st.warning(t("chat_history.delete_warning", "⚠️ Delete '{name}'?", name=conv["name"]))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("chat_history.delete_confirm", "Delete"), key=f"delete_yes_{conv['id']}", type="primary"):
                if history_manager.delete_conversation(conv["id"]):
                    # If this is the current conversation, clear it
                    if st.session_state.get("current_conversation_id") == conv["id"]:
                        st.session_state.messages = []
                        st.session_state.response_stats = []
                        st.session_state.current_conversation_id = None
                        st.session_state.conversation_name = None
                        st.session_state.conversation_saved = False
                    
                    st.success(t("chat_history.delete_success", "Conversation deleted"))
                    st.session_state[confirm_key] = False
                    st.rerun()
                else:
                    st.error(t("chat_history.delete_error", "Failed to delete conversation"))
        
        with col2:
            if st.button(t("chat_history.cancel", "Cancel"), key=f"delete_no_{conv['id']}"):
                st.session_state[confirm_key] = False
                st.rerun()
    else:
        # Show delete button
        st.session_state[confirm_key] = True


def _handle_export_conversation(history_manager: ChatHistoryManager, conv: Dict):
    """Handle conversation export."""
    export_format = st.selectbox(
        t("chat_history.export_format", "Format"),
        ["markdown", "json"],
        key=f"export_format_{conv['id']}"
    )
    
    if st.button(t("chat_history.export_button", "Export"), key=f"export_btn_{conv['id']}"):
        exported_content = history_manager.export_conversation(conv["id"], export_format)
        if exported_content:
            st.download_button(
                label=t("chat_history.download", "Download"),
                data=exported_content,
                file_name=f"{conv['name']}.{export_format}",
                mime="text/markdown" if export_format == "markdown" else "application/json",
                key=f"download_{conv['id']}"
            )
        else:
            st.error(t("chat_history.export_error", "Failed to export conversation"))


def _save_current_conversation(history_manager: ChatHistoryManager):
    """Save the current conversation."""
    messages = st.session_state.get("messages", [])
    if not messages:
        st.warning(t("chat_history.no_messages", "No messages to save"))
        return
    
    conversation_name = st.session_state.get("conversation_name", "")
    if not conversation_name.strip():
        # Generate automatic name
        settings = get_settings()
        if messages:
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if user_messages:
                first_message = user_messages[0].get("content", "")
                conversation_name = first_message[:50] + ("..." if len(first_message) > 50 else "")
            else:
                conversation_name = "Untitled Conversation"
    
    conversation_id = st.session_state.get("current_conversation_id")
    
    try:
        saved_id = history_manager.save_conversation(
            messages=messages,
            conversation_name=conversation_name,
            conversation_id=conversation_id
        )
        
        if saved_id:
            st.session_state.current_conversation_id = saved_id
            st.session_state.conversation_name = conversation_name
            st.session_state.conversation_saved = True
            st.success(t("chat_history.save_success", "Conversation saved: {name}", name=conversation_name))
        else:
            st.error(t("chat_history.save_error", "Failed to save conversation"))
    except Exception as e:
        st.error(t("chat_history.save_error", "Failed to save conversation: {error}", error=str(e)))


def check_auto_save(history_manager: ChatHistoryManager):
    """Check if auto-save should be triggered and save if needed."""
    if not st.session_state.get("auto_save_enabled", True):
        return
    
    messages = st.session_state.get("messages", [])
    if not messages:
        return
    
    # Check if we should auto-save
    auto_save_interval = st.session_state.get("auto_save_interval", 5)
    message_count = len(messages)
    
    # Auto-save every N messages (and only on assistant responses to avoid mid-conversation saves)
    if (message_count > 0 and 
        message_count % auto_save_interval == 0 and 
        messages[-1].get("role") == "assistant"):
        
        # Don't auto-save too frequently (check if we saved recently)
        last_save_count = st.session_state.get("last_auto_save_count", 0)
        if message_count > last_save_count:
            _save_current_conversation(history_manager)
            st.session_state.last_auto_save_count = message_count