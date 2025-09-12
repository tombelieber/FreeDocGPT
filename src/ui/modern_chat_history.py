import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import time

from ..config import get_settings
from ..core import ChatHistoryManager
from .i18n import t


def render_modern_chat_history(history_manager: ChatHistoryManager):
    """Render modern LobeHub/ChatGPT style chat history panel."""
    settings = get_settings()
    
    if not settings.enable_chat_history:
        return
    
    # Header with new chat button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header(t("chat_history.header", "💬 Chats"))
    with col2:
        if st.button("✏️", help=t("chat_history.new_chat", "New chat"), key="new_chat_btn"):
            _start_new_conversation()
    
    # Search bar (modern style)
    search_query = st.text_input(
        "",
        placeholder=t("chat_history.search_placeholder", "Search chats..."),
        key="chat_search",
        label_visibility="collapsed"
    )
    
    # Filter tabs
    tab1, tab2 = st.tabs([t("chat_history.recent", "Recent"), t("chat_history.pinned", "Pinned")])
    
    with tab1:
        _render_conversation_list(history_manager, search_query, show_pinned_only=False)
    
    with tab2:
        _render_conversation_list(history_manager, search_query, show_pinned_only=True)


def _render_conversation_list(history_manager: ChatHistoryManager, search_query: str = "", show_pinned_only: bool = False):
    """Render the conversation list with modern styling."""
    
    # Get conversations
    if search_query:
        conversations = history_manager.search_conversations(search_query, limit=50)
        if show_pinned_only:
            conversations = [c for c in conversations if c.get("pinned", False)]
    else:
        conversations = history_manager.list_conversations(limit=50)
        if show_pinned_only:
            conversations = [c for c in conversations if c.get("pinned", False)]
    
    current_conv_id = st.session_state.get("current_conversation_id")
    
    if not conversations:
        if search_query:
            st.caption(t("chat_history.no_search_results", "No chats found"))
        elif show_pinned_only:
            st.caption(t("chat_history.no_pinned", "No pinned chats"))
        else:
            st.caption(t("chat_history.no_conversations", "No chats yet"))
        return
    
    # Group conversations by time periods (LobeHub style)
    if not search_query and not show_pinned_only:
        grouped_conversations = _group_conversations_by_time(conversations)
        
        for period_name, convs in grouped_conversations.items():
            if convs:
                st.caption(f"**{period_name}**")
                for conv in convs:
                    _render_conversation_item(history_manager, conv, current_conv_id)
                st.write("")  # Add spacing between groups
    else:
        # Show flat list for search results or pinned
        for conv in conversations:
            _render_conversation_item(history_manager, conv, current_conv_id)


def _render_conversation_item(history_manager: ChatHistoryManager, conv: Dict, current_conv_id: Optional[str]):
    """Render a single conversation item with modern styling."""
    is_current = conv["id"] == current_conv_id
    is_pinned = conv.get("pinned", False)
    
    # Create container for the conversation item
    container = st.container()
    
    with container:
        # Main conversation row
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Conversation name and metadata
            conv_name = conv["name"]
            
            # Handle inline editing
            if st.session_state.get(f"editing_{conv['id']}", False):
                # Edit mode
                new_name = st.text_input(
                    "",
                    value=conv_name,
                    key=f"edit_name_{conv['id']}",
                    label_visibility="collapsed"
                )
                
                # Edit controls
                if st.button(t("common.save", "✓ Save"), key=f"save_edit_{conv['id']}", help=t("common.save", "Save")):
                    if new_name.strip() and new_name != conv_name:
                        history_manager.rename_conversation(conv["id"], new_name.strip())
                        st.rerun()
                    st.session_state[f"editing_{conv['id']}"] = False
                    st.rerun()
                
                if st.button(t("common.cancel", "✗ Cancel"), key=f"cancel_edit_{conv['id']}", help=t("common.cancel", "Cancel")):
                    st.session_state[f"editing_{conv['id']}"] = False
                    st.rerun()
            else:
                # Display mode
                # Style the conversation item based on state
                name_style = "**{name}**" if is_current else "{name}"
                pin_icon = "📌 " if is_pinned else ""
                
                display_name = name_style.format(name=pin_icon + conv_name)
                
                # Make conversation clickable
                if st.button(
                    display_name,
                    key=f"load_{conv['id']}",
                    help=t("chat_history.click_to_load", "Click to load this chat"),
                    use_container_width=True,
                    type="primary" if is_current else "secondary"
                ):
                    _load_conversation(history_manager, conv)
                
                # Show metadata (time and message count)
                time_str = _format_conversation_time(conv["updated_at"])
                msg_count = conv.get("message_count", 0)
                st.caption(f"{time_str} · {msg_count} messages")
        
        with col2:
            # Action menu (shows on hover in real UI, always visible in Streamlit)
            if not st.session_state.get(f"editing_{conv['id']}", False):
                action = st.selectbox(
                    "⋮",
                    ["", 
                     t("chat_history.edit", "✏️ Edit"),
                     t("chat_history.pin", "📌 Pin") if not is_pinned else t("chat_history.unpin", "📌 Unpin"),
                     t("chat_history.archive", "📁 Archive"),
                     t("chat_history.export", "📤 Export"),
                     t("chat_history.delete", "🗑️ Delete")],
                    key=f"action_{conv['id']}",
                    label_visibility="collapsed"
                )
                
                # Handle actions
                if action:
                    _handle_conversation_action(history_manager, conv, action)


def _handle_conversation_action(history_manager: ChatHistoryManager, conv: Dict, action: str):
    """Handle conversation actions."""
    conv_id = conv["id"]
    
    if "Edit" in action:
        st.session_state[f"editing_{conv_id}"] = True
        st.rerun()
    
    elif "Pin" in action:
        is_pinned = "Unpin" in action
        history_manager.pin_conversation(conv_id, not is_pinned)
        st.rerun()
    
    elif "Archive" in action:
        if st.session_state.get(f"confirm_archive_{conv_id}", False):
            history_manager.archive_conversation(conv_id, True)
            # If this is the current conversation, start a new one
            if st.session_state.get("current_conversation_id") == conv_id:
                _start_new_conversation()
            st.session_state[f"confirm_archive_{conv_id}"] = False
            st.rerun()
        else:
            st.session_state[f"confirm_archive_{conv_id}"] = True
            st.warning(f"Archive '{conv['name']}'?")
            if st.button(t("common.yes", "Yes"), key=f"confirm_archive_yes_{conv_id}"):
                st.session_state[f"confirm_archive_{conv_id}"] = False
                history_manager.archive_conversation(conv_id, True)
                # If this is the current conversation, start a new one
                if st.session_state.get("current_conversation_id") == conv_id:
                    _start_new_conversation()
                st.rerun()
            if st.button(t("common.cancel", "Cancel"), key=f"confirm_archive_no_{conv_id}"):
                st.session_state[f"confirm_archive_{conv_id}"] = False
                st.rerun()
    
    elif "Export" in action:
        _export_conversation(history_manager, conv)
    
    elif "Delete" in action:
        if st.session_state.get(f"confirm_delete_{conv_id}", False):
            history_manager.delete_conversation(conv_id)
            # If this is the current conversation, start a new one
            if st.session_state.get("current_conversation_id") == conv_id:
                _start_new_conversation()
            st.session_state[f"confirm_delete_{conv_id}"] = False
            st.rerun()
        else:
            st.session_state[f"confirm_delete_{conv_id}"] = True
            st.error(f"Delete '{conv['name']}'? This cannot be undone.")
            if st.button(t("common.delete", "Delete"), key=f"confirm_delete_yes_{conv_id}", type="primary"):
                st.session_state[f"confirm_delete_{conv_id}"] = False
                history_manager.delete_conversation(conv_id)
                # If this is the current conversation, start a new one
                if st.session_state.get("current_conversation_id") == conv_id:
                    _start_new_conversation()
                st.rerun()
            if st.button(t("common.cancel", "Cancel"), key=f"confirm_delete_no_{conv_id}"):
                st.session_state[f"confirm_delete_{conv_id}"] = False
                st.rerun()


def _load_conversation(history_manager: ChatHistoryManager, conv: Dict):
    """Load a conversation into the current session."""
    conversation_data = history_manager.load_conversation(conv["id"])
    if conversation_data:
        st.session_state.messages = conversation_data.get("messages", [])
        st.session_state.response_stats = []  # Reset stats for loaded conversation
        st.session_state.current_conversation_id = conv["id"]
        st.session_state.conversation_name = conv["name"]
        st.rerun()


def _start_new_conversation():
    """Start a new conversation (LobeHub/ChatGPT style)."""
    st.session_state.messages = []
    st.session_state.response_stats = []
    st.session_state.current_conversation_id = None
    st.session_state.conversation_name = None
    st.rerun()


def _export_conversation(history_manager: ChatHistoryManager, conv: Dict):
    """Export conversation with download button."""
    export_format = st.radio(
        t("chat_history.export_format", "Format"),
        ["markdown", "json"],
        key=f"export_format_{conv['id']}"
    )
    
    exported_content = history_manager.export_conversation(conv["id"], export_format)
    if exported_content:
        st.download_button(
            label=t("chat_history.download", "📥 Download"),
            data=exported_content,
            file_name=f"{conv['name']}.{export_format}",
            mime="text/markdown" if export_format == "markdown" else "application/json",
            key=f"download_{conv['id']}"
        )
    else:
        st.error(t("chat_history.export_error", "Export failed"))


def _group_conversations_by_time(conversations: List[Dict]) -> Dict[str, List[Dict]]:
    """Group conversations by time periods (Today, Yesterday, Last 7 days, etc.)."""
    now = datetime.now()
    today = now.date()
    yesterday = today - timedelta(days=1)
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)
    
    groups = {
        t("chat_history.today", "Today"): [],
        t("chat_history.yesterday", "Yesterday"): [],
        t("chat_history.last_7_days", "Last 7 days"): [],
        t("chat_history.last_30_days", "Last 30 days"): [],
        t("chat_history.older", "Older"): []
    }
    
    for conv in conversations:
        try:
            conv_date = datetime.fromisoformat(conv["updated_at"]).date()
            
            if conv_date == today:
                groups[t("chat_history.today", "Today")].append(conv)
            elif conv_date == yesterday:
                groups[t("chat_history.yesterday", "Yesterday")].append(conv)
            elif conv_date > week_ago:
                groups[t("chat_history.last_7_days", "Last 7 days")].append(conv)
            elif conv_date > month_ago:
                groups[t("chat_history.last_30_days", "Last 30 days")].append(conv)
            else:
                groups[t("chat_history.older", "Older")].append(conv)
        except (ValueError, TypeError):
            # If date parsing fails, put in older
            groups[t("chat_history.older", "Older")].append(conv)
    
    return groups


def _format_conversation_time(timestamp_str: str) -> str:
    """Format conversation timestamp in a user-friendly way."""
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        now = datetime.now()
        
        # If today, show time
        if timestamp.date() == now.date():
            return timestamp.strftime("%H:%M")
        
        # If this week, show day name
        days_ago = (now.date() - timestamp.date()).days
        if days_ago < 7:
            return timestamp.strftime("%a")
        
        # Otherwise show date
        if days_ago < 365:
            return timestamp.strftime("%b %d")
        else:
            return timestamp.strftime("%Y/%m/%d")
    
    except (ValueError, TypeError):
        return "Unknown"


def auto_save_current_conversation(history_manager: ChatHistoryManager):
    """Auto-save the current conversation (LobeHub/ChatGPT style).
    
    This is called automatically after each message exchange.
    Creates a new conversation on first message, updates existing ones.
    """
    messages = st.session_state.get("messages", [])
    if not messages:
        return
    
    # Only auto-save after assistant responses (complete exchanges)
    if messages[-1].get("role") != "assistant":
        return
    
    current_conv_id = st.session_state.get("current_conversation_id")
    
    # Auto-save the conversation
    saved_id = history_manager.auto_save_conversation(
        messages=messages,
        conversation_id=current_conv_id
    )
    
    # Update session state if we have a saved conversation
    if saved_id and not current_conv_id:
        # This is a new conversation, update session state
        st.session_state.current_conversation_id = saved_id
        
        # Load the conversation to get the auto-generated name
        conv_data = history_manager.load_conversation(saved_id)
        if conv_data:
            st.session_state.conversation_name = conv_data["name"]


def get_current_conversation_title() -> str:
    """Get the current conversation title for display in chat interface."""
    conv_name = st.session_state.get("conversation_name")
    if conv_name:
        return conv_name
    
    # If no name but we have messages, show a preview of the first message
    messages = st.session_state.get("messages", [])
    if messages:
        first_user_msg = next((msg for msg in messages if msg.get("role") == "user"), None)
        if first_user_msg:
            content = first_user_msg.get("content", "")
            return content[:30] + "..." if len(content) > 30 else content
    
    return t("chat_history.new_chat", "New Chat")