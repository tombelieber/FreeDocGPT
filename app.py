"""
Document Q&A System - Main Application

A modular RAG (Retrieval-Augmented Generation) system for document Q&A.
"""

import logging

import streamlit as st

from src.config import get_settings
from src.core import (
    DatabaseManager,
    DocumentIndexer,
    SearchService,
    ChatService
)
from src.ui import (
    render_sidebar,
    render_chat_interface,
    render_settings_panel
)
from src.utils import setup_logging

# Setup logging
logger = setup_logging(log_level="INFO")


def initialize_services():
    """Initialize all required services."""
    db_manager = DatabaseManager()
    indexer = DocumentIndexer(db_manager=db_manager)
    search_service = SearchService(db_manager=db_manager)
    chat_service = ChatService()
    
    return db_manager, indexer, search_service, chat_service


def main():
    """Main application entry point."""
    # Configure page
    settings = get_settings()
    st.set_page_config(
        page_title=settings.page_title,
        page_icon=settings.page_icon,
        layout=settings.layout
    )
    
    # Title and description
    st.title(settings.page_title)
    st.markdown("Simply add documents to the `documents` folder and start asking questions!")
    
    # Initialize services
    db_manager, indexer, search_service, chat_service = initialize_services()
    
    # Render sidebar
    render_sidebar(db_manager, indexer)
    
    # Render settings in sidebar
    with st.sidebar:
        st.divider()
        render_settings_panel()
    
    # Render main chat interface
    render_chat_interface(search_service, chat_service)


if __name__ == "__main__":
    main()