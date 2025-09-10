"""
Document Q&A System - Main Application

A modular RAG (Retrieval-Augmented Generation) system for document Q&A.
"""

import logging

import streamlit as st
import streamlit.components.v1 as components

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
from src.ui.i18n import t, normalize_locale
from src.ui.locale_persistence import load_locale, save_locale
from src.utils import setup_logging

# Setup logging
logger = setup_logging(log_level="INFO")


def initialize_services():
    """Initialize all required services (cached across reruns)."""
    # Reuse existing instances to avoid duplicate initialization and log spam
    if 'services' in st.session_state:
        svc = st.session_state['services']
        return svc['db_manager'], svc['indexer'], svc['search_service'], svc['chat_service']

    db_manager = DatabaseManager()
    indexer = DocumentIndexer(db_manager=db_manager)
    search_service = SearchService(db_manager=db_manager)
    chat_service = ChatService()

    st.session_state['services'] = {
        'db_manager': db_manager,
        'indexer': indexer,
        'search_service': search_service,
        'chat_service': chat_service,
    }
    return db_manager, indexer, search_service, chat_service


def main():
    """Main application entry point."""
    # Configure page: must be the first Streamlit call
    settings = get_settings()
    st.set_page_config(
        page_title=settings.page_title,
        page_icon=settings.page_icon,
        layout=settings.layout
    )

    # Initialize locale once per session: URL param -> saved file -> default
    if "locale" not in st.session_state:
        # 1) URL query param (use modern API)
        try:
            params = st.query_params  # Mapping[str, str | list[str]]
        except Exception:
            params = {}
        qp_locale = None
        if isinstance(params, dict) and "locale" in params and params["locale"]:
            qp = params["locale"]
            qp_raw = qp[0] if isinstance(qp, list) and qp else (qp if isinstance(qp, str) else None)
            qp_locale = normalize_locale(qp_raw)

        # 2) Saved file
        file_locale = normalize_locale(load_locale()) if load_locale() else None

        default_norm = normalize_locale(getattr(settings, "default_locale", "en"))
        st.session_state["locale"] = qp_locale or file_locale or default_norm

        # 3) If no qp/file (we just used default), attempt one-time browser detection via JS by setting ?locale=
        if not qp_locale and not file_locale:
            components.html(
                """
                <script>
                (function(){
                  try {
                    var url = new URL(window.location.href);
                    if (!url.searchParams.get('locale')) {
                      var lang = (navigator.language || 'en').toLowerCase();
                      var code = 'en';
                      if (lang.startsWith('zh')) {
                        if (lang.includes('hans') || lang.includes('cn') || lang.includes('sg')) code = 'zh-Hans';
                        else code = 'zh-Hant';
                      } else if (lang.startsWith('es')) {
                        code = 'es';
                      } else if (lang.startsWith('ja') || lang.startsWith('jp')) {
                        code = 'ja';
                      }
                      url.searchParams.set('locale', code);
                      window.location.replace(url.toString());
                    }
                  } catch (e) { /* no-op */ }
                })();
                </script>
                """,
                height=0,
            )
    
    # Title and description
    st.title(settings.page_title)
    st.markdown(t("app.subtitle", "Free, local document buddy — drop files in `documents/` and start asking questions."))
    
    # Initialize services
    db_manager, indexer, search_service, chat_service = initialize_services()
    
    # Render sidebar
    render_sidebar(db_manager, indexer)
    
    # Render settings in sidebar
    with st.sidebar:
        st.divider()
        render_settings_panel(search_service)
    
    # Render main chat interface
    render_chat_interface(search_service, chat_service)


if __name__ == "__main__":
    main()
