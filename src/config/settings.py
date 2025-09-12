import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set

from dotenv import load_dotenv

load_dotenv()


def _getenv(key: str, default: str) -> str:
    """Return environment value or the provided default if unset or empty.

    This makes .env truly optional and also guards against empty-string
    values (e.g., KEY="") which would otherwise override sensible defaults
    and cause type conversions like int("") to fail.
    """
    value = os.getenv(key)
    if value is None:
        return default
    value_str = str(value).strip()
    return value_str if value_str != "" else default


@dataclass
class Settings:
    """Application configuration settings."""
    
    # Database settings
    db_dir: str = field(default_factory=lambda: _getenv("DB_DIR", ".lancedb"))
    table_name: str = field(default_factory=lambda: _getenv("TABLE_NAME", "docs"))
    
    # Model settings
    embed_model: str = field(default_factory=lambda: _getenv("EMBED_MODEL", "embeddinggemma:300m"))
    gen_model: str = field(default_factory=lambda: _getenv("GEN_MODEL", "gpt-oss:20b"))
    vision_model: str = field(default_factory=lambda: _getenv("VISION_MODEL", "llava:7b"))
    
    # Document settings
    documents_folder: str = field(default_factory=lambda: _getenv("DOCUMENTS_FOLDER", "documents"))
    
    # Supported file extensions
    supported_extensions: Set[str] = field(default_factory=lambda: {
        ".pdf", ".txt", ".text", ".md", ".markdown",
        ".docx", ".doc", ".html", ".htm", ".csv",
        ".xlsx", ".xls", ".json", ".xml", ".yml",
        ".yaml", ".log", ".rtf"
    })
    
    # Document type configurations
    doc_type_params: dict = field(default_factory=lambda: {
        "meeting": {"chunk_size": 800, "overlap": 100, "emoji": "📝"},
        "prd": {"chunk_size": 1500, "overlap": 300, "emoji": "📋"},
        "technical": {"chunk_size": 1800, "overlap": 400, "emoji": "💻"},
        "wiki": {"chunk_size": 1200, "overlap": 250, "emoji": "📚"},
        "general": {"chunk_size": 1200, "overlap": 200, "emoji": "📄"}
    })
    
    # Language configurations
    language_adjustments: dict = field(default_factory=lambda: {
        "cjk": {"chunk_multiplier": 0.7, "overlap_multiplier": 0.8},
        "mixed": {"chunk_multiplier": 0.85, "overlap_multiplier": 1.0}
    })
    
    # UI settings
    page_title: str = "📚 FreeDocBuddy"
    page_icon: str = "📚"
    layout: str = "wide"
    default_locale: str = field(default_factory=lambda: _getenv("DEFAULT_LOCALE", "en"))
    
    # Ollama settings
    ollama_host: str = field(default_factory=lambda: _getenv("OLLAMA_HOST", "http://localhost:11434"))
    
    # Hybrid search settings
    hybrid_search_enabled: bool = field(default_factory=lambda: _getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true")
    default_search_mode: str = field(default_factory=lambda: _getenv("DEFAULT_SEARCH_MODE", "hybrid"))
    hybrid_alpha: float = field(default_factory=lambda: float(_getenv("HYBRID_ALPHA", "0.5")))
    search_result_limit: int = field(default_factory=lambda: int(_getenv("SEARCH_RESULT_LIMIT", "5")))
    
    # Prompt settings
    system_prompt_path: str = field(default_factory=lambda: _getenv("SYSTEM_PROMPT_PATH", "rag_prompt.md"))
    
    # Deduplication settings
    dedup_enabled: bool = field(default_factory=lambda: _getenv("DEDUP_ENABLED", "true").lower() == "true")
    dedup_threshold: float = field(default_factory=lambda: float(_getenv("DEDUP_THRESHOLD", "0.95")))
    
    # Performance settings
    batch_size: int = field(default_factory=lambda: int(_getenv("BATCH_SIZE", "10")))
    max_workers: int = field(default_factory=lambda: int(_getenv("MAX_WORKERS", "8")))
    
    # Token chunking settings
    use_token_chunking: bool = field(default_factory=lambda: _getenv("USE_TOKEN_CHUNKING", "false").lower() == "true")
    max_chunk_tokens: int = field(default_factory=lambda: int(_getenv("MAX_CHUNK_TOKENS", "512")))
    chunk_overlap_tokens: int = field(default_factory=lambda: int(_getenv("CHUNK_OVERLAP_TOKENS", "50")))
    generation_model: str = field(default_factory=lambda: _getenv("GENERATION_MODEL", "gpt-oss:20b"))
    
    # Reranking settings
    use_reranking: bool = field(default_factory=lambda: _getenv("USE_RERANKING", "false").lower() == "true")
    reranker_model: str = field(default_factory=lambda: _getenv("RERANKER_MODEL", "balanced"))  # fast, balanced, accurate
    rerank_top_k: int = field(default_factory=lambda: int(_getenv("RERANK_TOP_K", "5")))

    # Chat settings
    chat_history_limit: int = field(default_factory=lambda: int(_getenv("CHAT_HISTORY_LIMIT", "20")))
    conversation_context_turns: int = field(default_factory=lambda: int(_getenv("CONVERSATION_CONTEXT_TURNS", "3")))
    enable_query_reformulation: bool = field(default_factory=lambda: _getenv("ENABLE_QUERY_REFORMULATION", "true").lower() == "true")
    
    # Chat History Persistence settings
    enable_chat_history: bool = field(default_factory=lambda: _getenv("ENABLE_CHAT_HISTORY", "true").lower() == "true")
    chat_history_dir: str = field(default_factory=lambda: _getenv("CHAT_HISTORY_DIR", ".chat_history"))
    auto_save_conversations: bool = field(default_factory=lambda: _getenv("AUTO_SAVE_CONVERSATIONS", "true").lower() == "true")
    auto_save_interval: int = field(default_factory=lambda: int(_getenv("AUTO_SAVE_INTERVAL", "5")))
    max_saved_conversations: int = field(default_factory=lambda: int(_getenv("MAX_SAVED_CONVERSATIONS", "100")))
    conversation_name_strategy: str = field(default_factory=lambda: _getenv("CONVERSATION_NAME_STRATEGY", "first_message"))  # "first_message" or "ai_summary"
    auto_cleanup_days: int = field(default_factory=lambda: int(_getenv("AUTO_CLEANUP_DAYS", "90")))

    # Ollama/network timeouts (seconds)
    ollama_connect_timeout: float = field(default_factory=lambda: float(_getenv("OLLAMA_CONNECT_TIMEOUT", "5.0")))
    ollama_read_timeout: float = field(default_factory=lambda: float(_getenv("OLLAMA_READ_TIMEOUT", "60.0")))
    ollama_status_timeout: float = field(default_factory=lambda: float(_getenv("OLLAMA_STATUS_TIMEOUT", "30.0")))
    # Note: chat streaming uses Ollama's client which does not expose timeouts;
    # this value is for future UI hints/fallbacks
    first_token_timeout: float = field(default_factory=lambda: float(_getenv("FIRST_TOKEN_TIMEOUT", "30.0")))
    
    # UI notification settings
    enable_completion_sound: bool = field(default_factory=lambda: _getenv("ENABLE_COMPLETION_SOUND", "true").lower() == "true")
    
    # Context management settings
    max_context_tokens: int = field(default_factory=lambda: int(_getenv("MAX_CONTEXT_TOKENS", "128000")))
    context_warning_threshold: float = field(default_factory=lambda: float(_getenv("CONTEXT_WARNING_THRESHOLD", "0.75")))
    context_critical_threshold: float = field(default_factory=lambda: float(_getenv("CONTEXT_CRITICAL_THRESHOLD", "0.85")))
    sliding_window_size: int = field(default_factory=lambda: int(_getenv("SLIDING_WINDOW_SIZE", "4000")))
    enable_context_indicator: bool = field(default_factory=lambda: _getenv("ENABLE_CONTEXT_INDICATOR", "true").lower() == "true")
    
    def get_documents_path(self) -> Path:
        """Get the documents folder path, creating it if necessary."""
        path = Path(self.documents_folder)
        path.mkdir(exist_ok=True)
        return path
    
    def get_db_path(self) -> Path:
        """Get the database directory path."""
        return Path(self.db_dir)


_settings = None


def get_settings() -> Settings:
    """Get singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
