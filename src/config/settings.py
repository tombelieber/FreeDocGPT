import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Application configuration settings."""
    
    # Database settings
    db_dir: str = field(default_factory=lambda: os.getenv("DB_DIR", ".lancedb"))
    table_name: str = field(default_factory=lambda: os.getenv("TABLE_NAME", "docs"))
    
    # Model settings
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "embeddinggemma:300m"))
    gen_model: str = field(default_factory=lambda: os.getenv("GEN_MODEL", "gpt-oss:20b"))
    vision_model: str = field(default_factory=lambda: os.getenv("VISION_MODEL", "llava:7b"))
    
    # Document settings
    documents_folder: str = field(default_factory=lambda: os.getenv("DOCUMENTS_FOLDER", "documents"))
    
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
    page_title: str = "📚 Document Q&A System"
    page_icon: str = "📚"
    layout: str = "wide"
    
    # Ollama settings
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    
    # Hybrid search settings
    hybrid_search_enabled: bool = field(default_factory=lambda: os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true")
    default_search_mode: str = field(default_factory=lambda: os.getenv("DEFAULT_SEARCH_MODE", "hybrid"))
    hybrid_alpha: float = field(default_factory=lambda: float(os.getenv("HYBRID_ALPHA", "0.5")))
    search_result_limit: int = field(default_factory=lambda: int(os.getenv("SEARCH_RESULT_LIMIT", "5")))
    
    # Deduplication settings
    dedup_enabled: bool = field(default_factory=lambda: os.getenv("DEDUP_ENABLED", "true").lower() == "true")
    dedup_threshold: float = field(default_factory=lambda: float(os.getenv("DEDUP_THRESHOLD", "0.95")))
    
    # Performance settings
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "10")))
    max_workers: int = field(default_factory=lambda: int(os.getenv("MAX_WORKERS", "8")))
    
    # Token chunking settings
    use_token_chunking: bool = field(default_factory=lambda: os.getenv("USE_TOKEN_CHUNKING", "false").lower() == "true")
    max_chunk_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_CHUNK_TOKENS", "512")))
    chunk_overlap_tokens: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP_TOKENS", "50")))
    generation_model: str = field(default_factory=lambda: os.getenv("GENERATION_MODEL", "gpt-oss:20b"))
    
    # Reranking settings
    use_reranking: bool = field(default_factory=lambda: os.getenv("USE_RERANKING", "false").lower() == "true")
    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL", "balanced"))  # fast, balanced, accurate
    rerank_top_k: int = field(default_factory=lambda: int(os.getenv("RERANK_TOP_K", "5")))
    
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