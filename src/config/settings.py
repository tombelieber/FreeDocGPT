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