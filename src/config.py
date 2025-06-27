"""Configuration module for RAG QA System."""

import os
from typing import Optional
from pydantic import BaseModel, field_validator, Field


class Settings(BaseModel):
    """Application settings."""

    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4"
    embedding_model: str = "text-embedding-ada-002"

    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_environment: str = "us-east1-gcp"
    pinecone_index_name: str = "qa-documents"

    # Cohere Configuration
    cohere_api_key: Optional[str] = Field(default=None)

    # Application Settings
    max_file_size_mb: int = 50
    max_files_count: int = 100
    chunk_size: int = 1024
    chunk_overlap: int = 20
    similarity_top_k: int = 5
    similarity_threshold: float = 0.7

    # Enhancement Settings
    enable_self_correction: bool = Field(default=True)
    max_retry_attempts: int = Field(default=2)
    enable_contextual_extraction: bool = Field(default=True)
    reranker_top_n: int = Field(default=8)
    initial_retrieval_k: int = Field(default=20)

    # Database Settings
    database_url: str = "sqlite:///qa_system.db"

    def __init__(self, **kwargs):
        # Load from environment variables
        env_vars = {
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
            "pinecone_api_key": os.getenv("PINECONE_API_KEY", ""),
            "pinecone_environment": os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp"),
            "pinecone_index_name": os.getenv("PINECONE_INDEX_NAME", "qa-documents"),
            "cohere_api_key": os.getenv("COHERE_API_KEY", None),
            "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "50")),
            "max_files_count": int(os.getenv("MAX_FILES_COUNT", "100")),
            "chunk_size": int(os.getenv("CHUNK_SIZE", "1024")),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "20")),
            "similarity_top_k": int(os.getenv("SIMILARITY_TOP_K", "5")),
            "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
            "enable_self_correction": os.getenv("ENABLE_SELF_CORRECTION", "true").lower() == "true",
            "max_retry_attempts": int(os.getenv("MAX_RETRY_ATTEMPTS", "2")),
            "enable_contextual_extraction": os.getenv(
                "ENABLE_CONTEXTUAL_EXTRACTION", "true"
            ).lower()
            == "true",
            "reranker_top_n": int(os.getenv("RERANKER_TOP_N", "8")),
            "initial_retrieval_k": int(os.getenv("INITIAL_RETRIEVAL_K", "20")),
            "database_url": os.getenv("DATABASE_URL", "sqlite:///qa_system.db"),
        }
        env_vars.update(kwargs)
        super().__init__(**env_vars)

        # Validate required keys
        if not self.openai_api_key or self.openai_api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY must be set")
        if not self.pinecone_api_key or self.pinecone_api_key == "your_pinecone_api_key_here":
            raise ValueError("PINECONE_API_KEY must be set")

        # Warn about optional Cohere key
        if not self.cohere_api_key or self.cohere_api_key == "your_cohere_api_key_here":
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("COHERE_API_KEY not set - reranker will be disabled")


def get_settings() -> Settings:
    """Get application settings."""
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        from dotenv import load_dotenv
        
        # Try to load .env from multiple locations
        env_paths = [
            '.env',
            '../.env',
            os.path.join(os.path.dirname(__file__), '..', '.env'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
        ]
        
        env_loaded = False
        for env_path in env_paths:
            abs_path = os.path.abspath(env_path)
            if os.path.exists(abs_path):
                logger.info(f"Loading .env from: {abs_path}")
                load_dotenv(abs_path, override=True)
                env_loaded = True
                break
        
        if not env_loaded:
            logger.warning("No .env file found in any of the expected locations")
            logger.warning(f"Searched paths: {env_paths}")
            logger.warning(f"Current working directory: {os.getcwd()}")
            
    except ImportError:
        logger.warning("python-dotenv not available, using system environment variables only")
    
    # Debug environment variable loading
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        logger.error(f"Available env vars starting with OPENAI: {[k for k in os.environ.keys() if k.startswith('OPENAI')]}")
    else:
        logger.info(f"OPENAI_API_KEY loaded successfully (length: {len(openai_key)})")
    
    return Settings()
