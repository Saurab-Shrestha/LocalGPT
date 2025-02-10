from pathlib import Path
from typing import Optional, Dict
from pydantic_settings import BaseSettings
from rag.constants import (
    # Server settings
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_SHARE,
    
    # Paths
    DEFAULT_DATA_DIR,
    DEFAULT_ASSETS_DIR,
    DEFAULT_TEMP_DIR,
    DEFAULT_LOG_DIR,
    
    # Model settings
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    
    # Audio settings
    DEFAULT_SAMPLE_RATE,
    DEFAULT_RECORDING_DURATION,
    DEFAULT_AUDIO_CHANNELS,
    DEFAULT_AUDIO_FORMAT,
    
    # Vector store settings
    DEFAULT_COLLECTION_NAME,
    DEFAULT_VECTOR_DIMENSION,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    
    # Chat settings
    DEFAULT_MAX_HISTORY,
    
    # Supported file types
    SUPPORTED_DOCUMENT_TYPES,
    
    # Logging
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_LEVELS,

    # OLLAMA settings
    DEFAULT_OLLAMA_URL
)

class Config(BaseSettings):
    # Base paths

    BASE_DIR: Path = Path(__file__).parent.parent
    LOCAL_DATA_PATH: Path = BASE_DIR / DEFAULT_DATA_DIR
    ASSETS_DIR: Path = BASE_DIR / DEFAULT_ASSETS_DIR
    TEMP_DIR: Path = BASE_DIR / DEFAULT_TEMP_DIR
    LOG_DIR: Path = BASE_DIR / DEFAULT_LOG_DIR
    
    # Server settings
    HOST: str = DEFAULT_HOST
    PORT: int = DEFAULT_PORT
    SHARE: bool = DEFAULT_SHARE
    
    # Model settings
    LLM_MODEL: str = DEFAULT_LLM_MODEL
    EMBEDDING_MODEL: str = DEFAULT_EMBEDDING_MODEL
    CHUNK_SIZE: int = DEFAULT_CHUNK_SIZE
    CHUNK_OVERLAP: int = DEFAULT_CHUNK_OVERLAP
    MAX_TOKENS: int = DEFAULT_MAX_TOKENS
    TEMPERATURE: float = DEFAULT_TEMPERATURE
    
    # Voice settings
    RECORDING_DURATION: int = DEFAULT_RECORDING_DURATION
    SAMPLE_RATE: int = DEFAULT_SAMPLE_RATE
    AUDIO_CHANNELS: int = DEFAULT_AUDIO_CHANNELS
    AUDIO_FORMAT: str = DEFAULT_AUDIO_FORMAT
    
    # Vector store settings
    QDRANT_COLLECTION: str = DEFAULT_COLLECTION_NAME
    VECTOR_DIMENSION: int = DEFAULT_VECTOR_DIMENSION
    SIMILARITY_THRESHOLD: float = DEFAULT_SIMILARITY_THRESHOLD
    TOP_K: int = DEFAULT_TOP_K
    
    # Chat settings
    MAX_HISTORY: int = DEFAULT_MAX_HISTORY
    
    # File settings
    SUPPORTED_DOCUMENT_TYPES: Dict[str, str] = SUPPORTED_DOCUMENT_TYPES
    
    # Logging settings
    LOG_FORMAT: str = LOG_FORMAT
    LOG_DATE_FORMAT: str = LOG_DATE_FORMAT
    LOG_LEVEL: str = LOG_LEVELS["info"]
    
    # Optional API keys
    OPENAI_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None

    # OLLAMA settings
    OLLAMA_URL: str = DEFAULT_OLLAMA_URL
    
    class Config:
        env_file = ".env"

        env_file_encoding = 'utf-8'
        case_sensitive = True

    def setup_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        directories = [
            self.LOCAL_DATA_PATH,
            self.ASSETS_DIR,
            self.TEMP_DIR,
            self.LOG_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def setup_logging(self) -> None:
        """Configure logging for the application."""
        import logging
        import logging.handlers
        
        # Create log directory if it doesn't exist
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = self.LOG_DIR / "app.log"
        
        # Configure logging
        logging.basicConfig(
            level=self.LOG_LEVEL,
            format=self.LOG_FORMAT,
            datefmt=self.LOG_DATE_FORMAT,
            handlers=[
                logging.StreamHandler(),  # Console handler
                logging.handlers.RotatingFileHandler(  # File handler
                    log_file,
                    maxBytes=10485760,  # 10MB
                    backupCount=5,
                    encoding='utf-8'
                )
            ]
        )

    def initialize(self) -> None:
        """Initialize all necessary components."""
        self.setup_directories()
        self.setup_logging()
