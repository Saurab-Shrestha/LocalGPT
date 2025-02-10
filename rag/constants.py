"""
Constants used throughout the RAG application.
These values serve as defaults and can be overridden via environment variables or Config class.
"""

from pathlib import Path

# File paths
DEFAULT_CACHE_DIR = Path.home() / ".rag-assistant" / "cache"
DEFAULT_MODEL_DIR = Path.home() / ".rag-assistant" / "models"

# Server configuration
DEFAULT_HOST = "0.0.0.0"  # Listen on all available interfaces
DEFAULT_PORT = 7860       # Default Gradio port
DEFAULT_SHARE = False     # Default sharing setting for Gradio

# File system paths
DEFAULT_DATA_DIR = "data"           # Directory for storing indexed data
DEFAULT_ASSETS_DIR = "assets"       # Directory for storing static assets
DEFAULT_TEMP_DIR = "temp"           # Directory for temporary files
DEFAULT_LOG_DIR = "logs"            # Directory for log files

DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Model configuration
DEFAULT_LLM_MODEL = "gemma2:2b"        # Default language model
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"  # Default embedding model
DEFAULT_CHUNK_SIZE = 500           # Default text chunk size for processing
DEFAULT_CHUNK_OVERLAP = 200         # Default overlap between chunks

# Audio settings
DEFAULT_SAMPLE_RATE = 16000         # Default audio sample rate in Hz
DEFAULT_RECORDING_DURATION = 5      # Default recording duration in seconds
DEFAULT_AUDIO_CHANNELS = 1          # Mono audio
DEFAULT_AUDIO_FORMAT = "wav"        # Default audio format
DEFAULT_AUDIO_DTYPE = "float32"     # Default audio data type

# Vector store settings
DEFAULT_COLLECTION_NAME = "documents"    # Default Qdrant collection name
DEFAULT_VECTOR_DIMENSION = 768           # Default embedding dimension
DEFAULT_SIMILARITY_THRESHOLD = 0.7       # Default similarity threshold
DEFAULT_TOP_K = 5                        # Default number of similar documents to retrieve

# Chat settings
DEFAULT_MAX_TOKENS = 2000               # Maximum tokens for response
DEFAULT_TEMPERATURE = 0.7               # Default temperature for LLM
DEFAULT_MAX_HISTORY = 10                # Maximum number of chat history items to keep

# API settings
DEFAULT_API_VERSION = "v1"              # API version
DEFAULT_API_PREFIX = "/api"             # API route prefix
DEFAULT_TIMEOUT = 30                    # Default timeout in seconds

# File processing
SUPPORTED_DOCUMENT_TYPES = {            # Supported document file extensions
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".json": "application/json",
    ".md": "text/markdown"
}

# Error messages
ERROR_MESSAGES = {
    "file_not_found": "The specified file could not be found.",
    "invalid_file_type": "The file type is not supported.",
    "processing_error": "An error occurred while processing the file.",
    "index_error": "An error occurred while indexing the document.",
    "query_error": "An error occurred while processing your query.",
    "voice_error": "An error occurred while processing voice input.",
    "model_error": "An error occurred while accessing the language model."
}

# Success messages
SUCCESS_MESSAGES = {
    "file_uploaded": "File successfully uploaded and processed.",
    "index_created": "Index successfully created.",
    "query_success": "Query processed successfully."
}

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVELS = {
    "debug": "DEBUG",
    "info": "INFO",
    "warning": "WARNING",
    "error": "ERROR",
    "critical": "CRITICAL"
}

# Voice constants
MAX_TEXT_LENGTH = 1000