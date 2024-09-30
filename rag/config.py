from dataclasses import dataclass

@dataclass
class Config:
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "rag_collection"
    LOCAL_DATA_PATH: str = "local_data"
    SHOW_PROGRESS: bool = True
    
    OLLAMA_URL: str = "http://localhost:11434"
    EMBED_MODEL: str = "nomic-embed-text:latest"
    LLM_MODEL: str = "gemma2:2b"
    TEMPERATURE: float = 0.1
    TIMEOUT: float = 300.0
    SIMILARITY_TOP_K: int = 5
    SYSTEM_PROMPT: str = "You are a helpful AI assistant. Use the provided context to answer the user's questions."
    SIMILARITY_CUTTOFF: float = 0.2
    RERANK_TOP_K: int = 2
