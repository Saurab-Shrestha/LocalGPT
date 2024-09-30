import logging
from injector import inject, singleton
from llama_index.embeddings.ollama import OllamaEmbedding

logger = logging.getLogger(__name__)

@singleton
class EmbeddingManager:
    def __init__(self, config) -> None:
        self.embedding_model = OllamaEmbedding(
            model_name=config.EMBED_MODEL,
            base_url=config.OLLAMA_URL,
            ollama_additional_kwargs={"mirostat": 0},
        )
    