import logging
from injector import inject, singleton
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

from rag.config import Config
from rag.manager.base_manager import BaseManager

logger = logging.getLogger(__name__)

@singleton
class EmbeddingManager(BaseManager):
    embedding_model: BaseEmbedding | None = None

    @inject
    def __init__(self, config: Config):
        super().__init__(config)
        self._initialize_embeddings()
        logger.info(f"Initializing embedding model")

    def _initialize_embeddings(self) -> None:
        """Initialize the embedding model based on configuration."""
        try:
            self.embedding_model = OllamaEmbedding(
                model_name=self.config.EMBEDDING_MODEL,
                base_url=self.config.OLLAMA_URL,
                ollama_additional_kwargs={"mirostat": 0},
            )

            logger.info(f"Successfully initialized {self.config.EMBEDDING_MODEL} embedding model")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise