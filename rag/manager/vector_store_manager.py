import typing
from qdrant_client import QdrantClient
from injector import inject, singleton
from llama_index.core.vector_stores.types import VectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.indices.vector_store import VectorIndexRetriever

from rag.config import Config

@singleton
class VectorStoreManager:
    
    @inject
    def __init__(self, config: Config):
        try:
            client = QdrantClient(url="http://localhost:6333")
            self.vector_store = typing.cast(
                VectorStore,
                QdrantVectorStore(
                    client=client,
                    collection_name=config.QDRANT_COLLECTION,
                ),
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {str(e)}")

    def get_retriever(self, index, similarity_top_k: int = 5) -> VectorIndexRetriever:
        if not index:
            raise ValueError("Index cannot be None")
        return VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
        )

    def close(self) -> None:
        if hasattr(self, 'client'):
            self.client.close()
