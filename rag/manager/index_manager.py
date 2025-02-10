import os
import logging
import threading
from typing import List, Optional
from pathlib import Path
from injector import inject, singleton
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex, load_index_from_storage, Document
from llama_index.core.data_structs import IndexDict
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import TransformComponent
from rag.manager.base_manager import BaseManager
from rag.manager.vector_store_manager import VectorStoreManager
from rag.config import Config

from llama_index.core.storage.docstore import BaseDocumentStore, SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
# from llama_index.storage.graph_store import SimpleGraphStore
from rag.manager.node_manager import NodeManager
from rag.manager.embed_manager import EmbeddingManager

logger = logging.getLogger(__name__)

@singleton
class IndexManager(BaseManager):

    @inject
    def __init__(
        self, 
        config: Config, 
        node_manager: NodeManager,
        vector_store_manager: VectorStoreManager,
        embed_manager: EmbeddingManager
    ):
        super().__init__(config)
        logger.info(f"Initializing IndexManager")
        self.vector_store_manager = vector_store_manager
        self.node_manager = node_manager
        self.embed_manager = embed_manager
        
        # Initialize components
        self.storage_context = self._initialize_storage_context()
        self.transformations: List[TransformComponent] = []
        self._index_thread_lock = threading.Lock()
        self.show_progress = True
        
        # Initialize index
        self.index = self._initialize_index()

    def _initialize_storage_context(self) -> StorageContext:
        """Initialize storage context with all required components"""
        try:
            return StorageContext.from_defaults(
                docstore=self.node_manager.doc_store,
                index_store=self.node_manager.index_store,
                vector_store=self.vector_store_manager.vector_store
            )
        except Exception as e:
            logger.error(f"Failed to initialize storage context: {str(e)}")
            raise

    def _initialize_index(self) -> BaseIndex[IndexDict]:
        """Initialize the index from the storage context."""
        try:
            # Load existing index with store_nodes_override=True
            index = load_index_from_storage(
                storage_context=self.storage_context,
                store_nodes_override=True,
                show_progress=self.show_progress,
                embed_model=self.embed_manager.embedding_model,
                transformations=self.transformations
            )
            logger.info("Successfully loaded existing index")
            return index
        except ValueError:
            # Create new index if none exists
            logger.info("Creating new vector store index")
            index = VectorStoreIndex.from_documents(
                [],
                storage_context=self.storage_context,
                store_nodes_override=True,
                show_progress=self.show_progress,
                embed_model=self.embed_manager.embedding_model,
                transformations=self.transformations
            )
            self._save_index(index)
            return index
        except Exception as e:
            logger.error(f"Failed to initialize index: {str(e)}")
            raise

    def ingest(self, documents: List[Document]) -> bool:
        """Ingest documents into the index"""
        try:            
            if not documents:
                raise ValueError("No documents provided for ingestion")

            with self._index_thread_lock:
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=self.storage_context,
                    store_nodes_override=True,
                    show_progress=self.show_progress,
                    embed_model=self.embed_manager,
                    transformations=self.transformations
                )
                self._save_index(self.index)
                
            logger.info("Successfully ingested documents into index")
            return True
        except Exception as e:
            logger.error(f"Failed to ingest documents: {str(e)}")
            raise

    def query(self, query_text: str, similarity_top_k: int = None) -> str:
        """Query the index with configurable number of similar documents"""
        if not self.index:
            raise ValueError("No index available. Please ingest documents first.")
        
        try:
            similarity_top_k = similarity_top_k or self.config.TOP_K
            retriever = self.vector_store_manager.get_retriever(
                self.index,
                similarity_top_k=similarity_top_k
            )
            query_engine = self.index.as_query_engine(
                retriever=retriever
            )
            response = query_engine.query(query_text)
            return str(response)
        except Exception as e:
            logger.error(f"Failed to query index: {str(e)}")
            raise

    def _save_index(self, index: BaseIndex[IndexDict]) -> None:
        """Save the index to persistent storage"""
        try:
            persist_dir = str(self.config.LOCAL_DATA_PATH)
            index.storage_context.persist(persist_dir=persist_dir)
            logger.info(f"Index persisted to {persist_dir}")
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise

    def delete(self, doc_id: str) -> None:
        """Delete a document from the index"""
        with self._index_thread_lock:
            if not self.index:
                raise ValueError("No index available")
            self.index.delete_ref_doc(doc_id, delete_from_docstore=True)
            self._save_index(self.index)

    def get_document_count(self) -> int:
        """Get the total number of documents in the store"""
        return len(self.storage_context.docstore.docs)

    def get_node_count(self) -> int:
        """Get the total number of nodes in the store"""
        return len([key for key in self.storage_context.docstore.docs.keys() 
                   if isinstance(key, str) and key.startswith('node')])