import os
import logging
import threading
from typing import List
from pathlib import Path
from injector import inject, singleton
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex, load_index_from_storage, Document
from llama_index.core.data_structs import IndexDict
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import TransformComponent

logger = logging.getLogger(__name__)

@singleton
class IndexManager:
    @inject
    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: OllamaEmbedding,
        local_data_path: str,
        show_progress: bool,
        transformations: List[TransformComponent] = None
    ):
        self.storage_context = storage_context
        self.embed_model = embed_model
        self.local_data_path = Path(local_data_path)
        self.show_progress = show_progress
        self.transformations = transformations or [
            MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True),
        ]
        if self.embed_model not in self.transformations:
            self.transformations.append(self.embed_model)
        
        self._index_thread_lock = threading.Lock()
        self._index = self._initialize_index()
        
        if not self.local_data_path.exists():
            self.local_data_path.mkdir(parents=True, exist_ok=True)

    def _initialize_index(self) -> BaseIndex[IndexDict]:
        try:
            return load_index_from_storage(
                storage_context=self.storage_context,
                store_nodes_override=True,
                show_progress=self.show_progress,
                embed_model=self.embed_model,
                transformations=self.transformations,
            )
        except ValueError:
            logger.info("Creating a new vector store index")
            index = VectorStoreIndex.from_documents(
                [],
                storage_context=self.storage_context,
                store_nodes_override=True,
                show_progress=self.show_progress,
                embed_model=self.embed_model,
                transformations=self.transformations,
            )
            self._save_index(index)
            return index

    def _save_index(self, index: BaseIndex[IndexDict]):
        index.storage_context.persist(persist_dir=str(self.local_data_path))
        logger.info(f"Index persisted to {self.local_data_path}")

    def ingest(self, documents: List[Document]) -> List[Document]:
        # Assuming you have an IngestionHelper class similar to the one in the provided code
        # from private_gpt.components.ingest.ingest_helper import IngestionHelper
        
        # logger.info(f"Ingesting file_name={file_name}")
        # documents = IngestionHelper.transform_file_into_documents(file_name, file_data)
        return self.update_index(documents)

    def update_index(self, documents: List[Document]) -> List[Document]:
        if not documents:
            logger.warning("No documents provided for indexing.")
            return []
        
        with self._index_thread_lock:
            for document in documents:
                self._index.insert(document, show_progress=self.show_progress)
            
            self._save_index(self._index)
        
        return documents

    def delete(self, doc_id: str) -> None:
        with self._index_thread_lock:
            self._index.delete_ref_doc(doc_id, delete_from_docstore=True)
            self._save_index(self._index)

    def get_document_count(self) -> int:
        return len(self.storage_context.docstore.docs)

    def get_node_count(self) -> int:
        return len([key for key in self.storage_context.docstore.docs.keys() if isinstance(key, str) and key.startswith('node')])