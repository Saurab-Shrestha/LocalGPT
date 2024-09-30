import logging
from injector import inject, singleton
from llama_index.core.storage.docstore import BaseDocumentStore, SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.index_store.types import BaseIndexStore

logger = logging.getLogger(__name__)

@singleton
class NodeManager:
    index_store: BaseIndexStore
    doc_store: BaseDocumentStore
    
    @inject
    def __init__(self, config) -> None:
        try:
            self.index_store = SimpleIndexStore.from_persist_dir(
                persist_dir=str(config.LOCAL_DATA_PATH)
            )
        except FileNotFoundError:
            logger.debug("Local index store not found, creating a new one")
            self.index_store = SimpleIndexStore()
        try:
            self.doc_store = SimpleDocumentStore.from_persist_dir(
                persist_dir=str(config.LOCAL_DATA_PATH)
            )
        except FileNotFoundError:
            logger.debug("Local document store not found, creating a new one")
            self.doc_store = SimpleDocumentStore()