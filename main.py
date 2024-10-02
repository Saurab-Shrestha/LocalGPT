from injector import Module, Binder, singleton, provider
from rag.manager.llm_manager import LLMManager
from rag.manager.node_manager import NodeManager
from rag.manager.index_manager import IndexManager
from rag.manager.embed_manager import EmbeddingManager
from rag.manager.vector_store_manager import VectorStoreManager
from rag.manager.voice.voice_to_text_manager import VoiceToTextManager
from rag.manager.voice.text_to_voice_manager import TextToVoiceManager
from rag.services.voice_service import VoiceChatService
from rag.services.chat_service import ChatService

from rag.config import Config
from llama_index.core.storage import StorageContext

class ChatModule(Module):
    @singleton
    @provider
    def provide_config(self) -> Config:
        return Config() 

    @singleton
    @provider
    def provide_llm_manager(self, config: Config) -> LLMManager:
        return LLMManager(config)

    @singleton
    @provider
    def provide_embedding_manager(self, config: Config) -> EmbeddingManager:
        return EmbeddingManager(config)

    @singleton
    @provider
    def provide_vector_store_manager(self, config: Config) -> VectorStoreManager:
        return VectorStoreManager(config)

    @singleton
    @provider
    def provide_node_manager(self, config: Config) -> NodeManager:
        return NodeManager(config)

    @singleton
    @provider
    def provide_storage_context(self, vector_store_manager: VectorStoreManager, node_manager: NodeManager) -> StorageContext:
        return StorageContext.from_defaults(
            vector_store=vector_store_manager.vector_store,
            docstore=node_manager.doc_store,
            index_store=node_manager.index_store,
        )

    @singleton
    @provider
    def provide_index_manager(self, config: Config, storage_context: StorageContext, embedding_manager: EmbeddingManager) -> IndexManager:
        return IndexManager(
            storage_context=storage_context,
            embed_model=embedding_manager.embedding_model,
            local_data_path=config.LOCAL_DATA_PATH,
            show_progress=config.SHOW_PROGRESS,
        )
    @singleton
    @provider
    def provide_voice_chat_service(self, config: Config, chat_service: ChatService,
                                   voice_to_text: VoiceToTextManager, 
                                   text_to_voice: TextToVoiceManager) -> VoiceChatService:
        return VoiceChatService(config, chat_service, voice_to_text, text_to_voice)
    
    @singleton
    @provider
    def provide_voice_manager(self, config: Config) -> VoiceToTextManager:
        return VoiceToTextManager(config)

    @singleton
    @provider
    def provide_text_to_voice_manager(self, config: Config) -> TextToVoiceManager:
        return TextToVoiceManager(config)
    
    def configure(self, binder: Binder) -> None:
        # The bindings are now handled by the provider methods
        pass