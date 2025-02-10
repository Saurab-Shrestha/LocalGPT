from injector import Injector
from fastapi import Depends
from rag.config import Config
from rag.services.voice_service import VoiceChatService
from rag.services.chat_service import ChatService
from rag.manager.voice.text_to_voice_manager import TextToVoiceManager
from rag.manager.voice.voice_to_text_manager import VoiceToTextManager
from rag.manager.base_manager import BaseManager
from rag.manager.llm_manager import LLMManager
from rag.manager.embed_manager import EmbeddingManager
from rag.manager.vector_store_manager import VectorStoreManager
from rag.manager.index_manager import IndexManager
from rag.manager.node_manager import NodeManager

# Create injector instance
injector = Injector()

def configure_dependencies():
    # Bind base dependencies
    injector.binder.bind(Config, to=Config())
    injector.binder.bind(BaseManager, to=BaseManager)
    
    # Bind managers in dependency order
    injector.binder.bind(NodeManager, to=NodeManager)
    injector.binder.bind(VectorStoreManager, to=VectorStoreManager)
    injector.binder.bind(IndexManager, to=IndexManager)
    injector.binder.bind(LLMManager, to=LLMManager)
    injector.binder.bind(EmbeddingManager, to=EmbeddingManager)
    injector.binder.bind(TextToVoiceManager, to=TextToVoiceManager)
    injector.binder.bind(VoiceToTextManager, to=VoiceToTextManager)
    
    # Bind services
    injector.binder.bind(ChatService, to=ChatService)
    injector.binder.bind(VoiceChatService, to=VoiceChatService)

# Configure dependencies
configure_dependencies()

def get_voice_chat_service() -> VoiceChatService:
    """FastAPI dependency that provides a VoiceChatService instance."""
    return injector.get(VoiceChatService)

def get_chat_service() -> ChatService:
    """FastAPI dependency that provides a ChatService instance."""
    return injector.get(ChatService)

def get_index_manager() -> IndexManager:
    """FastAPI dependency that provides an IndexManager instance."""
    return injector.get(IndexManager) 