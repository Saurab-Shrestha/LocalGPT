from typing import List
from injector import inject, singleton
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core import get_response_synthesizer
from llama_index.core.schema import NodeWithScore

from rag.manager.llm_manager import LLMManager
from rag.manager.embed_manager import EmbeddingManager
from rag.manager.vector_store_manager import VectorStoreManager
from rag.manager.node_manager import NodeManager
from llama_index.core.storage import StorageContext
from rag.config import Config

@singleton
class ChatService:
    config: Config
    @inject
    def __init__(self,
        config: Config,
        llm_component: LLMManager,
        vector_store_component: VectorStoreManager,
        embedding_component: EmbeddingManager,
        node_store_component: NodeManager
        ):
        self.config = config
        self.llm = llm_component
        self.embedding_component = embedding_component
        self.vector_store_component = vector_store_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store_component.vector_store,
            storage_context=self.storage_context,
            llm=llm_component.llm,
            embed_model=embedding_component.embedding_model,
            show_progress=True,
        )

    def _setup_chat_engine(self, system_prompt) -> ContextChatEngine:
        vector_index_retriever = self.vector_store_component.get_retriever(
            index=self.index,
            similarity_top_k=self.config.SIMILARITY_TOP_K,
        )
        node_postprocessors = [
            MetadataReplacementPostProcessor(target_metadata_key="window"),
            SimilarityPostprocessor(
                similarity_cutoff=self.config.SIMILARITY_CUTTOFF,
            ),
        ]
        
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            llm=self.llm.llm,
        )
        
        return ContextChatEngine.from_defaults(
            system_prompt=system_prompt,
            retriever=vector_index_retriever,
            llm=self.llm.llm,
            node_postprocessors=node_postprocessors,
            response_synthesizer=response_synthesizer,
            verbose=True,
        )
    
    def chat(self, message: str) -> str:
        system_prompt = """
            You are an AI assistant designed to provide accurate and concise answers based on retrieved bank documents.
            """
        chat_engine = self._setup_chat_engine(
            system_prompt=system_prompt,
        )
        try:
            # Retrieve nodes
            retriever = chat_engine._retriever
            retrieved_nodes = retriever.retrieve(message)
            
            print(f"Retrieved {len(retrieved_nodes)} nodes")
            for i, node in enumerate(retrieved_nodes):
                print(f"Node {i + 1}:")
                print(f"  Content: {node.node.get_content()[:100]}...")
                print(f"  Score: {node.score}")

            # Generate response
            wrapped_response = chat_engine.chat(message)
            
            print(f"Raw response: {wrapped_response}")
            print(f"Response content: {wrapped_response.response}")
            
            if not wrapped_response.response:
                print("Empty response received from ContextChatEngine")
                return "I apologize, but I couldn't generate a response based on the retrieved information. This might be due to insufficient or irrelevant context. Could you please rephrase your question or ask about a different topic?"
            
            return wrapped_response.response
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return "I apologize, but an error occurred while processing your request. Please try again or contact support if the issue persists."