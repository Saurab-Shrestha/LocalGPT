import logging
from injector import inject, singleton
from llama_index.llms.ollama import Ollama
from rag.config import Config
from rag.manager.base_manager import BaseManager

logger = logging.getLogger(__name__)

@singleton
class LLMManager(BaseManager):
    @inject
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        try:
            self.llm = Ollama(
                model=config.LLM_MODEL, 
                temperature=config.TEMPERATURE, 
                request_timeout=config.TIMEOUT
            )
        except Exception as e:
            logger.debug(f"Could not load the model! {e}")
            self.llm = None
    