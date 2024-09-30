import logging
from injector import inject, singleton
from llama_index.llms.ollama import Ollama

logger = logging.getLogger(__name__)

@singleton
class LLMManager:

    @inject
    def __init__(self, config) -> None:
        try:
            self.llm = Ollama(
                model=config.LLM_MODEL, 
                temperature=config.TEMPERATURE, 
                request_timeout=config.TIMEOUT
            )
        except Exception as e:
            logger.debug(f"Could not load the model! {e}")
            self.llm = None
    