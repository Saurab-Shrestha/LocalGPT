from injector import inject, singleton
from rag.config import Config
import logging

@singleton
class BaseManager:
    @inject
    def __init__(self, config: Config):
        self.config = config

    def get_config(self) -> Config:
        return self.config  
    
    def get_logger(self) -> logging.Logger:
        return logging.getLogger(__name__)
