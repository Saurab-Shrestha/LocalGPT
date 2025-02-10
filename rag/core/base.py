from abc import ABC, abstractmethod
from ..config import Config
from typing import Any

class BaseManager(ABC):
    """Base class for all managers in the RAG system."""
    
    def __init__(self, config: Config):
        self.config = config
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the manager's resources."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        pass

class RAGAssistant:
    """Main RAG Assistant class that coordinates all components."""
    
    def __init__(self, config: Config):
        self.config = config
        self.managers = {}
    
    def register_manager(self, name: str, manager: BaseManager) -> None:
        """Register a manager with the assistant."""
        self.managers[name] = manager
    
    def initialize(self) -> None:
        """Initialize all registered managers."""
        for manager in self.managers.values():
            manager.initialize() 