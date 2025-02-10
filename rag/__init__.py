"""
RAG-based AI Assistant package.
"""

__version__ = "0.1.0"

from .core.base import RAGAssistant
from .manager.voice.text_to_voice_manager import TextToVoiceManager
from .manager.voice.voice_to_text_manager import VoiceToTextManager

__all__ = ["RAGAssistant", "TextToVoiceManager", "VoiceToTextManager"]
