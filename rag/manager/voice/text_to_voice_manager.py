import logging
from injector import inject, singleton
from TTS.api import TTS
from rag.config import Config
from rag.manager.base_manager import BaseManager

logger = logging.getLogger(__name__)

@singleton
class TextToVoiceManager(BaseManager):
    @inject
    def __init__(self, config: Config):
        super().__init__(config)
        try:
            self.model = self.load_model(config.TTS_MODEL_PATH)
        except Exception as e:
            logger.error(f"Could not load the TTS model! {e}")
            self.model = None

    @staticmethod
    def load_model(model_path):
        # If model_path is provided in config, use it, otherwise use default
        if model_path:
            return TTS(model_path)
        return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

    def text_to_speech(self, text):
        if self.model is None:
            logger.error("Model not loaded. Cannot convert text to speech.")
            return None
        try:
            wav = self.model.tts(text)
            return wav
        except Exception as e:
            logger.error(f"Error in text_to_speech: {str(e)}")
            return None