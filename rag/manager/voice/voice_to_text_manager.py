import logging
import whisper
from injector import inject, singleton
import numpy as np
from scipy.io.wavfile import write
from rag.config import Config
from rag.manager.base_manager import BaseManager

logger = logging.getLogger(__name__)

@singleton
class VoiceToTextManager(BaseManager):
    @inject
    def __init__(self, config: Config):
        super().__init__(config)
        try:
            self.model = self.load_model(config.AUDIO_MODEL_PATH)
        except Exception as e:
            logger.error(f"Could not load the Whisper model! {e}")
            self.model = None

    @staticmethod
    def load_model(model_path):
        # If model_path is provided in config, use it, otherwise use 'base'
        model_size = model_path if model_path else "base"
        return whisper.load_model(model_size)

    def transcribe_audio(self, audio_input):
        if self.model is None:
            logger.error("Model not loaded. Cannot transcribe audio.")
            return None

        try:
            # Save temporary file for Whisper
            temp_file = "temp_recording.wav"
            write(temp_file, 16000, audio_input)
            
            # Transcribe using Whisper
            result = self.model.transcribe(temp_file)
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            return None