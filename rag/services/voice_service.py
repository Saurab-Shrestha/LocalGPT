import tempfile
import logging
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from injector import inject, singleton
from pathlib import Path
from rag.config import Config
from rag.services.chat_service import ChatService
from rag.manager.voice.text_to_voice_manager import TextToVoiceManager
from rag.manager.voice.voice_to_text_manager import VoiceToTextManager

logger = logging.getLogger(__name__)

@singleton
class VoiceChatService:
    @inject
    def __init__(
        self,
        config: Config,
        chat_service: ChatService,
        text_to_voice: TextToVoiceManager,
        voice_to_text: VoiceToTextManager
    ):
        self.config = config
        self.chat_service = chat_service
        self.text_to_voice = text_to_voice
        self.voice_to_text = voice_to_text

    def record_audio(self, duration: int = 5, sample_rate: int = 16000) -> np.ndarray:
        """Record audio from microphone."""
        logger.info(f"Recording audio for {duration} seconds...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        return audio.flatten()

    def transcribe_audio(self, audio: np.ndarray) -> str | None:
        """Transcribe audio using Whisper."""
        return self.voice_to_text.transcribe_audio(audio)

    def text_to_speech(self, text: str) -> bytes | None:
        """Convert text to speech."""
        return self.text_to_voice.text_to_speech(text)

    def save_audio(self, audio: bytes | np.ndarray, filename: str = "output.wav", sample_rate: int = 16000) -> None:
        """Save audio data to a file."""
        try:
            if isinstance(audio, bytes):
                write(filename, sample_rate, np.frombuffer(audio, dtype=np.float32))
            else:
                write(filename, sample_rate, audio)
        except Exception as e:
            logger.error(f"Error saving audio: {str(e)}")
            raise

    def run_voice_chat(self) -> tuple[str | None, str, str | None]:
        """
        Run a complete voice chat interaction.
        Returns:
            Tuple of (transcription, response, audio_path)
        """
        try:
            audio = self.record_audio(duration=self.config.RECORDING_DURATION)
            
            transcription = self.voice_to_text.transcribe_audio(audio)
            if not transcription:
                logger.error("Failed to transcribe audio")
                return None, "I couldn't understand the audio. Could you please try again?", None

            logger.info(f"Transcription: {transcription}")
            
            response = self.chat_service.chat(transcription)
            if not response:
                logger.error("Failed to generate chat response")
                return transcription, "I couldn't generate a response. Please try asking differently.", None

            logger.info(f"Chat response: {response}")
            
            # Generate speech from response
            audio_data = self.text_to_voice.text_to_speech(response)
            if audio_data is None:
                logger.error("Failed to convert text to speech")
                return transcription, response, None

            # Create temporary file for audio
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    self.text_to_voice.save(temp_file.name, audio_data)
                    logger.info(f"Saved audio response to {temp_file.name}")
                    return transcription, response, temp_file.name
            except Exception as e:
                logger.error(f"Error saving audio file: {str(e)}")
                return transcription, response, None

        except Exception as e:
            logger.error(f"Error in voice chat: {str(e)}")
            return None, f"An error occurred: {str(e)}", None

    def streaming_voice_chat(self):
        # This method could be implemented for a streaming version of the voice chat
        # It would involve continuous audio recording, real-time transcription,
        # and immediate responses
        pass