import logging
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from injector import inject, singleton
from rag.config import Config
from rag.services.chat_service import ChatService
from rag.manager.voice.voice_to_text_manager import VoiceToTextManager
from rag.manager.voice.text_to_voice_manager import TextToVoiceManager

logger = logging.getLogger(__name__)

@singleton
class VoiceChatService:
    @inject
    def __init__(self, 
                 config: Config, 
                 chat_service: ChatService,
                 voice_to_text: VoiceToTextManager, 
                 text_to_voice: TextToVoiceManager):
        self.config = config
        self.chat_service = chat_service
        self.voice_to_text = voice_to_text
        self.text_to_voice = text_to_voice

    def record_audio(self, duration=5, sample_rate=16000):
        logger.info(f"Recording audio for {duration} seconds...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        return audio.flatten()

    def transcribe_audio(self, audio):
        return self.voice_to_text.transcribe_audio(audio)

    def chat(self, transcription):
        return self.chat_service.chat(transcription)

    def text_to_speech(self, text):
        return self.text_to_voice.text_to_speech(text)

    def save_audio(self, audio, filename="output.wav", sample_rate=16000):
        write(filename, sample_rate, audio)

    def run_voice_chat(self):
        try:
            # Record audio
            audio = self.record_audio(duration=self.config.RECORDING_DURATION)
            
            # Transcribe audio to text
            transcription = self.transcribe_audio(audio)
            if not transcription:
                logger.error("Failed to transcribe audio.")
                return None, "I'm sorry, I couldn't understand the audio. Could you please try again?"

            logger.info(f"Transcription: {transcription}")
            
            # Get chat response
            response = self.chat(transcription)
            if not response:
                logger.error("Failed to generate chat response.")
                return transcription, "I apologize, but I couldn't generate a response. Please try asking in a different way."

            logger.info(f"Chat response: {response}")
            
            # Convert response to speech
            speech = self.text_to_speech(response)
            if speech is None:
                logger.error("Failed to convert text to speech.")
                return transcription, response

            # Save the response audio
            self.save_audio(speech)
            
            return transcription, response

        except Exception as e:
            logger.error(f"An error occurred during voice chat: {str(e)}")
            return None, "I'm sorry, an error occurred. Please try again or contact support if the issue persists."

    def streaming_voice_chat(self):
        # This method could be implemented for a streaming version of the voice chat
        # It would involve continuous audio recording, real-time transcription,
        # and immediate responses
        pass