import torch
import logging
from injector import inject, singleton
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

logger = logging.getLogger(__name__)

@singleton
class VoiceToTextManager:
    @inject
    def __init__(self, config) -> None:
        try:
            self.processor, self.model = self.load_model(config.AUDIO_MODEL_PATH)
        except Exception as e:
            logger.debug(f"Could not load the model! {e}")
            self.processor, self.model = None, None

    @staticmethod
    def load_model(model_path):
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
        return processor, model

    def transcribe_audio(self, audio_input):
        if self.processor is None or self.model is None:
            logger.error("Model not loaded. Cannot transcribe audio.")
            return None

        input_features = self.processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features
        self.model.eval()
        with torch.no_grad():
            predicted_ids = self.model.generate(inputs=input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription