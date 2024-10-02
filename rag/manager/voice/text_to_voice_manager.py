import logging
from injector import inject, singleton
import torch
from transformers import AutoTokenizer, AutoModelForTextToSpeech

logger = logging.getLogger(__name__)

@singleton
class TextToVoiceManager:
    @inject
    def __init__(self, config) -> None:
        try:
            self.tokenizer, self.model = self.load_model(config.TTS_MODEL_PATH)
        except Exception as e:
            logger.debug(f"Could not load the model! {e}")
            self.tokenizer, self.model = None, None

    @staticmethod
    def load_model(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTextToSpeech.from_pretrained(model_path)
        return tokenizer, model

    def text_to_speech(self, text):
        if self.tokenizer is None or self.model is None:
            logger.error("Model not loaded. Cannot convert text to speech.")
            return None

        inputs = self.tokenizer(text, return_tensors="pt")
        self.model.eval()
        with torch.no_grad():
            speech = self.model.generate_speech(inputs["input_ids"], self.tokenizer)

        return speech.numpy()