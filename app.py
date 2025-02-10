import base64
import gradio as gr
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from rag.config import Config
from rag.manager.file_manager import FileManager
from rag.dependencies import injector, get_voice_chat_service, get_chat_service
from rag.services.chat_service import ChatService
from rag.services.voice_service import VoiceChatService
from rag.manager.llm_manager import LLMManager
from rag.manager.embed_manager import EmbeddingManager
from rag.manager.vector_store_manager import VectorStoreManager
from rag.manager.index_manager import IndexManager
from rag.manager.node_manager import NodeManager
from rag.manager.voice.text_to_voice_manager import TextToVoiceManager
from rag.manager.voice.voice_to_text_manager import VoiceToTextManager

import uvicorn
from fastapi import FastAPI
from rag.api.routes import app as api_app
from rag.constants import DEFAULT_HOST, DEFAULT_PORT
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioRAGChat:
    def __init__(self):
        # Get services from injector
        self.chat_service = injector.get(ChatService)
        self.voice_chat_service = injector.get(VoiceChatService)
        self.index_manager = injector.get(IndexManager)
        self.config = injector.get(Config)
        self.chat_history = []

    def upload_file(self, file: gr.File) -> str:
        """
        Handle file upload and indexing.
        
        Args:
            file: Gradio file object
            
        Returns:
            str: Status message
        """
        try:
            if file is None:
                return "No file uploaded."

            logger.info(f"Processing uploaded file: {file.name}")
            documents = FileManager.load_file(file.name)
            
            if not documents:
                return "Error: Could not process the file. Please check the file format."

            success = self.index_manager.ingest(documents)
            if success:
                doc_count = self.index_manager.get_document_count()
                return f"File uploaded and indexed successfully! Total documents: {doc_count}"
            else:
                return "Error: Failed to index the file."

        except Exception as e:
            logger.error(f"Error in upload_file: {str(e)}")
            return f"Error processing file: {str(e)}"

    def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Process user message and update chat history.
        
        Args:
            message: User input message
            history: Current chat history
            
        Returns:
            Tuple containing updated chat history
        """
        try:
            if not message.strip():
                return history, history

            self.chat_history = history or []
            response = self.chat_service.chat(message)
            self.chat_history.append((message, response))
            return self.chat_history, self.chat_history

        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            error_msg = "I apologize, but I encountered an error. Please try again."
            self.chat_history.append((message, error_msg))
            return self.chat_history, self.chat_history

    def voice_chat(self, audio: Optional[Tuple[int, np.ndarray]]) -> Tuple[Optional[str], List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Process voice input, transcribe, chat, and convert response to speech.
        
        Args:
            audio: Tuple containing sample rate and audio data
            
        Returns:
            Tuple containing audio output path and updated chat history
        """
        try:
            if audio is None:
                return None, self.chat_history, self.chat_history

            sample_rate, audio_data = audio
            audio_data = audio_data.astype(np.float32)
            
            transcription, response, audio_path = self.voice_chat_service.run_voice_chat()
            
            if transcription and response:
                self.chat_history.append((transcription, response))
                
                if audio_path:
                    return (
                        gr.Audio.update(value=audio_path, visible=True),
                        self.chat_history,
                        self.chat_history
                    )
                else:
                    logger.warning("Audio response not available")
                    return None, self.chat_history, self.chat_history

            return None, self.chat_history, self.chat_history

        except Exception as e:
            logger.error(f"Error in voice_chat: {str(e)}")
            error_msg = "I apologize, but I encountered an error processing the voice input."
            self.chat_history.append(("Voice Input", error_msg))
            return None, self.chat_history, self.chat_history

    def reset_chat(self) -> List[Tuple[str, str]]:
        """
        Reset the chat history and conversation context.
        
        Returns:
            Empty chat history
        """
        self.chat_history = []
        self.chat_service.reset_chat()
        return self.chat_history

    def launch(self):
        """Launch the Gradio app with a custom layout and functionality."""
        # Custom CSS for styling
        custom_css = """
        .logo { display: flex; background-color: #C7BAFF; height: 80px; border-radius: 8px; align-content: center; justify-content: center; align-items: center; }
        .logo img { height: 50px; }
        .footer { text-align: center; margin-top: 20px; font-size: 14px; display: flex; align-items: center; justify-content: center; }
        .footer-link { text-decoration: auto; color: var(--body-text-color); }
        .footer-link:hover { color: #C7BAFF; }
        .footer-logo { height: 20px; margin-left: 5px; }
        """

        # Load logo
        logo_path = Path(self.config.ASSETS_DIR) / "logo.png"
        if logo_path.exists():
            try:
                avatar_byte = logo_path.read_bytes()
                f_base64 = f"data:image/png;base64,{base64.b64encode(avatar_byte).decode('utf-8')}"
            except Exception as e:
                logger.error(f"Error loading logo: {str(e)}")
                f_base64 = ""
        else:
            logger.warning(f"Logo file not found at {logo_path}")
            f_base64 = ""

        with gr.Blocks(
            title="RAG Chat Application",
            theme=gr.themes.Soft(primary_hue="slate"),
            css=custom_css,
        ) as demo:
            gr.Markdown("# RAG Chat Application")

            # Header with logo
            with gr.Row():
                gr.HTML(f"<div class='logo'><img src={f_base64} alt='RAG Chat'></div>")

            # Upload Document Tab
            with gr.Tab("Upload Document"):
                file_upload = gr.File(
                    label="Upload Document",
                    file_types=[".txt", ".pdf", ".docx", ".json", ".md"]
                )
                upload_button = gr.Button("Upload and Index")
                upload_output = gr.Textbox(label="Upload Status")
                upload_button.click(
                    fn=self.upload_file,
                    inputs=file_upload,
                    outputs=upload_output
                )

            # Text Chat Tab
            with gr.Tab("Text Chat"):
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=400
                )
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Type your message here...",
                    show_label=True
                )
                clear = gr.Button("Clear Chat")
                
                msg.submit(
                    fn=self.chat,
                    inputs=[msg, chatbot],
                    outputs=[chatbot, chatbot]
                )
                clear.click(
                    fn=self.reset_chat,
                    outputs=chatbot
                )

            # Voice Chat Tab
            with gr.Tab("Voice Chat"):
                audio_input = gr.Audio(
                    label="Voice Input",
                    type="numpy",
                    sources=["microphone"],
                    streaming=False
                )
                voice_output = gr.Audio(
                    label="AI Response",
                    visible=False,
                    type="filepath"
                )
                voice_chatbot = gr.Chatbot(
                    label="Voice Chat History",
                    height=400
                )
                voice_clear = gr.Button("Clear Voice Chat")
                
                audio_input.stop_recording(
                    fn=self.voice_chat,
                    inputs=[audio_input],
                    outputs=[voice_output, voice_chatbot, voice_chatbot]
                )
                voice_clear.click(
                    fn=self.reset_chat,
                    outputs=voice_chatbot
                )

            # Footer
            with gr.Row():
                gr.HTML(
                    f"<div class='footer'><img class='footer-logo' src='{f_base64}' alt='Chat'></div>"
                )

        # Launch the interface
        demo.launch(
            server_name=self.config.HOST,
            server_port=self.config.PORT,
            share=self.config.SHARE
        )

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        Path.home() / ".rag-assistant" / "cache",
        Path.home() / ".rag-assistant" / "models",
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

def main():
    # Setup directories
    setup_directories()
    
    # Initialize config and bind to injector
    config = Config()
    injector.binder.bind(Config, to=config)
    
    # Bind all managers and services
    injector.binder.bind(LLMManager, to=LLMManager)
    injector.binder.bind(EmbeddingManager, to=EmbeddingManager)
    injector.binder.bind(VectorStoreManager, to=VectorStoreManager)
    injector.binder.bind(IndexManager, to=IndexManager)
    injector.binder.bind(NodeManager, to=NodeManager)
    injector.binder.bind(TextToVoiceManager, to=TextToVoiceManager)
    injector.binder.bind(VoiceToTextManager, to=VoiceToTextManager)
    
    # Create FastAPI app
    app = FastAPI(title="RAG Assistant")
    app.mount("/api", api_app)

    # Create Gradio app
    rag_chat = GradioRAGChat()
    gradio_app = gr.routes.App.create_app(rag_chat.launch())
    app.mount("/ui", gradio_app)

    logger.info(f"API: {DEFAULT_HOST}:{DEFAULT_PORT}")
    logger.info(f"Gradio UI: {DEFAULT_HOST}:{DEFAULT_PORT}/ui")
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)

if __name__ == "__main__":
    main()