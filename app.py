import base64
import gradio as gr
from pathlib import Path
from rag.config import Config
from rag.manager.file_manager import FileManager

from injector import Injector
from rag.services.chat_service import ChatService
from rag.services.voice_service import VoiceChatService
from rag.manager.llm_manager import LLMManager
from rag.manager.embed_manager import EmbeddingManager
from rag.manager.vector_store_manager import VectorStoreManager
from rag.manager.index_manager import IndexManager
from rag.manager.node_manager import NodeManager
from main import ChatModule

# Injector configuration to initialize dependencies
injector = Injector([ChatModule()])

class GradioRAGChat:
    def __init__(self):
        # Inject necessary services
        self.chat_service = injector.get(ChatService)
        self.voice_chat_service = injector.get(VoiceChatService)
        self.index_manager = injector.get(IndexManager)
        self.chat_history = []

    def upload_file(self, file):
        """Handle file upload and indexing."""
        if file is not None:
            documents = FileManager.load_file(file.name)
            if documents:
                self.index_manager.ingest(documents)
                return "File uploaded and indexed successfully!"
            else:
                return "Error processing the file."
        return "No file uploaded."

    def chat(self, message, history):
        """Process user message and update chat history."""
        self.chat_history = history or []
        response = self.chat_service.chat(message)
        self.chat_history.append((message, response))
        return self.chat_history, self.chat_history

    def voice_chat(self, audio):
        """Process voice input, transcribe, chat, and convert response to speech."""
        if audio is not None:
            transcription, response = self.voice_chat_service.run_voice_chat()
            if transcription and response:
                self.chat_history.append((transcription, response))
                return gr.Audio(value="output.wav", visible=True), self.chat_history, self.chat_history
        return None, self.chat_history, self.chat_history

    def reset_chat(self):
        """Reset the chat history and conversation context."""
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

        # Load footer logo as base64
        AVATAR_BOT = Path(r"C:\Users\shres\Desktop\New folder (2)\baseline\assets\logo.png")  # Replace with your path
        avatar_byte = AVATAR_BOT.read_bytes()
        f_base64 = f"data:image/png;base64,{base64.b64encode(avatar_byte).decode('utf-8')}"

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
                file_upload = gr.File(label="Upload PDF Document")
                upload_button = gr.Button("Upload and Index")
                upload_output = gr.Textbox(label="Upload Status")
                upload_button.click(self.upload_file, inputs=file_upload, outputs=upload_output)

            # Text Chat Tab
            with gr.Tab("Text Chat"):
                chatbot = gr.Chatbot(label="Chat History")
                msg = gr.Textbox(label="Message")
                clear = gr.Button("Clear")
                msg.submit(self.chat, inputs=[msg, chatbot], outputs=[chatbot, chatbot])
                clear.click(self.reset_chat, outputs=chatbot)

            # Voice Chat Tab
            with gr.Tab("Voice Chat"):
                audio_input = gr.Audio(source="microphone", type="filepath")
                voice_output = gr.Audio(label="AI Response", visible=False)
                voice_chatbot = gr.Chatbot(label="Voice Chat History")
                voice_clear = gr.Button("Clear Voice Chat")
                audio_input.stop_recording(self.voice_chat, inputs=[audio_input], outputs=[voice_output, voice_chatbot, voice_chatbot])
                voice_clear.click(self.reset_chat, outputs=voice_chatbot)

            # Footer with logo
            with gr.Row():
                gr.HTML(
                    f"<div class='footer'><img class='footer-logo' src='{f_base64}' alt='Chat'></div>"
                )

        # Launch the Gradio interface
        demo.launch()


if __name__ == "__main__":
    app = GradioRAGChat()
    app.launch()