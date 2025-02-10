import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional

from ..core.base import RAGAssistant
from ..config import Config
from rag.services.voice_service import VoiceChatService
from rag.dependencies import get_voice_chat_service, get_chat_service, get_index_manager
from rag.services.chat_service import ChatService
from rag.manager.index_manager import IndexManager
from rag.api.models import ChatRequest, TextToSpeechRequest, IngestRequest, ChatResponse, IngestResponse

logger = logging.getLogger(__name__)
app = APIRouter()

class TextToSpeechRequest(BaseModel):
    text: str
    output_path: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    try:
        response = chat_service.chat(request.message)
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def text_to_speech(
    request: TextToSpeechRequest,
    voice_service: VoiceChatService = Depends(get_voice_chat_service)
):
    try:
        audio = voice_service.text_to_speech(request.text)
        if audio is None:
            raise HTTPException(status_code=500, detail="Failed to generate speech")
            
        output_path = request.output_path or "output.wav"
        voice_service.save_audio(audio, filename=output_path)
        return {"audio_path": output_path}
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    files: list[UploadFile] = File(...),
    index_manager: IndexManager = Depends(get_index_manager)
):
    try:
        documents = []
        for file in files:
            content = await file.read()
            from llama_index import Document
            documents.append(Document(text=content.decode()))
        
        success = index_manager.ingest(documents)
        doc_count = index_manager.get_document_count() if success else 0
        
        return IngestResponse(
            success=success,
            document_count=doc_count
        )
    except Exception as e:
        logger.error(f"Ingest error: {str(e)}")
        return IngestResponse(
            success=False,
            error=str(e)
        ) 