from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None

class TextToSpeechRequest(BaseModel):
    text: str
    output_path: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    source_documents: Optional[list[str]] = None

class IngestRequest(BaseModel):
    files: list[bytes]
    metadata: Optional[dict] = None

class IngestResponse(BaseModel):
    success: bool
    document_count: Optional[int] = None
    error: Optional[str] = None 