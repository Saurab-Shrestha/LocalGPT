# Saves files and load data
import os
import tempfile
import logging
from typing import Optional, List
import pymupdf4llm
from llama_index.core import Document

logger = logging.getLogger(__name__)

class FileManager:
    @staticmethod
    def save_uploaded_file(uploaded_file) -> Optional[str]:
        try:
            if hasattr(uploaded_file, 'read'):
                file_extension = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    return tmp_file.name
            elif isinstance(uploaded_file, str):
                # If the uploaded_file is a string (e.g., file path)
                return uploaded_file
            else:
                print(f"Uploaded file is not a valid file object or path: {uploaded_file}")
                return None
        except Exception as e:
            print(f"Error saving uploaded file: {e}")
            return None

    @staticmethod
    def load_file(file) -> Optional[List[Document]]:
        if file:
            file_path = FileManager.save_uploaded_file(file)
            if file_path:
                try:
                    llama_reader = pymupdf4llm.LlamaMarkdownReader()
                    documents = llama_reader.load_data(file_path)
                    os.unlink(file_path)
                    return documents
                except Exception as e:
                    logger.error(f"Error loading file content: {e}")
                    os.unlink(file_path)
        return None
