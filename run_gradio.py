from rag.dependencies import configure_dependencies, injector
from app import GradioRAGChat, setup_directories
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Setup directories
    setup_directories()
    
    # Configure all dependencies
    configure_dependencies()
    
    # Create and launch Gradio interface
    gradio_app = GradioRAGChat()
    gradio_app.launch()

if __name__ == "__main__":
    main() 