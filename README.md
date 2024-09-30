# LocalGPT

LocalGPT is a Retrieval-Augmented Generation (RAG) system built using **LlamaIndex** for managing documents, **Ollama** for language model inference, and **Qdrant** for vector-based search and retrieval. This project enables efficient, context-aware Q&A from your local document files using advanced LLMs and vector databases.

## Table of Contents
- [LocalGPT](#localgpt)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Configuration](#configuration)

## Features

- **LlamaIndex Integration**: Enables document ingestion, chunking, and semantic search for relevant information.
- **Ollama for LLM Inference**: Use locally hosted Llama models through Ollama API for language generation tasks.
- **Qdrant Integration**: Fast and efficient vector-based search using Qdrant for document indexing and retrieval.
- **Gradio Interface**: Simple web interface for uploading documents, interacting with the chatbot, and retrieving answers from your knowledge base.

## Requirements

- Docker (if using Docker-based installation)
- Python 3.8 or later (for manual setup)
- **Qdrant**: Vector database, preferably running on localhost.
- **Ollama**: For local Llama-based models.
- **Gradio**: Web UI for interacting with the chatbot.

## Installation

1. **Clone the Repository**:
```bash
   git clone https://github.com/Saurab-Shrestha/LocalGPT.git
   cd localgpt
```

2. **Create a Virtual Environment**:
```bash
   python3 -m venv env
   source env/bin/activate
```

3. **Install Python Dependencies**:
```bash
   pip install -r requirements.txt
```

4. **Install and Run Qdrant**:
```
docker run -p 6333:6333 qdrant/qdrant
```

5. **Install and Configure Ollama**:
Install and Configure Ollama: Follow Ollama installation instructions for your operating system.

6. **Run the Application**:
```
gradio app.py
```


## Usage
### Configuration
Before running the app, you need to configure the settings for the LLM, Qdrant and the other services.
1. **Edit the Configuration File**: Update the `config.py` file to specify the correct host, port and model setting for you setup.
2. **Model Configuration**: Make sure you have your LLM model and embedding running in the Ollama, configured as per you needs.