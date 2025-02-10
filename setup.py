from setuptools import setup, find_packages

setup(
    name="rag-assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "pyaudio",
        "pydub",
        "pydub-time",
    ],
    author="Saurab Shrestha",
    description="RAG-based AI Assistant",
    python_requires=">=3.11",
) 