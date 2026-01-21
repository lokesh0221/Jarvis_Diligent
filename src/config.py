from __future__ import annotations

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "jarvis-assistant")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "")
    pinecone_region: str = os.getenv("PINECONE_REGION", "")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "default")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.1")
    top_k: int = int(os.getenv("TOP_K", "4"))


settings = Settings()
