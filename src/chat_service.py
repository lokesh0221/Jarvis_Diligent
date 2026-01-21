from __future__ import annotations

from typing import List, Tuple
from datetime import datetime, timezone

from src.config import settings
from src.embeddings import Embeddings
from src.llm_client import OllamaClient
from src.vector_store import PineconeVectorStore


class ChatService:
    def __init__(self) -> None:
        self.embedder = Embeddings()
        self.vector_store = PineconeVectorStore(self.embedder.dimension)
        self.llm = OllamaClient()

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        namespace: str | None = None,
        system_prompt: str | None = None,
    ) -> Tuple[str, List[dict]]:
        query_vector = self.embedder.embed_texts([question])[0]
        matches = self.vector_store.query(query_vector, top_k or settings.top_k, namespace)

        memory_matches = self.vector_store.query(
            query_vector,
            top_k=min(5, top_k or settings.top_k),
            namespace=settings.chat_namespace,
        )

        context_blocks = []
        for match in matches:
            metadata = match.get("metadata", {})
            text = metadata.get("text", "")
            source = metadata.get("source", "")
            if text:
                header = f"Source: {source}" if source else "Source: unknown"
                context_blocks.append(f"{header}\n{text}")

        memory_blocks = []
        for match in memory_matches:
            metadata = match.get("metadata", {})
            role = metadata.get("role", "unknown")
            text = metadata.get("text", "")
            if text:
                memory_blocks.append(f"{role}: {text}")

        context = "\n\n".join(context_blocks)
        memory_context = "\n".join(memory_blocks)
        prompt = (
            "You are a helpful enterprise assistant. Use the provided context to answer the question. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nConversation Memory:\n{memory_context}\n\nQuestion:\n{question}"
        )

        answer = self.llm.generate(prompt, system_prompt=system_prompt)
        return answer, matches

    def store_message(self, role: str, content: str, namespace: str | None = None) -> None:
        if not content.strip():
            return
        vector = self.embedder.embed_texts([content])[0]
        metadata = {
            "role": role,
            "text": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.vector_store.upsert_vectors(
            [vector],
            metadatas=[metadata],
            namespace=namespace or settings.chat_namespace,
        )
