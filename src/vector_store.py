from __future__ import annotations

from typing import Iterable, List
from uuid import uuid4

from pinecone import Pinecone, ServerlessSpec

from src.config import settings


class PineconeVectorStore:
    def __init__(self, dimension: int) -> None:
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is not set.")

        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index
        self._ensure_index(dimension)
        self.index = self.pc.Index(self.index_name)

    def _ensure_index(self, dimension: int) -> None:
        existing = self.pc.list_indexes().names()
        if self.index_name in existing:
            return

        if not settings.pinecone_cloud or not settings.pinecone_region:
            raise ValueError(
                "PINECONE_CLOUD and PINECONE_REGION are required to create a serverless index. "
                "Create the index manually or set these environment variables."
            )

        self.pc.create_index(
            name=self.index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )

    def upsert_vectors(
        self,
        vectors: Iterable[list[float]],
        metadatas: Iterable[dict] | None = None,
        ids: Iterable[str] | None = None,
        namespace: str | None = None,
    ) -> None:
        namespace = namespace or settings.pinecone_namespace
        metadata_list = list(metadatas) if metadatas is not None else []
        id_list = list(ids) if ids is not None else []

        payload = []
        for idx, values in enumerate(vectors):
            metadata = metadata_list[idx] if idx < len(metadata_list) else {}
            record_id = id_list[idx] if idx < len(id_list) else str(uuid4())
            payload.append({"id": record_id, "values": values, "metadata": metadata})

        if payload:
            self.index.upsert(vectors=payload, namespace=namespace)

    def query(self, vector: list[float], top_k: int, namespace: str | None = None) -> List[dict]:
        namespace = namespace or settings.pinecone_namespace
        result = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
        )
        return list(result.get("matches", []))
