from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from src.config import settings
from src.embeddings import Embeddings
from src.vector_store import PineconeVectorStore


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        return [text]
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if overlap > 0 else end
    return chunks


def iter_text_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return

    for ext in (".txt", ".md"):
        yield from path.rglob(f"*{ext}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone.")
    parser.add_argument("--path", default="data", help="File or folder with .txt/.md documents")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--namespace", default=settings.pinecone_namespace)
    args = parser.parse_args()

    base_path = Path(args.path)
    if not base_path.exists():
        raise FileNotFoundError(f"Path not found: {base_path}")

    embedder = Embeddings()
    store = PineconeVectorStore(embedder.dimension)

    texts: List[str] = []
    metadatas: List[dict] = []

    for file_path in iter_text_files(base_path):
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        for chunk in chunk_text(content, args.chunk_size, args.overlap):
            texts.append(chunk)
            metadatas.append({"source": str(file_path), "text": chunk})

    if not texts:
        print("No documents found to ingest.")
        return

    embeddings = embedder.embed_texts(texts)
    store.upsert_vectors(embeddings, metadatas=metadatas, namespace=args.namespace)
    print(f"Ingested {len(texts)} chunks into namespace '{args.namespace}'.")


if __name__ == "__main__":
    main()
