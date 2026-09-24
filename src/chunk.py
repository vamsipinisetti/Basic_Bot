from __future__ import annotations

from typing import List, Dict, Any


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    if not text:
        return []

    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start += step
    return chunks


def chunk_documents(documents: List[Dict[str, Any]], chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
    chunks: List[Dict[str, Any]] = []
    for doc in documents:
        text = doc.get("text", "")
        page = doc.get("page", "N/A")
        source = doc.get("source", "unknown")
        for chunk_text_value in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append({
                "text": chunk_text_value.strip(),
                "source": source,
                "page": page,
            })
    return chunks
