from __future__ import annotations

import pickle
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
from openai import OpenAI

from src.config import load_config
from src.embed import generate_embeddings


ROOT_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT_DIR / "faiss_index"


def get_client():
    config = load_config()
    api_key = config["GROQ_API_KEY"]
    base_url = config["GROQ_BASE_URL"]
    model = config["GROQ_MODEL"]

    if not api_key or not model:
        return None, None

    return OpenAI(api_key=api_key, base_url=base_url), model


def load_index_and_metadata(index_path: Path = INDEX_DIR / "index.faiss", metadata_path: Path = INDEX_DIR / "metadata.pkl"):
    index = faiss.read_index(str(index_path))
    with metadata_path.open("rb") as file:
        metadata = pickle.load(file)
    return index, metadata


def format_sources(chunks):
    sources = []
    seen = set()
    for chunk in chunks:
        page_label = chunk.get("page", "N/A")
        source = chunk.get("source", "unknown")
        item = f"- {source}, page {page_label}"
        if item not in seen:
            sources.append(item)
            seen.add(item)
    return sources


def build_fallback_answer(question: str, selected_chunks):
    if not selected_chunks:
        return "I could not find relevant information in the provided knowledge base for this question.", []

    snippet = selected_chunks[0]["text"]
    answer = (
        "Based on the retrieved knowledge base, "
        f"{snippet[:500].strip()}"
        "\n\nThis answer is generated from the closest matching passages in the local documents."
    )
    return answer, format_sources(selected_chunks)


def answer_question(question: str, top_k: int = 4):
    index, metadata = load_index_and_metadata()
    query_embedding = generate_embeddings([question], batch_size=1)
    _, indices = index.search(np.asarray(query_embedding, dtype=np.float32), top_k)

    selected_chunks = []
    for idx in indices[0]:
        if idx == -1:
            continue
        selected_chunks.append(metadata[int(idx)])

    context = "\n\n".join(
        f"Source: {chunk['source']} | Page: {chunk['page']}\n{chunk['text']}"
        for chunk in selected_chunks
    )

    client, model_name = get_client()
    if client is None or model_name is None:
        answer, sources = build_fallback_answer(question, selected_chunks)
        return {"answer": answer, "sources": sources}

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer the question using the provided context. If the answer is not in the context, say so clearly. Include source citations using the provided source names and page numbers."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        answer = response.choices[0].message.content.strip()
        sources = format_sources(selected_chunks)
        return {"answer": answer, "sources": sources}
    except Exception:
        answer, sources = build_fallback_answer(question, selected_chunks)
        return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    while True:
        question = input("Ask a question (or type 'exit' to quit): ").strip()
        if not question or question.lower() == "exit":
            break
        result = answer_question(question)
        print("\nAnswer:")
        print(result["answer"])
        print("\nSources:")
        for item in result["sources"]:
            print(item)
