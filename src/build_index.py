from __future__ import annotations

import pickle
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np

from src.chunk import chunk_documents
from src.embed import generate_embeddings
from src.ingest import load_documents


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"
INDEX_DIR.mkdir(exist_ok=True)


def build_index(data_dir: Path = DATA_DIR):
    documents = load_documents(data_dir)
    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError("No chunks were created from the source documents.")

    embeddings = generate_embeddings(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.asarray(embeddings, dtype=np.float32))

    index_path = INDEX_DIR / "index.faiss"
    metadata_path = INDEX_DIR / "metadata.pkl"

    faiss.write_index(index, str(index_path))
    with metadata_path.open("wb") as file:
        pickle.dump(chunks, file)

    return index, chunks


if __name__ == "__main__":
    build_index()
    print("FAISS index created successfully.")
