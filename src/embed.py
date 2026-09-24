from __future__ import annotations

from typing import List, Dict, Any, Union

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def generate_embeddings(items: Union[List[str], List[Dict[str, Any]]], model_name: str = MODEL_NAME, batch_size: int = 32):
    model = SentenceTransformer(model_name)
    texts = []
    for item in items:
        if isinstance(item, dict):
            texts.append(item.get("text", ""))
        else:
            texts.append(str(item))

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)
