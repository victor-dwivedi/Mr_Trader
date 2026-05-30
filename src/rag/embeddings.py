from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding


@lru_cache(maxsize=1)
def _get_model(model_name: str = "BAAI/bge-small-en-v1.5") -> TextEmbedding:
    return TextEmbedding(model_name=model_name)


def embed_texts(texts: list[str], model_name: str = "BAAI/bge-small-en-v1.5") -> list[list[float]]:
    model = _get_model(model_name)
    return [e.tolist() for e in model.embed(texts)]


def embed_text(text: str, model_name: str = "BAAI/bge-small-en-v1.5") -> list[float]:
    return embed_texts([text], model_name)[0]
