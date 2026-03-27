"""Embedding-based evidence retrieval for conversation grounding."""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_embedder():
    """Lazy-load embedding model to avoid startup overhead."""
    if os.getenv("DISABLE_TEMPLATE_EMBEDDINGS", "0") == "1":
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


def _lexical_fallback(
    patient_messages: list[str],
    query_text: str,
    top_k: int,
) -> list[str]:
    """Fallback retrieval using token overlap when embedding model fails."""
    q_terms = set(query_text.lower().split())
    ranked: list[tuple[int, str]] = []
    for msg in patient_messages:
        m_terms = set(msg.lower().split())
        score = len(q_terms & m_terms)
        ranked.append((score, msg))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [m for s, m in ranked[:top_k] if s > 0]


def retrieve_relevant_patient_evidence(
    conversation: list[dict[str, str]],
    query_text: str,
    top_k: int = 3,
) -> list[str]:
    """
    Retrieve the most relevant prior patient messages for a query.

    Uses sentence embeddings + cosine similarity. Falls back to lexical overlap.
    """
    patient_messages = [
        (m.get("message") or "").strip()
        for m in conversation
        if m.get("role") == "assistant" and (m.get("message") or "").strip()
    ]
    if not patient_messages or not query_text.strip():
        return []

    try:
        import numpy as np

        embedder = _get_embedder()
        if embedder is None:
            return _lexical_fallback(patient_messages, query_text, top_k)
        msg_emb = embedder.encode(patient_messages, convert_to_numpy=True, normalize_embeddings=True)
        q_emb = embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = np.dot(msg_emb, q_emb)
        idxs = np.argsort(-sims)[:top_k]
        return [patient_messages[i] for i in idxs if float(sims[i]) > 0.2]
    except Exception:
        return _lexical_fallback(patient_messages, query_text, top_k)
