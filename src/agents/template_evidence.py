"""Template-based turn evidence scoring using symptom templates."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os
from typing import TypedDict

import yaml

from src.bdi_mapper import BDI_SYMPTOMS
from src.config import PROJECT_ROOT


class TemplateMatch(TypedDict):
    symptom: str
    template: str
    score: float


_TEMPLATE_PATH = PROJECT_ROOT / "knowledge" / "symptom_templates.yaml"
_RISK_LEXICON_PATH = PROJECT_ROOT / "knowledge" / "risk_lexicon.yaml"


@lru_cache(maxsize=1)
def _load_template_entries() -> list[tuple[str, str]]:
    """Return flattened (symptom, template_text) entries."""
    data = yaml.safe_load(Path(_TEMPLATE_PATH).read_text(encoding="utf-8"))
    entries: list[tuple[str, str]] = []
    for item in data.get("templates", []):
        symptom = item.get("symptom", "")
        if symptom not in BDI_SYMPTOMS:
            continue
        for text in item.get("texts", []):
            if text:
                entries.append((symptom, text))
    for item in data.get("direct_descriptions", []):
        symptom = item.get("symptom", "")
        text = item.get("text", "")
        if symptom in BDI_SYMPTOMS and text:
            entries.append((symptom, text))
    return entries


@lru_cache(maxsize=1)
def _load_risk_lexicon() -> dict[str, list[str]]:
    data = yaml.safe_load(Path(_RISK_LEXICON_PATH).read_text(encoding="utf-8"))
    return {
        "acute_safety": [str(x).lower() for x in data.get("acute_safety", [])],
        "hopeless_worthless": [str(x).lower() for x in data.get("hopeless_worthless", [])],
    }


@lru_cache(maxsize=1)
def _get_embedder():
    if os.getenv("DISABLE_TEMPLATE_EMBEDDINGS", "0") == "1":
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


@lru_cache(maxsize=1)
def _template_embedding_cache() -> tuple[list[tuple[str, str]], "object"]:
    import numpy as np

    entries = _load_template_entries()
    embedder = _get_embedder()
    if embedder is None:
        return entries, np.asarray([])
    texts = [tpl for _, tpl in entries]
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return entries, np.asarray(embeddings)


def _lexical_matches(text: str, top_k: int) -> list[TemplateMatch]:
    """Fast lexical fallback if embeddings fail."""
    text_l = text.lower()
    scored: list[TemplateMatch] = []
    for symptom, tpl in _load_template_entries():
        tokens = [t for t in tpl.lower().replace(".", "").split() if len(t) > 2]
        if not tokens:
            continue
        overlap = sum(1 for t in tokens if t in text_l)
        if overlap <= 0:
            continue
        score = overlap / max(len(tokens), 1)
        scored.append({"symptom": symptom, "template": tpl, "score": float(score)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def get_top_template_matches(text: str, top_k: int = 3) -> list[TemplateMatch]:
    """Return top-k symptom template matches for one assistant turn."""
    if not text.strip():
        return []
    try:
        import numpy as np

        entries, embeddings = _template_embedding_cache()
        if embeddings.size == 0:
            return _lexical_matches(text, top_k=top_k)
        embedder = _get_embedder()
        if embedder is None:
            return _lexical_matches(text, top_k=top_k)
        query = embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = np.dot(embeddings, query)
        idxs = np.argsort(-sims)[: max(top_k * 3, top_k)]

        results: list[TemplateMatch] = []
        seen_symptoms: set[str] = set()
        for i in idxs:
            symptom, tpl = entries[int(i)]
            score = float(sims[int(i)])
            if score < 0.2:
                continue
            if symptom in seen_symptoms:
                continue
            results.append({"symptom": symptom, "template": tpl, "score": round(score, 4)})
            seen_symptoms.add(symptom)
            if len(results) >= top_k:
                break
        return results
    except Exception:
        return _lexical_matches(text, top_k=top_k)


def compute_turn_risk_score(turn_text: str, matches: list[TemplateMatch]) -> float:
    """Combine template similarity and acute lexicon cues into a risk score."""
    text_l = turn_text.lower()
    base = max((m["score"] for m in matches), default=0.0)
    lexicon = _load_risk_lexicon()
    acute_bonus = 0.0
    if any(p in text_l for p in lexicon["acute_safety"]):
        acute_bonus += 0.5
    if any(p in text_l for p in lexicon["hopeless_worthless"]):
        acute_bonus += 0.2
    suicidal_bonus = 0.2 if any(m["symptom"] == "Suicidal Thoughts or Wishes" for m in matches) else 0.0
    return round(min(1.0, base + acute_bonus + suicidal_bonus), 4)
