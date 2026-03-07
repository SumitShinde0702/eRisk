"""Scorer Agent: rule-based BDI score and key symptoms from symptom signals."""

from __future__ import annotations

from src.bdi_mapper import (
    BDI_SYMPTOMS,
    BDI_MAX_SCORE,
    validate_key_symptoms,
)


def compute_bdi_score(symptom_signals: dict[str, int]) -> int:
    """Sum all symptom scores (0-3 each) and clamp to 0-63."""
    total = sum(symptom_signals.get(s, 0) for s in BDI_SYMPTOMS)
    return min(max(total, 0), BDI_MAX_SCORE)


def select_key_symptoms(
    symptom_signals: dict[str, int],
    max_count: int = 4,
) -> list[str]:
    """
    Select up to 4 key symptoms with highest scores.
    Ties: prefer symptoms that appear earlier in BDI order (clinical convention).
    """
    scored = [(s, symptom_signals.get(s, 0)) for s in BDI_SYMPTOMS if symptom_signals.get(s, 0) > 0]
    scored.sort(key=lambda x: (-x[1], x[0]))  # descending score, then by name order
    selected = [s for s, _ in scored[:max_count]]
    return validate_key_symptoms(selected)


def score(symptom_signals: dict[str, int]) -> tuple[int, list[str]]:
    """
    Given symptom signals (symptom name -> 0-3 score), return (bdi_score, key_symptoms).
    """
    bdi_score = compute_bdi_score(symptom_signals)
    key_symptoms = select_key_symptoms(symptom_signals)
    return bdi_score, key_symptoms


# Alias for orchestrator
score_bdi = score
