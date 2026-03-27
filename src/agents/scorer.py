"""Scorer Agent: rule-based BDI score and key symptoms from symptom signals."""

from __future__ import annotations

from src.agents.risk_router import acute_ladder_progress, has_acute_signal
from src.bdi_mapper import (
    BDI_SYMPTOMS,
    BDI_MAX_SCORE,
    validate_key_symptoms,
)


def compute_bdi_score(symptom_signals: dict[str, int]) -> int:
    """Sum all symptom scores (0-3 each) and clamp to 0-63."""
    total = sum(symptom_signals.get(s, 0) for s in BDI_SYMPTOMS)
    return min(max(total, 0), BDI_MAX_SCORE)


def _calibrate_score_for_acute_risk(
    base_score: int,
    symptom_signals: dict[str, int],
    *,
    conversation: list[dict] | None = None,
    run_policy: dict | None = None,
    risk_buffer: list[dict] | None = None,
) -> int:
    """
    Apply calibrated boosting when acute suicidal indicators are strongly supported.
    """
    conversation = conversation or []
    run_policy = run_policy or {}
    acute_floor = int(run_policy.get("acute_boost_floor", 38))
    required_ladder_steps = int(run_policy.get("required_acute_ladder_steps", 4))
    moderate_acute_floor = int(run_policy.get("moderate_acute_boost_floor", max(acute_floor - 5, 32)))
    mild_acute_floor = int(run_policy.get("mild_acute_boost_floor", max(acute_floor - 8, 28)))
    asked_questions = [
        (m.get("message") or "").strip()
        for m in conversation
        if m.get("role") == "user"
    ]
    acute_signal = has_acute_signal(conversation, risk_buffer=risk_buffer)
    ladder_steps = acute_ladder_progress(asked_questions)
    suicidal_signal = int(symptom_signals.get("Suicidal Thoughts or Wishes", 0))
    hopeless_signal = int(symptom_signals.get("Pessimism", 0))
    max_buffer_risk = max((float(x.get("risk_score", 0.0)) for x in (risk_buffer or [])), default=0.0)

    if acute_signal and suicidal_signal >= 2 and ladder_steps >= required_ladder_steps:
        additive = min(10, 3 * suicidal_signal + hopeless_signal + (2 if max_buffer_risk >= 0.85 else 0))
        return min(BDI_MAX_SCORE, max(base_score, acute_floor) + additive)
    if acute_signal and suicidal_signal >= 1 and ladder_steps >= max(2, required_ladder_steps - 1):
        additive = min(8, 2 * suicidal_signal + hopeless_signal + (1 if max_buffer_risk >= 0.8 else 0))
        return min(BDI_MAX_SCORE, max(base_score, moderate_acute_floor) + additive)
    if acute_signal and hopeless_signal >= 2 and max_buffer_risk >= 0.8:
        return max(base_score, mild_acute_floor)
    return base_score


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


def score(
    symptom_signals: dict[str, int],
    *,
    conversation: list[dict] | None = None,
    risk_buffer: list[dict] | None = None,
    run_policy: dict | None = None,
) -> tuple[int, list[str]]:
    """
    Given symptom signals (symptom name -> 0-3 score), return (bdi_score, key_symptoms).
    """
    bdi_score = compute_bdi_score(symptom_signals)
    bdi_score = _calibrate_score_for_acute_risk(
        bdi_score,
        symptom_signals,
        conversation=conversation,
        run_policy=run_policy,
        risk_buffer=risk_buffer,
    )
    key_symptoms = select_key_symptoms(symptom_signals)
    return bdi_score, key_symptoms


# Alias for orchestrator
score_bdi = score
