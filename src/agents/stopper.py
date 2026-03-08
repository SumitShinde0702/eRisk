"""Stopping Agent: rule-based decision to continue or CLASSIFY."""

from __future__ import annotations

from src.bdi_mapper import BDI_SYMPTOMS
from src.config import MAX_MESSAGES, MIN_EXCHANGES_BEFORE_STOP, MIN_SYMPTOMS_FOR_EARLY_STOP


def should_classify(
    *,
    message_count: int,
    symptom_signals: dict[str, int],
    last_bdi_estimate: int | None = None,
    prev_bdi_estimate: int | None = None,
) -> bool:
    """
    Decide whether to continue the conversation or CLASSIFY.

    Returns True if we should classify now, False to continue.

    Rules:
    - Always stop at MAX_MESSAGES
    - Early stop if BDI estimate stable (unchanged) and we have enough symptom coverage
    - Early stop if clearly control (low score) after minimum messages
    - Early stop if clearly depressed (high score) after minimum messages
    """
    if message_count >= MAX_MESSAGES:
        return True

    # Enforce minimum interview length for stronger evidence collection.
    if message_count < MIN_EXCHANGES_BEFORE_STOP:
        return False

    # Count how many symptoms we have signals for (score > 0)
    symptoms_with_signals = sum(1 for s in BDI_SYMPTOMS if symptom_signals.get(s, 0) > 0)
    total_score = sum(symptom_signals.get(s, 0) for s in BDI_SYMPTOMS)

    # Clearly control: low total score, few symptoms
    if message_count >= 3 and total_score <= 5 and symptoms_with_signals <= 2:
        return True

    # Clearly depressed: high score
    if message_count >= 4 and total_score >= 25:
        return True

    # BDI estimate stable and enough coverage
    if (
        last_bdi_estimate is not None
        and prev_bdi_estimate is not None
        and last_bdi_estimate == prev_bdi_estimate
        and symptoms_with_signals >= MIN_SYMPTOMS_FOR_EARLY_STOP
        and message_count >= 4
    ):
        return True

    return False


# Positive framing phrases - if present in early responses, use relaxed control threshold
_POSITIVE_FRAMING = (
    "doing well", "doing pretty well", "feeling good", "overall feeling good",
    "pretty good", "alright", "doing okay", "doing fine", "feeling fine",
    "overall fine", "nothing to report", "i'm good", "i've been good",
)

_ACUTE_SAFETY_CUES = (
    "end it",
    "kill myself",
    "going to do it",
    "going to end",
    "not be here",
    "don't want to be here",
    "wanna die",
    "want to die",
    "gone soon",
    "leave the earth",
    "ending it",
)

_SAFETY_PROBE_CUES = (
    "soon",
    "trigger",
    "strongest",
    "hold on",
    "alone",
    "what do you mean",
    "immediate",
    "waves",
)


def _has_positive_framing(conversation: list) -> bool:
    """Check if early assistant messages contain positive framing."""
    text = " ".join(
        (m.get("message") or "").lower()
        for m in conversation[:6]  # First 3 exchanges
        if m.get("role") == "assistant"
    )
    return any(p in text for p in _POSITIVE_FRAMING)


def _has_acute_safety_signals(conversation: list) -> bool:
    """Detect high-risk language in patient responses."""
    text = " ".join(
        (m.get("message") or "").lower()
        for m in conversation
        if m.get("role") == "assistant"
    )
    return any(cue in text for cue in _ACUTE_SAFETY_CUES)


def _count_safety_probes(conversation: list) -> int:
    """Estimate how many risk-focused questions were asked."""
    count = 0
    for m in conversation:
        if m.get("role") != "user":
            continue
        q = (m.get("message") or "").lower()
        if any(cue in q for cue in _SAFETY_PROBE_CUES):
            count += 1
    return count


def should_stop(
    conversation: list,
    symptom_signals: dict[str, int],
    probed_symptoms: set[int],
) -> tuple[bool, str]:
    """
    Wrapper for orchestrator: (conversation, symptom_signals, probed_symptoms) -> (stop, reason).
    """
    message_count = len(conversation) // 2
    total = sum(symptom_signals.get(s, 0) for s in BDI_SYMPTOMS)
    symptoms_with_signals = sum(1 for s in BDI_SYMPTOMS if symptom_signals.get(s, 0) > 0)
    acute_risk = _has_acute_safety_signals(conversation)

    # Risk-first policy: if acute safety cues exist, keep probing safety intent/context
    # before ending conversation, unless max cap is reached.
    if acute_risk and message_count < MAX_MESSAGES:
        if message_count < max(MIN_EXCHANGES_BEFORE_STOP, 12):
            return False, "acute_safety_min_depth"
        if _count_safety_probes(conversation) < 4:
            return False, "acute_safety_insufficient_probes"

    # If extractor has not produced any evidence yet, continue probing.
    if message_count < MIN_EXCHANGES_BEFORE_STOP:
        return False, "min_exchanges_not_reached"

    # If early responses have positive framing, use relaxed control threshold
    # (stops sooner to avoid over-probing and false symptom extraction)
    if message_count >= MIN_EXCHANGES_BEFORE_STOP and _has_positive_framing(conversation):
        if total <= 8 and symptoms_with_signals <= 4:
            return True, "positive_framing_early_stop"

    stop = should_classify(
        message_count=message_count,
        symptom_signals=symptom_signals,
        last_bdi_estimate=total,
        prev_bdi_estimate=total,
    )
    reason = "max_messages" if message_count >= MAX_MESSAGES else "early_stop"
    return stop, reason
