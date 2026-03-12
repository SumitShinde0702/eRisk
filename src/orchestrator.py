"""A2A loop coordinator - orchestrates Prober, Extractor, Stopper, Scorer, and Persona."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.agents.extractor import extract_symptoms
from src.agents.prober import get_next_question
from src.agents.scorer import score_bdi
from src.agents.stopper import should_stop
from src.agents.template_evidence import compute_turn_risk_score, get_top_template_matches


class PersonaProtocol(Protocol):
    """Protocol for persona clients (real or mock)."""

    def chat(self, user_message: str) -> str: ...


@dataclass
class ConversationState:
    """State for one persona conversation."""

    conversation: list[dict[str, str]] = field(default_factory=list)
    symptom_signals: dict[str, int] = field(default_factory=dict)
    probed_symptoms: set[int] = field(default_factory=set)
    turn_evidence: list[dict[str, Any]] = field(default_factory=list)
    risk_buffer: list[dict[str, Any]] = field(default_factory=list)
    bdi_trace: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation": list(self.conversation),
            "symptom_signals": dict(self.symptom_signals),
            "probed_symptoms": list(self.probed_symptoms),
            "turn_evidence": list(self.turn_evidence),
            "risk_buffer": list(self.risk_buffer),
            "bdi_trace": list(self.bdi_trace),
        }


def _infer_probed_from_questions(conversation: list[dict[str, str]]) -> set[int]:
    """Infer which symptom indices were probed from user messages (question bank match)."""
    probed: set[int] = set()
    from src.bdi_mapper import BDI_QUESTION_BANK

    for msg in conversation:
        if msg.get("role") != "user":
            continue
        text = (msg.get("message") or "").strip().lower()
        for idx, q in enumerate(BDI_QUESTION_BANK):
            if idx in probed:
                continue
            if q.strip().lower()[:30] in text or text[:40] in q.strip().lower()[:40]:
                probed.add(idx)
                break
    return probed


def _update_risk_buffer(
    state: ConversationState,
    *,
    max_size: int,
    recency_weight: float = 0.15,
) -> None:
    """
    Maintain top-K risky assistant turns with slight recency preference.
    """
    if not state.turn_evidence:
        return
    latest_turn = max(e["turn_index"] for e in state.turn_evidence)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for e in state.turn_evidence:
        turn_gap = max(0, latest_turn - int(e.get("turn_index", latest_turn)))
        recency_bonus = recency_weight / (turn_gap + 1)
        blended = float(e.get("risk_score", 0.0)) + recency_bonus
        ranked.append((blended, e))
    ranked.sort(key=lambda x: x[0], reverse=True)
    state.risk_buffer = [item for _, item in ranked[:max_size]]


def run_conversation(
    persona: PersonaProtocol,
    persona_id: str,
    *,
    use_extractor: bool = True,
    run_policy: dict[str, Any] | None = None,
) -> tuple[list[dict[str, str]], int, list[str]]:
    """
    Run the A2A loop for one persona.

    Returns:
        (conversation, bdi_score, key_symptoms)
    """
    from src.config import MAX_MESSAGES

    state = ConversationState()

    policy = run_policy or {}
    risk_buffer_size = int(policy.get("risk_buffer_size", 6))
    recency_weight = float(policy.get("recency_weight", 0.15))

    while len(state.conversation) // 2 < MAX_MESSAGES:
        # 1. Stopper: continue or CLASSIFY?
        stop, _ = should_stop(
            state.conversation,
            state.symptom_signals,
            state.probed_symptoms,
            risk_buffer=state.risk_buffer,
            run_policy=run_policy,
            recent_bdi_estimates=state.bdi_trace,
        )
        if stop:
            break

        # 2. Prober: next question
        probed = _infer_probed_from_questions(state.conversation)
        state.probed_symptoms = probed
        question = get_next_question(
            state.conversation,
            probed,
            risk_buffer=state.risk_buffer,
            run_policy=run_policy,
        )

        # 3. Persona: response
        response = persona.chat(question)

        # 4. Append to conversation
        state.conversation.append({"role": "user", "message": question})
        state.conversation.append({"role": "assistant", "message": response})

        # 4b. Template-based evidence scoring for this assistant turn.
        matches = get_top_template_matches(response, top_k=3)
        risk_score = compute_turn_risk_score(response, matches)
        evidence = {
            "turn_index": len(state.conversation) // 2,
            "assistant_message": response,
            "template_matches": matches,
            "risk_score": risk_score,
        }
        state.turn_evidence.append(evidence)
        _update_risk_buffer(
            state,
            max_size=risk_buffer_size,
            recency_weight=recency_weight,
        )

        # 5. Extractor: update symptom signals
        if use_extractor:
            state.symptom_signals = extract_symptoms(
                state.conversation,
                existing_signals=state.symptom_signals,
            )
        else:
            # Fallback: no extraction (e.g. missing API key)
            state.symptom_signals = state.symptom_signals or {}
        state.bdi_trace.append(sum(state.symptom_signals.values()))

    # 6. Scorer: final BDI score and key symptoms
    bdi_score, key_symptoms = score_bdi(
        state.symptom_signals,
        conversation=state.conversation,
        risk_buffer=state.risk_buffer,
        run_policy=run_policy,
    )

    return state.conversation, bdi_score, key_symptoms
