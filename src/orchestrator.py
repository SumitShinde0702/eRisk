"""A2A loop coordinator - orchestrates Prober, Extractor, Stopper, Scorer, and Persona."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.agents.extractor import extract_symptoms
from src.agents.prober import get_next_question
from src.agents.scorer import score_bdi
from src.agents.stopper import should_stop


class PersonaProtocol(Protocol):
    """Protocol for persona clients (real or mock)."""

    def chat(self, user_message: str) -> str: ...


@dataclass
class ConversationState:
    """State for one persona conversation."""

    conversation: list[dict[str, str]] = field(default_factory=list)
    symptom_signals: dict[str, int] = field(default_factory=dict)
    probed_symptoms: set[int] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation": list(self.conversation),
            "symptom_signals": dict(self.symptom_signals),
            "probed_symptoms": list(self.probed_symptoms),
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


def run_conversation(
    persona: PersonaProtocol,
    persona_id: str,
    *,
    use_extractor: bool = True,
) -> tuple[list[dict[str, str]], int, list[str]]:
    """
    Run the A2A loop for one persona.

    Returns:
        (conversation, bdi_score, key_symptoms)
    """
    from src.config import MAX_MESSAGES

    state = ConversationState()

    while len(state.conversation) // 2 < MAX_MESSAGES:
        # 1. Stopper: continue or CLASSIFY?
        stop, _ = should_stop(
            state.conversation,
            state.symptom_signals,
            state.probed_symptoms,
        )
        if stop:
            break

        # 2. Prober: next question
        probed = _infer_probed_from_questions(state.conversation)
        state.probed_symptoms = probed
        question = get_next_question(state.conversation, probed)

        # 3. Persona: response
        response = persona.chat(question)

        # 4. Append to conversation
        state.conversation.append({"role": "user", "message": question})
        state.conversation.append({"role": "assistant", "message": response})

        # 5. Extractor: update symptom signals
        if use_extractor:
            state.symptom_signals = extract_symptoms(
                state.conversation,
                existing_signals=state.symptom_signals,
            )
        else:
            # Fallback: no extraction (e.g. missing API key)
            state.symptom_signals = state.symptom_signals or {}

    # 6. Scorer: final BDI score and key symptoms
    bdi_score, key_symptoms = score_bdi(state.symptom_signals)

    return state.conversation, bdi_score, key_symptoms
