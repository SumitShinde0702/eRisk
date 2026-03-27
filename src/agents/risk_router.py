"""Risk-cluster routing for prober question selection."""

from __future__ import annotations

import json
import re
from typing import TypedDict

from openai import OpenAI

from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


class ClusterDecision(TypedDict):
    cluster: str
    confidence: float
    rationale: str


CLUSTERS: tuple[str, ...] = (
    "AcuteSafety",
    "HopelessWorthless",
    "CoreDepression",
    "VegetativeCognitive",
    "BehavioralArousal",
    "GeneralCheckin",
)

ACUTE_EUPHEMISMS: tuple[str, ...] = (
    "end soon",
    "all going to end",
    "going to end soon",
    "leave earth",
    "leave the earth",
    "put everything behind",
    "be gone soon",
    "it will all be over",
    "won't be here",
    "wont be here",
    "waiting for the timer",
    "timer to go off",
    "no one would notice if i wasn't here",
    "no one would notice if i wasnt here",
    "want to disappear",
    "end of something",
)

ACUTE_LADDER: list[tuple[str, str, tuple[str, ...]]] = [
    (
        "intent_clarification",
        "When you say this will end soon, are you talking about ending your life?",
        ("ending your life", "talking about ending", "end your life"),
    ),
    (
        "plan_concreteness",
        "Have you thought about specific steps or a specific way you would do it?",
        ("specific steps", "specific way", "how you would do it", "plan"),
    ),
    (
        "timeline_imminence",
        "Have you thought about when you might do this?",
        ("when you might", "when this might happen", "timeline", "soon"),
    ),
    (
        "means_access",
        "Do you currently have access to what you would use?",
        ("access", "what you would use", "means"),
    ),
    (
        "protective_factors",
        "What has stopped you so far from acting on these thoughts?",
        ("stopped you so far", "acting on these thoughts", "hold on", "kept you safe"),
    ),
]

CLUSTER_QUESTION_BANK: dict[str, list[str]] = {
    "AcuteSafety": [
        "You said this might happen soon - what has made things feel this immediate right now?",
        "When these thoughts get strongest, what usually triggers them?",
        "Have these thoughts been constant lately, or do they come in waves?",
        "What has helped you hold on, even a little, when those thoughts show up?",
        "How alone do you feel with this right now?",
    ],
    "HopelessWorthless": [
        "When did this sense of emptiness start feeling this strong?",
        "What parts of life feel most difficult or empty lately?",
        "How do you see the next few weeks unfolding for you?",
        "How much of this feels like giving up versus feeling exhausted?",
        "Do you feel like people would be better off without you, or is it more that you feel stuck?",
    ],
    "CoreDepression": [
        "How often have you felt low like this over the past week?",
        "What used to lift your mood before, and does any of that still help now?",
        "When during the day do you feel the worst?",
        "Have you noticed yourself pulling away from people or activities you usually care about?",
        "How different does your current mood feel compared with your usual self?",
    ],
    "VegetativeCognitive": [
        "How has your sleep been affecting your day-to-day energy lately?",
        "How hard has it been to focus on routine tasks recently?",
        "Have your appetite or eating patterns changed much lately?",
        "Do simple decisions feel harder than they used to?",
        "How much physical exhaustion are you carrying most days?",
    ],
    "BehavioralArousal": [
        "Have you felt more on edge or restless than usual?",
        "Do you find yourself getting irritated faster lately?",
        "Have others noticed changes in your reactions or temper?",
        "When stress rises, what does it look like in your body or behavior?",
        "How hard has it been to stay calm in everyday situations lately?",
    ],
    "GeneralCheckin": [
        "How have things been for you lately?",
        "What has this past week felt like for you overall?",
        "What has been the hardest part of your days recently?",
        "How are you managing day to day at the moment?",
        "What has changed most for you recently?",
    ],
}

_LEXICAL_RULES: dict[str, tuple[str, ...]] = {
    "AcuteSafety": (
        "end it",
        "kill myself",
        "going to do it",
        "going to end",
        "die",
        "gone soon",
        "leave the earth",
        "leave earth",
        "put everything behind",
        "all going to end",
        "end soon",
        "not be here",
        "don't want to be here",
        "no one would notice if i wasn't here",
        "no one would notice if i wasnt here",
        "want to disappear",
        "waiting for the timer",
        "timer to go off",
        "suicide",
    ),
    "HopelessWorthless": (
        "nothing matters",
        "pointless",
        "worthless",
        "better without me",
        "accepted my fate",
        "know how this ends",
        "made my peace",
        "hopeless",
    ),
    "CoreDepression": ("down", "sad", "empty", "don't enjoy", "anhedonia", "no interest"),
    "VegetativeCognitive": ("sleep", "insomnia", "tired", "fatigue", "focus", "concentrate", "appetite"),
    "BehavioralArousal": ("restless", "agitated", "irritable", "on edge", "snappy"),
}

_AMBIGUOUS_ACUTE_CUES: tuple[str, ...] = (
    "disappear",
    "just existing",
    "just exist",
    "no one would notice if i wasn't here",
    "no one would notice if i wasnt here",
    "waiting for the timer",
    "timer to go off",
    "end of something",
)

_CLASSIFIER_SYSTEM = """You are a clinical routing assistant for depression-screening interviews.
Classify the CURRENT dominant cluster based on patient statements.

Allowed clusters:
- AcuteSafety
- HopelessWorthless
- CoreDepression
- VegetativeCognitive
- BehavioralArousal
- GeneralCheckin

Prioritization:
- If patient indicates imminent self-harm, intent, plan, or "doing it soon", choose AcuteSafety.
- If resigned/fatalistic/worthlessness language dominates without clear imminence, choose HopelessWorthless.
- Otherwise choose the best-fit non-risk cluster.

Output ONLY valid JSON:
{"cluster":"<one_of_allowed>","confidence":0.0-1.0,"rationale":"short reason"}
"""


def _get_client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def _recent_patient_text(conversation: list[dict[str, str]], max_msgs: int = 4) -> str:
    msgs = [
        (m.get("message") or "").strip()
        for m in conversation
        if m.get("role") == "assistant" and (m.get("message") or "").strip()
    ]
    return "\n".join(msgs[-max_msgs:])


def _risk_buffer_text(risk_buffer: list[dict[str, object]] | None, max_msgs: int = 6) -> str:
    if not risk_buffer:
        return ""
    msgs = [
        str(item.get("assistant_message", "")).strip()
        for item in risk_buffer
        if str(item.get("assistant_message", "")).strip()
    ]
    return "\n".join(msgs[:max_msgs])


def _lexical_cluster(
    conversation: list[dict[str, str]],
    risk_buffer: list[dict[str, object]] | None = None,
) -> ClusterDecision:
    text = (_risk_buffer_text(risk_buffer, max_msgs=6) or _recent_patient_text(conversation, max_msgs=6)).lower()
    if any(p in text for p in ACUTE_EUPHEMISMS):
        return {"cluster": "AcuteSafety", "confidence": 0.9, "rationale": "matched acute euphemistic cues"}
    for cluster in ("AcuteSafety", "HopelessWorthless", "CoreDepression", "VegetativeCognitive", "BehavioralArousal"):
        if any(p in text for p in _LEXICAL_RULES.get(cluster, ())):
            return {"cluster": cluster, "confidence": 0.7, "rationale": f"matched {cluster} lexical cues"}
    return {"cluster": "GeneralCheckin", "confidence": 0.5, "rationale": "no strong lexical cue"}


def classify_cluster(
    conversation: list[dict[str, str]],
    *,
    risk_buffer: list[dict[str, object]] | None = None,
) -> ClusterDecision:
    """Classify current dominant interview cluster."""
    if has_acute_signal(conversation, risk_buffer=risk_buffer):
        return {"cluster": "AcuteSafety", "confidence": 0.95, "rationale": "acute safety cues detected"}

    if not DEEPSEEK_API_KEY:
        return _lexical_cluster(conversation, risk_buffer=risk_buffer)

    patient_excerpt = _risk_buffer_text(risk_buffer, max_msgs=6) or _recent_patient_text(conversation, max_msgs=6)
    if not patient_excerpt:
        return {"cluster": "GeneralCheckin", "confidence": 0.5, "rationale": "no patient history yet"}

    prompt = f"""Patient recent responses:
{patient_excerpt}

Return dominant cluster now."""
    client = _get_client()
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": _CLASSIFIER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=120,
        temperature=0.0,
    )
    text = (resp.choices[0].message.content or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return _lexical_cluster(conversation, risk_buffer=risk_buffer)
    try:
        parsed = json.loads(match.group())
        cluster = parsed.get("cluster", "GeneralCheckin")
        if cluster not in CLUSTERS:
            return _lexical_cluster(conversation)
        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        rationale = str(parsed.get("rationale", ""))
        return {"cluster": cluster, "confidence": confidence, "rationale": rationale[:200]}
    except Exception:
        return _lexical_cluster(conversation, risk_buffer=risk_buffer)


def next_cluster_question(cluster: str, asked_questions: list[str]) -> str:
    """Return next non-repeated question from selected cluster bank."""
    candidates = CLUSTER_QUESTION_BANK.get(cluster) or CLUSTER_QUESTION_BANK["GeneralCheckin"]
    asked_set = {q.lower().strip() for q in asked_questions}
    for q in candidates:
        if q.lower().strip() not in asked_set:
            return q
    return candidates[0]


def has_acute_signal(
    conversation: list[dict[str, str]],
    *,
    risk_buffer: list[dict[str, object]] | None = None,
) -> bool:
    """Detect acute safety intent language from patient messages."""
    text = (_risk_buffer_text(risk_buffer, max_msgs=8) or _recent_patient_text(conversation, max_msgs=8)).lower()
    acute_terms = _LEXICAL_RULES["AcuteSafety"] + ACUTE_EUPHEMISMS
    if any(term in text for term in acute_terms):
        return True
    ambiguous_hits = sum(1 for cue in _AMBIGUOUS_ACUTE_CUES if cue in text)
    return ambiguous_hits >= 2


def acute_ladder_progress(asked_questions: list[str]) -> int:
    """Return how many acute ladder stages have already been asked."""
    asked_lower = [q.lower() for q in asked_questions]
    completed = 0
    for _, _, markers in ACUTE_LADDER:
        if any(any(marker in q for marker in markers) for q in asked_lower):
            completed += 1
    return completed


def next_acute_ladder_question(
    conversation: list[dict[str, str]],
    asked_questions: list[str],
    *,
    risk_buffer: list[dict[str, object]] | None = None,
) -> str:
    """Return the next required acute-safety ladder question when high-risk cues are present."""
    if not has_acute_signal(conversation, risk_buffer=risk_buffer):
        return ""
    asked_lower = [q.lower() for q in asked_questions]
    for _, question, markers in ACUTE_LADDER:
        if any(any(marker in q for marker in markers) for q in asked_lower):
            continue
        return question
    return ""
