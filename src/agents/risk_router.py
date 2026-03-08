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

CLUSTER_QUESTION_BANK: dict[str, list[str]] = {
    "AcuteSafety": [
        "You said this might happen soon - what has made things feel this immediate right now?",
        "When these thoughts get strongest, what usually triggers them?",
        "Have these thoughts been constant lately, or do they come in waves?",
        "What has helped you hold on, even a little, when those thoughts show up?",
        "How alone do you feel with this right now?",
    ],
    "HopelessWorthless": [
        "You mentioned that things feel pointless - when did that start feeling this strong?",
        "When you say nothing matters, what parts of life feel most empty lately?",
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
        "not be here",
        "don't want to be here",
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


def _lexical_cluster(conversation: list[dict[str, str]]) -> ClusterDecision:
    text = _recent_patient_text(conversation, max_msgs=6).lower()
    for cluster in ("AcuteSafety", "HopelessWorthless", "CoreDepression", "VegetativeCognitive", "BehavioralArousal"):
        if any(p in text for p in _LEXICAL_RULES.get(cluster, ())):
            return {"cluster": cluster, "confidence": 0.7, "rationale": f"matched {cluster} lexical cues"}
    return {"cluster": "GeneralCheckin", "confidence": 0.5, "rationale": "no strong lexical cue"}


def classify_cluster(conversation: list[dict[str, str]]) -> ClusterDecision:
    """Classify current dominant interview cluster."""
    if not DEEPSEEK_API_KEY:
        return _lexical_cluster(conversation)

    patient_excerpt = _recent_patient_text(conversation, max_msgs=6)
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
        return _lexical_cluster(conversation)
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
        return _lexical_cluster(conversation)


def next_cluster_question(cluster: str, asked_questions: list[str]) -> str:
    """Return next non-repeated question from selected cluster bank."""
    candidates = CLUSTER_QUESTION_BANK.get(cluster) or CLUSTER_QUESTION_BANK["GeneralCheckin"]
    asked_set = {q.lower().strip() for q in asked_questions}
    for q in candidates:
        if q.lower().strip() not in asked_set:
            return q
    return candidates[0]
