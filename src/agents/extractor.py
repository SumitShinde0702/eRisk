"""Extractor Agent - extracts BDI-II symptom signals from conversation (DeepSeek)."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from openai import OpenAI

from src.bdi_mapper import BDI_SYMPTOMS
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

if TYPE_CHECKING:
    pass


EXTRACTOR_SYSTEM = """You are an Extractor agent. Your ONLY job is to analyze patient responses and output BDI-II symptom scores.

BDI-II has 21 symptoms. Each is scored 0-3:
- 0 = no indication
- 1 = mild
- 2 = moderate  
- 3 = severe

Output a JSON object with keys exactly matching these symptom names (use exact spelling):
Sadness, Pessimism, Past Failure, Loss of Pleasure, Guilty Feelings, Punishment Feelings, Self-Dislike, Self-Criticalness, Suicidal Thoughts or Wishes, Crying, Agitation, Loss of Interest, Indecisiveness, Worthlessness, Loss of Energy, Changes in Sleeping Pattern, Irritability, Changes in Appetite, Concentration Difficulty, Tiredness or Fatigue, Loss of Interest in Sex

Only include symptoms that you can infer from the given text. For symptoms not mentioned or unclear, use 0.

CRITICAL - Suicidal ideation and worthlessness: If the person expresses wanting to die, not wanting to be here, that the world would be better without them, or that they are worthless/capable of nothing/a speck, you MUST score "Suicidal Thoughts or Wishes" and/or "Worthlessness" at 2 or 3. Do not under-score these. Informal language (wanna die, wanna dieee, etc.) counts. These are clinically critical.

CRITICAL - Explicit suicide plans/statements: If the person says they will end their life, kill themselves, "gona end it", "im happy its coming to an end", "sleep forever soon", "when i go" (meaning death), "happy to die"—score "Suicidal Thoughts or Wishes" at 3 (severe). The word "happy" in that context refers to the suicide, NOT positive mental health. Do NOT apply the subclinical/positive-tone rule. "Happy to die" = severe suicidal ideation.

CRITICAL - Fatalistic/resigned language: Phrases like "accepted my fate", "know how this ends", "ik how this ends", "made my peace", "it doesn't matter anymore" can indicate suicidal ideation or severe hopelessness—even when the person says they feel "happy" or "good". Paradoxical calm before crisis. Score "Suicidal Thoughts or Wishes" and/or "Pessimism" at 2 when you see this pattern.

CRITICAL - Consider overall tone: If the person says they are "doing well", "feeling good", "overall fine", "pretty good", or similar POSITIVE framing, treat minor complaints (a bit tired, putting things off, foggy sometimes) as SUBCLINICAL. Use 0 or at most 1 for those. Do NOT score 2 for fatigue/focus/interest when the person explicitly states they are doing well overall. EXCEPTION: If they are "happy" about ending their life or death, that is NOT positive—score Suicidal at 3.

Output ONLY valid JSON, no other text.
"""


def _get_client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY is required for Extractor agent. Set it in .env")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def extract_symptoms(
    conversation: list[dict[str, str]],
    existing_signals: dict[str, int] | None = None,
) -> dict[str, int]:
    """
    Extract BDI symptom scores (0-3) from the conversation.

    Args:
        conversation: List of {"role": "user"|"assistant", "message": "..."}
        existing_signals: Previous extraction to merge/update (we take max per symptom)

    Returns:
        {symptom_name: score 0-3} - only symptoms with score > 0, plus any from existing
    """
    existing_signals = existing_signals or {}
    conv_text = "\n".join(
        f"{m['role'].upper()}: {m['message']}" for m in conversation
    )
    if not conv_text.strip():
        return dict(existing_signals)

    prompt = f"""Analyze this conversation and output BDI-II symptom scores (0-3) as JSON.

Conversation:
{conv_text}

Output a JSON object with symptom names as keys and 0-3 as values. Only include symptoms you can infer. Example: {{"Sadness": 2, "Loss of Interest": 1}}"""

    client = _get_client()
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTOR_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
        temperature=0.1,
    )
    text = (resp.choices[0].message.content or "").strip()

    # Parse JSON from response (handle markdown code blocks)
    parsed: dict[str, int] = {}
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Validate keys and clamp values
    result = dict(existing_signals)
    for sym in BDI_SYMPTOMS:
        if sym in parsed:
            val = parsed[sym]
            if isinstance(val, (int, float)):
                result[sym] = max(result.get(sym, 0), min(3, max(0, int(val))))

    # Contradiction-aware damping: preserve non-suicidal hopeful framing from being over-penalized.
    hopeful_markers = (
        "i want to be happy again",
        "i don't want to give up",
        "i dont want to give up",
        "i hope things get better",
        "trying to get through",
        "i want things to get better",
        "i'm trying",
        "im trying",
    )
    patient_text = " ".join(
        (m.get("message") or "").lower()
        for m in conversation
        if m.get("role") == "assistant"
    )
    has_hopeful = any(m in patient_text for m in hopeful_markers)
    has_acute = any(
        p in patient_text
        for p in (
            "end it",
            "kill myself",
            "want to die",
            "wanna die",
            "going to do it",
            "leave earth",
            "better off without me",
        )
    )
    if has_hopeful and not has_acute:
        for sym in ("Pessimism", "Worthlessness", "Suicidal Thoughts or Wishes"):
            if result.get(sym, 0) > 1:
                result[sym] = 1

    # Evidence-count gating for volatile cognitive/affective symptoms.
    patient_messages = [
        (m.get("message") or "").lower()
        for m in conversation
        if m.get("role") == "assistant"
    ]
    symptom_patterns: dict[str, tuple[str, ...]] = {
        "Agitation": ("snappy", "on edge", "restless", "agitated", "irritable", "frustrated with everyone"),
        "Indecisiveness": ("can't decide", "cannot decide", "hard to decide", "stuck", "don't know what to do"),
        "Worthlessness": ("worthless", "not good enough", "better without me", "capable of nothing", "i'm a burden", "im a burden"),
        "Concentration Difficulty": ("can't concentrate", "cannot concentrate", "mind wanders", "staring at my screen", "miss deadlines", "can't focus", "cannot focus"),
    }
    for symptom, patterns in symptom_patterns.items():
        if result.get(symptom, 0) <= 1:
            continue
        evidence_hits = sum(1 for msg in patient_messages if any(p in msg for p in patterns))
        if evidence_hits < 2:
            result[symptom] = 1

    # Safety override: explicit suicidal statements always score 3
    _SUICIDAL_OVERRIDE = [
        "gona end it", "gonna end it", "end it", "im happy its coming to an end",
        "sleep forever", "sleep forver", "happy to die", "when i go", "coming to an end",
    ]
    if any(p in patient_text for p in _SUICIDAL_OVERRIDE):
        result["Suicidal Thoughts or Wishes"] = max(
            result.get("Suicidal Thoughts or Wishes", 0), 3
        )

    return result


# Keyword patterns for fallback extraction (when DeepSeek unavailable, e.g. mock runs)
# Maps BDI symptom -> list of (phrase_patterns, score 1-3). First match wins.
_FALLBACK_PATTERNS: list[tuple[list[str], int]] = [
    (["not great", "sad", "unhappy", "miserable", "down", "low", "terrible"], 2),
    (["hopeless", "future", "nothing will work", "accepted my fate", "know how it ends", "how this ends", "ik how this ends", "it doesn't matter anymore"], 2),
    (["failure", "failed", "let everyone down"], 2),
    (["no pleasure", "nothing interests", "don't enjoy", "can't enjoy"], 2),
    (["guilty", "guilt"], 1),
    (["punished", "punishment"], 1),
    (["dislike myself", "hate myself", "worthless", "disappointed in myself", "capable of nothing", "just a speck", "better without me"], 2),
    (["harder on myself", "critical of myself", "blame myself"], 1),
    (["suicidal", "kill myself", "don't want to be here", "end it", "wanna die", "want to die", "accepted my fate", "know how this ends", "ik how this ends", "how it ends", "made my peace"], 3),
    (["crying", "cry", "tears", "emotional"], 1),
    (["restless", "agitated", "on edge", "wound up"], 1),
    (["lost interest", "nothing interests", "don't care about", "lost most of my interest"], 2),
    (["hard to decide", "can't decide", "indecisive"], 1),
    (["worthless", "useless", "nothing i do matters", "capable of nothing", "just a speck", "world will be better without me"], 3),
    (["no energy", "tired all the time", "exhausted", "drained"], 2),
    (["sleep", "insomnia", "wake up", "can't sleep", "sleep's been"], 1),
    (["irritable", "snap at", "short-tempered", "annoyed"], 1),
    (["appetite", "eating", "no appetite", "crave food"], 1),
    (["can't concentrate", "mind wanders", "focus", "concentrate"], 1),
    (["tired", "fatigued", "worn out", "exhausted"], 2),
    (["interest in sex", "libido", "not interested in that"], 1),
]


def extract_symptoms_fallback(
    conversation: list[dict[str, str]],
    existing_signals: dict[str, int] | None = None,
) -> dict[str, int]:
    """
    Rule-based fallback when DeepSeek API unavailable (e.g. mock runs).
    Uses keyword matching on assistant messages to infer BDI symptoms.
    """
    existing_signals = existing_signals or {}
    result = dict(existing_signals)
    text = " ".join(
        (m.get("message") or "").lower()
        for m in conversation
        if m.get("role") == "assistant"
    )
    if not text:
        return result

    for idx, (patterns, score) in enumerate(_FALLBACK_PATTERNS):
        if idx >= len(BDI_SYMPTOMS):
            break
        sym = BDI_SYMPTOMS[idx]
        for p in patterns:
            if p in text:
                result[sym] = max(result.get(sym, 0), min(3, score))
                break

    return result
