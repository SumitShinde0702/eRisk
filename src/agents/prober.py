"""Prober Agent - topic-based, conversational questioning. General -> specific, questions relate to each other."""

from __future__ import annotations

from openai import OpenAI

from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from src.topic_hierarchy import (
    TOPIC_ORDER,
    get_next_topic,
    get_topic_by_id,
)

_PROBER_SYSTEM = """You are a Prober Agent for a clinical conversation. Your job is to output the NEXT question to ask a simulated patient.

CRITICAL RULES:
- Output ONLY a single natural question. No explanations, no JSON, no preamble.
- NEVER ask directly about depression, mental health, or "Are you depressed?"
- Start GENERAL, then move to SPECIFIC. Do not interrogate symptom-by-symptom.
- Questions must RELATE to each other. Build on what they said. E.g. "You mentioned sleep - has that affected your energy during the day?"
- CRITICAL: If the patient says something concerning, ambiguous, or unusual (e.g. "accepted my fate", "know how this ends", "made my peace", "it doesn't matter anymore", "how it ends"), you MUST ask a follow-up to explore it BEFORE moving to a new topic. Reference what they said: "You mentioned accepting your fate—what do you mean by that?" Do not skip over red flags.
- Keep it short, casual, conversational. One question at a time.
- The patient may deflect direct inquiries. Use indirect, natural probes.

TOPIC HIERARCHY (use this to guide your questions):
1. General/Mood - overall state, how things feel (Sadness, Crying)
2. Physical - sleep, energy, fatigue, appetite (Sleep, Energy, Fatigue, Appetite)
3. Motivation - interest, pleasure, drive (Loss of Interest, Loss of Pleasure)
4. Self/Outlook - self-view, future, guilt, worth (Pessimism, Worthlessness, Self-Dislike, Guilt)
5. Cognitive - focus, decisions (Concentration, Indecisiveness)
6. Behavioral - restlessness, irritability (Agitation, Irritability)

Flow: Open general -> branch into topics based on what they say -> drill down within topic. Connect questions: "You said X - what about Y?"
"""


def _get_client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def get_next_question(
    conversation: list[dict[str, str]],
    probed_indices: set[int],
) -> str:
    """
    Get the next question using topic-based, conversational flow.
    General -> specific. Questions relate to each other.
    """
    if not DEEPSEEK_API_KEY:
        return _fallback_question(conversation, probed_indices)

    conv_text = "\n".join(
        f"{m['role']}: {m['message']}" for m in conversation
    )
    covered_topics = _infer_covered_topics(conversation)
    next_topic = get_next_topic(covered_topics)

    topic_info = ""
    if next_topic:
        t = get_topic_by_id(next_topic)
        topic_info = f"\n\nNext topic to explore: {t['name']} (symptoms: {', '.join(t['symptoms'])}). Keywords for this topic: {', '.join(t['keywords'][:8])}."

    user_content = f"""Conversation so far:
{conv_text}

Topics already explored: {covered_topics or 'none yet'}.{topic_info}

Generate the NEXT question. It should:
- If they said something concerning or ambiguous (fate, how it ends, peace, doesn't matter), ask a follow-up FIRST. E.g. "You said you've accepted your fate—what does that mean to you?"
- Otherwise, flow naturally from what they just said (reference it if relevant)
- Explore the next topic or drill into the current one
- Be indirect and conversational
- ONE question only

Output ONLY the question:"""

    client = _get_client()
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": _PROBER_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        max_tokens=100,
        temperature=0.4,
    )
    text = (resp.choices[0].message.content or "").strip()
    return text.split("\n")[0].strip().strip('"') or "How have things been for you lately?"


def _infer_covered_topics(conversation: list[dict[str, str]]) -> list[str]:
    """Infer which topics have been touched based on user questions."""
    from src.topic_hierarchy import TOPICS

    covered = []
    for msg in conversation:
        if msg.get("role") != "user":
            continue
        q = (msg.get("message") or "").lower()
        for topic in TOPICS:
            if topic.name in covered:
                continue
            if any(kw in q for kw in topic.keywords):
                covered.append(topic.name)
                break
            for oq in topic.opening_questions + topic.follow_up_questions:
                if oq.lower()[:25] in q or (len(q) > 20 and q[:30] in oq.lower()[:40]):
                    covered.append(topic.name)
                    break
    return covered


def _fallback_question(
    conversation: list[dict[str, str]],
    probed_indices: set[int],
) -> str:
    """Fallback when no API key - use topic-based static questions."""
    from src.topic_hierarchy import get_next_topic, get_topic_by_id

    covered = _infer_covered_topics(conversation)
    next_topic = get_next_topic(covered)

    if not next_topic:
        return "Is there anything else you'd like to share about how you've been feeling?"

    t = get_topic_by_id(next_topic)
    if t.get("opening_questions"):
        return t["opening_questions"][0]
    return f"How have things been for you in terms of {t['name'].lower()}?"
