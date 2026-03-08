"""Prober Agent - topic-based, conversational questioning. General -> specific, questions relate to each other."""

from __future__ import annotations

from openai import OpenAI

from src.agents.evidence_memory import retrieve_relevant_patient_evidence
from src.agents.risk_router import classify_cluster, next_cluster_question
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from src.topic_hierarchy import (
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

_DISALLOWED_DIRECT_TERMS = ("depress", "depression", "mental health")

_RED_FLAG_PATTERNS: tuple[str, ...] = (
    "accepted my fate",
    "know how this ends",
    "ik how this ends",
    "made my peace",
    "it doesn't matter anymore",
    "doesnt matter anymore",
    "end it",
    "better without me",
    "not wanting to be here",
    "want to die",
    "wanna die",
)


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
    asked_questions = _recent_user_questions(conversation)
    last_patient_message = _last_assistant_message(conversation)
    cluster_decision = classify_cluster(conversation)
    active_cluster = cluster_decision["cluster"]

    # Force a direct red-flag follow-up before topic switching.
    red_flag_follow_up = _red_flag_follow_up(last_patient_message, asked_questions)
    if red_flag_follow_up:
        return red_flag_follow_up

    # Risk-first routing: keep focus on high-risk clusters before broad topic sweep.
    if active_cluster in ("AcuteSafety", "HopelessWorthless"):
        cluster_q = _normalize_question(next_cluster_question(active_cluster, asked_questions))
        if _is_usable_question(cluster_q, asked_questions):
            return cluster_q

    if not DEEPSEEK_API_KEY:
        return _fallback_question(conversation, probed_indices, asked_questions)

    conv_text = "\n".join(
        f"{m['role']}: {m['message']}" for m in conversation
    )
    covered_topics = _infer_covered_topics(conversation)
    next_topic = get_next_topic(covered_topics)

    topic_info = ""
    retrieval_query = ""
    if next_topic:
        t = get_topic_by_id(next_topic)
        topic_info = f"\n\nNext topic to explore: {t['name']} (symptoms: {', '.join(t['symptoms'])}). Keywords for this topic: {', '.join(t['keywords'][:8])}."
        retrieval_query = " ".join(
            [
                t["name"],
                " ".join(t["symptoms"]),
                " ".join(t["keywords"][:8]),
                last_patient_message,
            ]
        ).strip()
    evidence_snippets = retrieve_relevant_patient_evidence(
        conversation,
        retrieval_query or last_patient_message or "overall functioning",
        top_k=3,
    )

    user_content = f"""Conversation so far:
{conv_text}

Topics already explored: {covered_topics or 'none yet'}.{topic_info}
Active risk cluster: {active_cluster} (confidence {cluster_decision['confidence']:.2f}, reason: {cluster_decision['rationale'] or 'n/a'}).
Previously asked questions (avoid repeating these): {asked_questions or 'none'}.
Most relevant prior patient evidence: {evidence_snippets or 'none yet'}.

Generate the NEXT question. It should:
- If they said something concerning or ambiguous (fate, how it ends, peace, doesn't matter), ask a follow-up FIRST. E.g. "You said you've accepted your fate—what does that mean to you?"
- If active cluster is AcuteSafety or HopelessWorthless, stay in that cluster instead of switching to generic physical topics
- Otherwise, flow naturally from what they just said (reference it if relevant)
- Ground your question in the most relevant prior evidence if possible
- Explore the next topic or drill into the current one
- Be indirect and conversational
- ONE question only
- Do not ask directly about depression or mental health
- Do not repeat a previously asked question

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
    candidate = _normalize_question(text)
    if _is_usable_question(candidate, asked_questions):
        return candidate
    return _fallback_question(conversation, probed_indices, asked_questions)


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
    asked_questions: list[str] | None = None,
) -> str:
    """Fallback when no API key - use topic-based static questions."""
    from src.topic_hierarchy import TOPICS, get_next_topic, get_topic_by_id

    _ = probed_indices  # Reserved for future scoring-aware probing.
    asked_questions = asked_questions or _recent_user_questions(conversation)
    active_cluster = classify_cluster(conversation)["cluster"]

    if active_cluster in ("AcuteSafety", "HopelessWorthless"):
        candidate = _normalize_question(next_cluster_question(active_cluster, asked_questions))
        if _is_usable_question(candidate, asked_questions):
            return candidate

    covered = _infer_covered_topics(conversation)
    next_topic = get_next_topic(covered)

    if not next_topic:
        return "Is there anything else you'd like to share about how you've been feeling?"

    t = get_topic_by_id(next_topic)
    for q in t.get("opening_questions", []):
        candidate = _normalize_question(q)
        if _is_usable_question(candidate, asked_questions):
            return candidate
    for topic in TOPICS:
        for q in topic.follow_up_questions:
            candidate = _normalize_question(q)
            if _is_usable_question(candidate, asked_questions):
                return candidate
    return f"How have things been for you in terms of {t['name'].lower()}?"

def _normalize_question(text: str) -> str:
    """Keep one clean, single-line question."""
    line = (text or "").split("\n")[0].strip().strip('"').strip()
    if not line:
        return ""
    if not line.endswith("?"):
        line = f"{line.rstrip('.!')}?"
    return line


def _recent_user_questions(conversation: list[dict[str, str]]) -> list[str]:
    """Return already-asked user questions (normalized)."""
    return [
        _normalize_question(m.get("message", ""))
        for m in conversation
        if m.get("role") == "user" and _normalize_question(m.get("message", ""))
    ]


def _last_assistant_message(conversation: list[dict[str, str]]) -> str:
    """Return the latest patient response text."""
    for msg in reversed(conversation):
        if msg.get("role") == "assistant":
            return (msg.get("message") or "").strip()
    return ""


def _red_flag_follow_up(last_message: str, asked_questions: list[str]) -> str:
    """Return a mandatory follow-up if latest response contains critical language."""
    if not last_message:
        return ""
    lowered = last_message.lower()
    matched = next((p for p in _RED_FLAG_PATTERNS if p in lowered), "")
    if not matched:
        return ""
    question = _normalize_question(
        f"You mentioned '{matched}' earlier - what did you mean by that?"
    )
    if _is_usable_question(question, asked_questions):
        return question
    return ""


def _is_usable_question(question: str, asked_questions: list[str]) -> bool:
    """Validate quality constraints and repetition."""
    if not question or len(question) < 10:
        return False
    lowered = question.lower()
    if any(term in lowered for term in _DISALLOWED_DIRECT_TERMS):
        return False
    asked_set = {q.lower() for q in asked_questions}
    return lowered not in asked_set
