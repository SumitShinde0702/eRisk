"""Prober Agent - topic-based, conversational questioning. General -> specific, questions relate to each other."""

from __future__ import annotations

from openai import OpenAI

from src.agents.evidence_memory import retrieve_relevant_patient_evidence
from src.agents.risk_router import (
    classify_cluster,
    next_acute_ladder_question,
    next_cluster_question,
)
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
import src.agents.interview_banks as interview_banks
from src.topic_hierarchy import (
    TOPICS,
    get_next_topic,
    get_symptom_group,
    get_topic_by_id,
    get_topic_group,
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
    *,
    risk_buffer: list[dict] | None = None,
    run_policy: dict | None = None,
    symptom_question_counts: dict[str, int] | None = None,
    group_question_counts: dict[str, int] | None = None,
    topic_question_counts: dict[str, int] | None = None,
    group_screen_counts: dict[str, int] | None = None,
    drilldown_counts: dict[str, int] | None = None,
    symptom_signals: dict[str, int] | None = None,
    route_meta: dict | None = None,
) -> str:
    """
    Get the next question using topic-based, conversational flow.
    General -> specific. Questions relate to each other.
    """
    run_policy = run_policy or {}
    symptom_question_counts = symptom_question_counts or {}
    group_question_counts = group_question_counts or {}
    topic_question_counts = topic_question_counts or {}
    group_screen_counts = group_screen_counts or {}
    drilldown_counts = drilldown_counts or {}
    symptom_signals = symptom_signals or {}
    asked_questions = _recent_user_questions(conversation)
    last_patient_message = _last_assistant_message(conversation)
    cluster_decision = classify_cluster(conversation, risk_buffer=risk_buffer)
    active_cluster = cluster_decision["cluster"]
    route_limits = _routing_constraints(
        run_policy,
        symptom_question_counts=symptom_question_counts,
        group_question_counts=group_question_counts,
        topic_question_counts=topic_question_counts,
    )

    # Highest-priority policy: acute safety ladder (intent -> plan -> timeline -> means -> protective factors).
    ladder_q = _normalize_question(
        next_acute_ladder_question(
            conversation,
            asked_questions,
            risk_buffer=risk_buffer,
        )
    )
    if _is_usable_question(ladder_q, asked_questions):
        return ladder_q

    # Force a direct red-flag follow-up before topic switching.
    red_flag_follow_up = _red_flag_follow_up(last_patient_message, asked_questions)
    if red_flag_follow_up:
        return red_flag_follow_up

    # If an ambiguous-risk bridge was asked and response suggests self-erasure/non-existence, escalate directly.
    bridge_escalation_q = _bridge_to_ladder_follow_up(conversation, asked_questions)
    if bridge_escalation_q:
        return bridge_escalation_q

    # Bridge ambiguous withdrawal language before topic switching.
    ambiguous_bridge_q = _ambiguous_risk_bridge(last_patient_message, asked_questions)
    if ambiguous_bridge_q:
        return ambiguous_bridge_q

    # Risk-first routing: keep focus on high-risk clusters before broad topic sweep.
    if active_cluster in ("AcuteSafety", "HopelessWorthless"):
        cluster_q = _normalize_question(next_cluster_question(active_cluster, asked_questions))
        if _is_usable_question(cluster_q, asked_questions):
            return cluster_q

    # Hard group screen (3+ per group) then symptom drilldown (extractor signal > 0).
    asked_norm = {_norm_q_match(q) for q in asked_questions}
    if bool(run_policy.get("group_screen_enabled", True)):
        if not interview_banks.group_screen_complete(group_screen_counts):
            out = interview_banks.next_screen_question_and_meta(group_screen_counts, asked_norm)
            if out:
                q_raw, meta = out
                q = _normalize_question(q_raw)
                if _is_usable_question(q, asked_questions):
                    if route_meta is not None:
                        route_meta.clear()
                        route_meta.update(meta)
                    return q
        elif bool(run_policy.get("symptom_drilldown_enabled", True)):
            max_drill_total = run_policy.get("max_drilldown_questions_total")
            max_drill = int(max_drill_total) if max_drill_total is not None else None
            out = interview_banks.next_drilldown_question_and_meta(
                symptom_signals,
                drilldown_counts,
                asked_norm,
                max_total=max_drill,
            )
            if out:
                q_raw, meta = out
                q = _normalize_question(q_raw)
                if _is_usable_question(q, asked_questions):
                    if route_meta is not None:
                        route_meta.clear()
                        route_meta.update(meta)
                    return q

    if not DEEPSEEK_API_KEY:
        return _fallback_question(
            conversation,
            probed_indices,
            asked_questions,
            risk_buffer=risk_buffer,
            blocked_topics=route_limits["blocked_topics"],
            preferred_group=route_limits["preferred_group"],
        )

    conv_text = "\n".join(
        f"{m['role']}: {m['message']}" for m in conversation
    )
    covered_topics = _infer_covered_topics(conversation)
    next_topic = _select_next_topic(
        covered_topics,
        blocked_topics=route_limits["blocked_topics"],
        preferred_group=route_limits["preferred_group"],
    )
    if not next_topic:
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
    evidence_source: list[dict[str, str]] = conversation
    if risk_buffer:
        evidence_source = [
            {"role": "assistant", "message": str(item.get("assistant_message", ""))}
            for item in risk_buffer
            if item.get("assistant_message")
        ] or conversation
    evidence_snippets = retrieve_relevant_patient_evidence(
        evidence_source,
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
        temperature=_prober_temperature(run_policy),
    )
    text = (resp.choices[0].message.content or "").strip()
    candidate = _normalize_question(text)
    candidate = _de_lead_question(candidate, conversation)
    if _is_usable_question(candidate, asked_questions):
        return candidate
    return _fallback_question(
        conversation,
        probed_indices,
        asked_questions,
        risk_buffer=risk_buffer,
        blocked_topics=route_limits["blocked_topics"],
        preferred_group=route_limits["preferred_group"],
    )


def _infer_covered_topics(conversation: list[dict[str, str]]) -> list[str]:
    """Infer which topics have been touched based on user questions."""
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
    *,
    risk_buffer: list[dict] | None = None,
    blocked_topics: set[str] | None = None,
    preferred_group: str | None = None,
) -> str:
    """Fallback when no API key - use topic-based static questions."""
    from src.topic_hierarchy import get_next_topic, get_topic_by_id

    _ = probed_indices  # Reserved for future scoring-aware probing.
    asked_questions = asked_questions or _recent_user_questions(conversation)
    blocked_topics = blocked_topics or set()
    active_cluster = classify_cluster(conversation, risk_buffer=risk_buffer)["cluster"]

    if active_cluster in ("AcuteSafety", "HopelessWorthless"):
        candidate = _normalize_question(next_cluster_question(active_cluster, asked_questions))
        if _is_usable_question(candidate, asked_questions):
            return candidate

    covered = _infer_covered_topics(conversation)
    next_topic = _select_next_topic(
        covered,
        blocked_topics=blocked_topics,
        preferred_group=preferred_group,
    )
    if not next_topic:
        next_topic = get_next_topic(covered)

    if not next_topic:
        return "Is there anything else you'd like to share about how you've been feeling?"

    t = get_topic_by_id(next_topic)
    for q in t.get("opening_questions", []):
        candidate = _normalize_question(q)
        if _is_usable_question(candidate, asked_questions):
            return candidate
    for topic in TOPICS:
        if topic.name in blocked_topics:
            continue
        if preferred_group and get_topic_group(topic.name) != preferred_group:
            continue
        for q in topic.follow_up_questions:
            candidate = _normalize_question(q)
            if _is_usable_question(candidate, asked_questions):
                return candidate
    return f"How have things been for you in terms of {t['name'].lower()}?"

def _norm_q_match(q: str) -> str:
    return " ".join((q or "").lower().split())


def _normalize_question(text: str) -> str:
    """Keep one clean, single-line question."""
    line = (text or "").split("\n")[0].strip().strip('"').strip()
    if not line:
        return ""
    if not line.endswith("?"):
        line = f"{line.rstrip('.!')}?"
    return line


def _select_next_topic(
    covered_topics: list[str],
    *,
    blocked_topics: set[str],
    preferred_group: str | None = None,
) -> str | None:
    covered = set(covered_topics)
    for topic in TOPICS:
        if topic.name in covered or topic.name in blocked_topics:
            continue
        if preferred_group and get_topic_group(topic.name) != preferred_group:
            continue
        return topic.name
    if preferred_group:
        for topic in TOPICS:
            if topic.name in blocked_topics:
                continue
            if get_topic_group(topic.name) == preferred_group:
                return topic.name
    return None


def _routing_constraints(
    run_policy: dict,
    *,
    symptom_question_counts: dict[str, int],
    group_question_counts: dict[str, int],
    topic_question_counts: dict[str, int],
) -> dict[str, object]:
    max_q_symptom = int(run_policy.get("max_questions_per_symptom", 3))
    max_q_group = int(run_policy.get("max_questions_per_group", 6))
    switch_on_saturation = bool(run_policy.get("group_switch_on_saturation", True))

    saturated_symptoms = {
        symptom for symptom, count in symptom_question_counts.items() if int(count) >= max_q_symptom
    }
    saturated_groups = {
        group for group, count in group_question_counts.items() if int(count) >= max_q_group
    }

    blocked_topics: set[str] = set()
    for topic in TOPICS:
        topic_group = get_topic_group(topic.name)
        if topic_group in saturated_groups:
            blocked_topics.add(topic.name)
            continue
        if topic.symptom_names and all(sym in saturated_symptoms for sym in topic.symptom_names):
            blocked_topics.add(topic.name)

    # Repetition guard: if same topic keeps getting selected, force switching.
    for topic_name, count in topic_question_counts.items():
        if int(count) >= max(max_q_symptom, 3):
            blocked_topics.add(topic_name)

    preferred_group: str | None = None
    if switch_on_saturation and saturated_groups:
        candidate_counts = {
            group: int(group_question_counts.get(group, 0))
            for group in ("Affective", "Executive", "Somatic", "Cognitive")
            if group not in saturated_groups
        }
        if candidate_counts:
            preferred_group = min(candidate_counts, key=lambda x: candidate_counts[x])

    return {"blocked_topics": blocked_topics, "preferred_group": preferred_group}


def infer_question_targets(question: str, route_meta: dict | None = None) -> dict[str, object]:
    """Infer likely topic/group/symptoms a question is targeting."""
    from src.bdi_mapper import BDI_QUESTION_BANK, get_symptom_by_index

    bank_meta = interview_banks.match_screen_or_drilldown_meta(question)
    if bank_meta:
        out = dict(bank_meta)
        if route_meta:
            out.update(route_meta)
        return out

    text = (question or "").lower().strip()
    best_topic = ""
    best_score = 0
    for topic in TOPICS:
        score = 0
        score += sum(2 for kw in topic.keywords if kw in text)
        score += sum(1 for q in topic.opening_questions if q.lower()[:25] in text)
        score += sum(1 for q in topic.follow_up_questions if q.lower()[:25] in text)
        if score > best_score:
            best_score = score
            best_topic = topic.name

    symptoms: list[str] = []
    if best_topic:
        symptoms.extend(get_topic_by_id(best_topic).get("symptoms", []))
    for idx, canonical in enumerate(BDI_QUESTION_BANK):
        canonical_l = canonical.lower()
        if canonical_l[:25] in text or text[:30] in canonical_l[:30]:
            symptom = get_symptom_by_index(idx)
            if symptom and symptom not in symptoms:
                symptoms.append(symptom)

    group = get_topic_group(best_topic) if best_topic else ""
    if not group and symptoms:
        group = get_symptom_group(symptoms[0])
    out = {"topic": best_topic, "group": group, "symptoms": symptoms[:4]}
    if route_meta:
        out.update(route_meta)
    return out


def _recent_assistant_messages(conversation: list[dict[str, str]], max_count: int = 3) -> list[str]:
    msgs = [
        (m.get("message") or "").strip().lower()
        for m in conversation
        if m.get("role") == "assistant" and (m.get("message") or "").strip()
    ]
    return msgs[-max_count:]


def _de_lead_question(question: str, conversation: list[dict[str, str]]) -> str:
    """
    Guard against assumptive/attribution prompts not grounded in recent patient text.
    """
    ql = question.lower()
    loaded_terms = ("nothing matters", "feel pointless", "pointless", "better off without you")
    if any(t in ql for t in loaded_terms):
        recent_blob = " ".join(_recent_assistant_messages(conversation, max_count=3))
        if not any(t in recent_blob for t in loaded_terms):
            return "What has felt most difficult for you lately?"

    attribution_markers = ("you said", "you mentioned", "when you say", "as you said")
    if not any(m in ql for m in attribution_markers):
        return question

    recent = _recent_assistant_messages(conversation, max_count=3)
    if not recent:
        return "Can you tell me a bit more about how that has been affecting you lately?"

    # If quoted span exists, require lexical grounding in recent text.
    if "'" in question or '"' in question:
        import re

        quoted = re.findall(r"['\"]([^'\"]{3,80})['\"]", question)
        for span in quoted:
            span_l = span.strip().lower()
            if span_l and any(span_l in msg for msg in recent):
                return question
        return "Can you tell me more about what that experience has felt like for you?"

    # Without explicit quotes, allow only if core content is grounded.
    content_terms = [
        t for t in ql.replace("?", "").replace(",", " ").split()
        if len(t) > 4 and t not in {"mentioned", "feeling", "lately", "about", "which", "could", "would"}
    ]
    if content_terms and any(any(t in msg for t in content_terms[:3]) for msg in recent):
        return question

    return "Can you tell me more about how that has been affecting your day-to-day?"


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


def _ambiguous_risk_bridge(last_message: str, asked_questions: list[str]) -> str:
    """Clarify withdrawal/futility language that can mask acute risk."""
    if not last_message:
        return ""
    lowered = last_message.lower()
    ambiguous_patterns = (
        "disappear",
        "why bother",
        "just existing",
        "just exist",
        "invisible",
        "ghost in the background",
        "nothing feels real",
    )
    if not any(p in lowered for p in ambiguous_patterns):
        return ""
    question = _normalize_question(
        "When you say that, is it more about wanting to withdraw from everything, or feeling like you don't want to be here?"
    )
    if _is_usable_question(question, asked_questions):
        return question
    return ""


def _bridge_to_ladder_follow_up(conversation: list[dict[str, str]], asked_questions: list[str]) -> str:
    """Escalate bridge responses into explicit intent clarification when risk wording persists."""
    if len(conversation) < 2:
        return ""
    last_patient = _last_assistant_message(conversation).lower()
    last_user = ""
    for msg in reversed(conversation):
        if msg.get("role") == "user":
            last_user = (msg.get("message") or "").lower()
            break
    if not last_user:
        return ""
    if "withdraw from everything" not in last_user and "don't want to be here" not in last_user:
        return ""

    escalation_markers = (
        "both",
        "don't want to be here",
        "dont want to be here",
        "not be here",
        "disappear",
        "no one would notice",
        "waiting for the timer",
        "timer to go off",
        "end of something",
    )
    if not any(m in last_patient for m in escalation_markers):
        return ""

    question = _normalize_question(
        "When you say that, are you having thoughts about ending your life?"
    )
    if _is_usable_question(question, asked_questions):
        return question
    return ""


def _prober_temperature(run_policy: dict) -> float:
    """Lower temperature for submission-style policies to improve stability."""
    if "prober_temperature" in run_policy:
        try:
            return max(0.0, min(1.0, float(run_policy["prober_temperature"])))
        except (TypeError, ValueError):
            pass
    policy_name = str(run_policy.get("name", "")).strip().lower()
    if policy_name in {"balanced", "high_recall", "high_precision"}:
        return 0.25
    return 0.4


def _is_usable_question(question: str, asked_questions: list[str]) -> bool:
    """Validate quality constraints and repetition."""
    if not question or len(question) < 10:
        return False
    lowered = question.lower()
    if any(term in lowered for term in _DISALLOWED_DIRECT_TERMS):
        return False
    asked_set = {q.lower() for q in asked_questions}
    return lowered not in asked_set
