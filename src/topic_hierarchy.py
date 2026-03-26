"""Topic hierarchy and symptom taxonomy for BDI-II probing."""

from __future__ import annotations

from dataclasses import dataclass

from src.bdi_mapper import BDI_SYMPTOMS


@dataclass
class Topic:
    """A topic that maps to one or more BDI symptoms."""

    name: str
    keywords: list[str]
    symptom_names: list[str]
    opening_questions: list[str]
    follow_up_questions: list[str]
    related_topics: list[str]


# Topic hierarchy: Physical, Motivation, Mood/Emotional, Self/Outlook, Cognitive, Behavioral
TOPICS: list[Topic] = [
    Topic(
        name="General_Mood",
        keywords=["feel", "feeling", "mood", "sad", "down", "low", "okay", "fine", "great", "terrible"],
        symptom_names=["Sadness", "Loss of Pleasure"],
        opening_questions=[
            "How have things been for you lately?",
            "What's been on your mind recently?",
            "How are you doing overall?",
        ],
        follow_up_questions=[
            "When you say that, what does a typical day feel like for you?",
            "Has that been consistent or does it come and go?",
        ],
        related_topics=["Physical", "Motivation", "Self_Outlook"],
    ),
    Topic(
        name="Physical",
        keywords=[
            "sleep", "tired", "fatigue", "energy", "appetite", "eating",
            "wake", "insomnia", "exhausted", "drained", "rest",
        ],
        symptom_names=[
            "Changes in Sleeping Pattern",
            "Tiredness or Fatigue",
            "Loss of Energy",
            "Changes in Appetite",
        ],
        opening_questions=[
            "How have you been sleeping?",
            "Do you feel like you have enough energy for your usual day?",
            "How's your appetite been?",
        ],
        follow_up_questions=[
            "You mentioned sleep - has that affected your energy during the day?",
            "When you're tired, does it get in the way of things you used to do?",
            "Any changes in how much you're eating or what you feel like eating?",
        ],
        related_topics=["Motivation", "General_Mood"],
    ),
    Topic(
        name="Motivation",
        keywords=[
            "interest", "enjoy", "hobbies", "motivation", "bother", "effort",
            "care", "nothing", "pointless", "bored",
        ],
        symptom_names=["Loss of Interest", "Loss of Pleasure", "Loss of Interest in Sex"],
        opening_questions=[
            "Been doing anything you enjoy lately?",
            "Are you still into the things you used to care about?",
            "What do you do to unwind or have fun?",
        ],
        follow_up_questions=[
            "When you try to do those things, how does it feel?",
            "Has it been harder to get motivated for stuff?",
            "Anything you've stopped doing that you used to do?",
        ],
        related_topics=["Physical", "General_Mood", "Self_Outlook"],
    ),
    Topic(
        name="Self_Outlook",
        keywords=[
            "worth", "failure", "guilty", "blame", "future", "hopeless",
            "useful", "disappointed", "critical", "punish", "deserve",
        ],
        symptom_names=[
            "Pessimism",
            "Past Failure",
            "Worthlessness",
            "Self-Dislike",
            "Self-Criticalness",
            "Guilty Feelings",
            "Punishment Feelings",
        ],
        opening_questions=[
            "How do you see things going for you in the near future?",
            "How do you feel about yourself these days?",
            "How do you feel about how things have gone for you recently?",
        ],
        follow_up_questions=[
            "Are you harder on yourself than you used to be?",
            "Do you feel like you're still useful or needed?",
            "Is there anything weighing on your mind that you wish you could change?",
        ],
        related_topics=["General_Mood", "Motivation"],
    ),
    Topic(
        name="Cognitive",
        keywords=[
            "focus", "concentrate", "decide", "decisions", "mind", "wander",
            "forget", "thinking", "clear",
        ],
        symptom_names=["Indecisiveness", "Concentration Difficulty"],
        opening_questions=[
            "Is it harder to focus or concentrate than it used to be?",
            "Do you find it harder to make decisions than before?",
        ],
        follow_up_questions=[
            "When you need to make a decision, what happens?",
            "Does your mind wander when you're trying to do something?",
        ],
        related_topics=["Physical", "Self_Outlook"],
    ),
    Topic(
        name="Behavioral_Emotional",
        keywords=[
            "cry", "emotional", "irritable", "restless", "agitation",
            "snap", "edge", "tears", "upset",
        ],
        symptom_names=["Crying", "Agitation", "Irritability"],
        opening_questions=[
            "Have you found yourself getting emotional more easily lately?",
            "Have you been more irritable or short-tempered than usual?",
            "Do you feel restless or on edge sometimes?",
        ],
        follow_up_questions=[
            "When that happens, what does it look like?",
            "Has that affected how you interact with people?",
        ],
        related_topics=["General_Mood", "Self_Outlook"],
    ),
]

# Four groups requested for route balancing (BDI scoring remains unchanged).
SYMPTOM_GROUPS: tuple[str, ...] = ("Affective", "Executive", "Somatic", "Cognitive")

_SYMPTOM_TO_GROUP_OVERRIDES: dict[str, str] = {
    # Affective (mood / emotion)
    "Sadness": "Affective",
    "Pessimism": "Affective",
    "Loss of Pleasure": "Affective",
    "Guilty Feelings": "Affective",
    "Punishment Feelings": "Affective",
    "Self-Dislike": "Affective",
    "Self-Criticalness": "Affective",
    "Suicidal Thoughts or Wishes": "Affective",
    "Crying": "Affective",
    "Worthlessness": "Affective",
    # Executive (organizational / attention / decision)
    "Indecisiveness": "Executive",
    "Concentration Difficulty": "Executive",
    "Past Failure": "Executive",
    # Somatic (sleep / appetite / energy / psychomotor)
    "Loss of Energy": "Somatic",
    "Changes in Sleeping Pattern": "Somatic",
    "Changes in Appetite": "Somatic",
    "Tiredness or Fatigue": "Somatic",
    "Agitation": "Somatic",
    "Irritability": "Somatic",
    # Cognitive (thought/speech/self-processing)
    "Loss of Interest": "Cognitive",
    "Loss of Interest in Sex": "Cognitive",
}

TOPIC_GROUP: dict[str, str] = {
    "General_Mood": "Affective",
    "Physical": "Somatic",
    "Motivation": "Cognitive",
    "Self_Outlook": "Affective",
    "Cognitive": "Executive",
    "Behavioral_Emotional": "Somatic",
}

TOPIC_BY_NAME: dict[str, Topic] = {t.name: t for t in TOPICS}
TOPIC_ORDER: list[str] = [t.name for t in TOPICS]


def get_next_topic(covered_topics: list[str]) -> str | None:
    """Return the next topic to explore (first not yet covered)."""
    covered = set(covered_topics)
    for name in TOPIC_ORDER:
        if name not in covered:
            return name
    return None


def get_topic_by_id(topic_id: str) -> dict:
    """Return topic as dict for prober."""
    t = TOPIC_BY_NAME.get(topic_id)
    if not t:
        return {"name": topic_id, "keywords": [], "symptoms": [], "opening_questions": []}
    return {
        "name": t.name,
        "keywords": t.keywords,
        "symptoms": t.symptom_names,
        "opening_questions": t.opening_questions,
    }


def get_symptom_group(symptom_name: str) -> str:
    """Return one of the four groups for a BDI symptom."""
    return _SYMPTOM_TO_GROUP_OVERRIDES.get(symptom_name, "Affective")


def get_group_symptoms(group_name: str) -> list[str]:
    """Return all symptoms that map to one group."""
    return [s for s in BDI_SYMPTOMS if get_symptom_group(s) == group_name]


def get_topic_group(topic_name: str) -> str:
    """Return the group label for a topic."""
    return TOPIC_GROUP.get(topic_name, "Affective")


def get_symptom_to_group_map() -> dict[str, str]:
    """Return a full symptom-to-group mapping."""
    return {s: get_symptom_group(s) for s in BDI_SYMPTOMS}


def get_group_to_symptoms_map() -> dict[str, list[str]]:
    """Return group to symptoms mapping."""
    return {group: get_group_symptoms(group) for group in SYMPTOM_GROUPS}


def get_topic_symptom_indices(topic_name: str) -> list[int]:
    """Get 0-based BDI symptom indices for a topic."""
    t = TOPIC_BY_NAME.get(topic_name)
    if not t:
        return []
    return [i for i, s in enumerate(BDI_SYMPTOMS) if s in t.symptom_names]


def get_all_topic_keywords() -> dict[str, list[str]]:
    """Topic name -> keywords for response matching."""
    return {t.name: t.keywords for t in TOPICS}


def get_probed_topics_from_conversation(conversation: list[dict[str, str]]) -> set[str]:
    """Infer which topics have been probed from user questions."""
    probed: set[str] = set()
    for msg in conversation:
        if msg.get("role") != "user":
            continue
        text = (msg.get("message") or "").lower()
        for topic in TOPICS:
            if topic.name in probed:
                continue
            if any(kw in text for kw in topic.keywords):
                probed.add(topic.name)
                break
            for q in topic.opening_questions + topic.follow_up_questions:
                if q.lower()[:25] in text or text[:30] in q.lower()[:30]:
                    probed.add(topic.name)
                    break
    return probed
