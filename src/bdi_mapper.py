"""BDI-II symptom list and mapping logic for eRisk 2026 Task 1."""

# Official 21 BDI-II symptoms (from eRisk 2019 Task 3)
# Each maps to BDI-II item number (1-21) and has an indirect question for probing
BDI_SYMPTOMS: list[str] = [
    "Sadness",
    "Pessimism",
    "Past Failure",
    "Loss of Pleasure",
    "Guilty Feelings",
    "Punishment Feelings",
    "Self-Dislike",
    "Self-Criticalness",
    "Suicidal Thoughts or Wishes",
    "Crying",
    "Agitation",
    "Loss of Interest",
    "Indecisiveness",
    "Worthlessness",
    "Loss of Energy",
    "Changes in Sleeping Pattern",
    "Irritability",
    "Changes in Appetite",
    "Concentration Difficulty",
    "Tiredness or Fatigue",
    "Loss of Interest in Sex",
]

# BDI-II question bank: indirect questions to probe each symptom
# Order matches BDI_SYMPTOMS (index 0 = Sadness, etc.)
# Avoid direct mental-health probes; personas deflect those
BDI_QUESTION_BANK: list[str] = [
    "How have you been feeling lately?",
    "How do you see things going for you in the near future?",
    "How do you feel about how things have gone for you recently?",
    "Been doing anything you enjoy recently?",
    "Is there anything weighing on your mind that you wish you could change?",
    "Do you ever feel like things are going wrong for a reason?",
    "How do you feel about yourself these days?",
    "Are you harder on yourself than you used to be?",
    # Item 9 (Suicidal): infer only from context - never ask directly
    "What's been on your mind when you're alone?",
    "Have you found yourself getting emotional more easily lately?",
    "Do you feel restless or on edge sometimes?",
    "Are you still interested in the things you used to care about?",
    "Do you find it harder to make decisions than before?",
    "Do you feel like you're still useful or needed?",
    "Do you feel like you have enough energy for your usual day?",
    "How have you been sleeping lately?",
    "Have you been more irritable or short-tempered than usual?",
    "How's your appetite been?",
    "Is it harder to focus or concentrate than it used to be?",
    "Do you get tired or worn out more easily than before?",
    "Have you noticed any changes in what you're interested in lately?",
]

# High-yield symptoms to probe first (often most informative)
HIGH_YIELD_SYMPTOM_INDICES: list[int] = [0, 11, 15, 19, 14]  # Sadness, Loss of Interest, Sleep, Fatigue, Energy

# BDI score cutoffs
BDI_MINIMAL: tuple[int, int] = (0, 13)
BDI_MILD: tuple[int, int] = (14, 19)
BDI_MODERATE: tuple[int, int] = (20, 28)
BDI_SEVERE: tuple[int, int] = (29, 63)

BDI_MAX_SCORE: int = 63
BDI_MAX_KEY_SYMPTOMS: int = 4


def get_symptom_by_index(index: int) -> str | None:
    """Get symptom name by 0-based index."""
    if 0 <= index < len(BDI_SYMPTOMS):
        return BDI_SYMPTOMS[index]
    return None


def get_question_by_index(index: int) -> str | None:
    """Get probing question by 0-based index."""
    if 0 <= index < len(BDI_QUESTION_BANK):
        return BDI_QUESTION_BANK[index]
    return None


def symptom_to_index(symptom: str) -> int | None:
    """Get 0-based index for a symptom name (exact match)."""
    try:
        return BDI_SYMPTOMS.index(symptom)
    except ValueError:
        return None


def validate_key_symptoms(symptoms: list[str]) -> list[str]:
    """Validate and truncate key symptoms to max 4, using exact BDI names."""
    valid = [s for s in symptoms if s in BDI_SYMPTOMS]
    return valid[:BDI_MAX_KEY_SYMPTOMS]


def get_severity_label(score: int) -> str:
    """Get severity label for a BDI score."""
    if score <= BDI_MINIMAL[1]:
        return "minimal"
    if score <= BDI_MILD[1]:
        return "mild"
    if score <= BDI_MODERATE[1]:
        return "moderate"
    return "severe"
