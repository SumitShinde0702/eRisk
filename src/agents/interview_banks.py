"""YAML-backed group screen + symptom drilldown question banks (BDI-II, 1A taxonomy)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.bdi_mapper import BDI_SYMPTOMS
from src.config import PROJECT_ROOT
from src.topic_hierarchy import get_symptom_group

_GROUP_YAML = PROJECT_ROOT / "knowledge" / "group_screen_questions.yaml"
_DRILL_YAML = PROJECT_ROOT / "knowledge" / "symptom_drilldown_questions.yaml"


@lru_cache(maxsize=1)
def _load_group_screen() -> dict[str, Any]:
    if not _GROUP_YAML.exists():
        return {}
    return yaml.safe_load(_GROUP_YAML.read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=1)
def _load_drilldown() -> dict[str, Any]:
    if not _DRILL_YAML.exists():
        return {}
    return yaml.safe_load(_DRILL_YAML.read_text(encoding="utf-8")) or {}


def get_min_questions_per_group_screen() -> int:
    data = _load_group_screen()
    return int(data.get("min_questions_per_group", 3))


def get_group_order() -> list[str]:
    data = _load_group_screen()
    order = data.get("group_order") or ["Affective", "Executive", "Somatic", "Cognitive"]
    return [str(x) for x in order]


def group_screen_complete(group_screen_counts: dict[str, int]) -> bool:
    """True when each group has at least min screen questions."""
    min_q = get_min_questions_per_group_screen()
    order = get_group_order()
    for g in order:
        if int(group_screen_counts.get(g, 0)) < min_q:
            return False
    return True


def next_screen_group(group_screen_counts: dict[str, int]) -> str:
    """Pick group with fewest screen questions so far (tie-break by group_order)."""
    order = get_group_order()
    min_q = min(int(group_screen_counts.get(g, 0)) for g in order)
    for g in order:
        if int(group_screen_counts.get(g, 0)) == min_q:
            return g
    return order[0]


def next_screen_question_and_meta(
    group_screen_counts: dict[str, int],
    asked_normalized: set[str],
) -> tuple[str, dict[str, Any]] | None:
    """
    Return next screen question text and route meta, or None if screen phase complete.
    asked_normalized: lowercased stripped question strings already asked.
    """
    if group_screen_complete(group_screen_counts):
        return None
    data = _load_group_screen()
    groups = data.get("groups") or {}
    g = next_screen_group(group_screen_counts)
    entries = (groups.get(g) or {}).get("screen_questions") or []
    if not entries:
        return None
    idx = int(group_screen_counts.get(g, 0))
    entry = entries[idx % len(entries)]
    text = str(entry.get("text", "")).strip()
    targets = [str(x) for x in (entry.get("targets") or [])]
    if not text:
        return None
    norm = _norm_q(text)
    if norm in asked_normalized:
        # Skip duplicate: try next entry in pool
        for off in range(1, len(entries)):
            entry2 = entries[(idx + off) % len(entries)]
            t2 = str(entry2.get("text", "")).strip()
            if t2 and _norm_q(t2) not in asked_normalized:
                text = t2
                targets = [str(x) for x in (entry2.get("targets") or [])]
                break
    meta = {
        "phase": "screen",
        "screen_group": g,
        "group": g,
        "symptoms": targets[:6],
        "topic": "",
    }
    return text, meta


def _norm_q(q: str) -> str:
    return " ".join((q or "").lower().split())


def match_screen_or_drilldown_meta(question: str) -> dict[str, Any] | None:
    """Match a question string to bank entries for route counting."""
    nq = _norm_q(question)
    data = _load_group_screen()
    for gname, gdata in (data.get("groups") or {}).items():
        for ent in (gdata.get("screen_questions") or []):
            t = str(ent.get("text", "")).strip()
            if t and _norm_q(t) == nq:
                return {
                    "phase": "screen",
                    "screen_group": gname,
                    "group": gname,
                    "symptoms": [str(x) for x in (ent.get("targets") or [])][:6],
                    "topic": "",
                }
    drill = _load_drilldown()
    for sym, qs in (drill.get("symptoms") or {}).items():
        for line in qs or []:
            if isinstance(line, str) and _norm_q(line) == nq:
                return {
                    "phase": "drilldown",
                    "screen_group": "",
                    "group": get_symptom_group(sym),
                    "symptoms": [sym],
                    "topic": "",
                }
    return None


def max_drilldown_per_symptom() -> int:
    return int(_load_drilldown().get("max_questions_per_symptom", 2))


def next_drilldown_question_and_meta(
    symptom_signals: dict[str, int],
    drilldown_counts: dict[str, int],
    asked_normalized: set[str],
    *,
    max_total: int | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """
    Ask follow-ups for symptoms in order of current signal strength, then BDI order.
    Up to max_drilldown_per_symptom questions per symptom.
    """
    drill = _load_drilldown()
    bank = drill.get("symptoms") or {}
    max_per = max_drilldown_per_symptom()
    if max_total is not None and sum(int(drilldown_counts.get(s, 0)) for s in BDI_SYMPTOMS) >= max_total:
        return None

    def sort_key(s: str) -> tuple[int, int]:
        score = int(symptom_signals.get(s, 0))
        idx = BDI_SYMPTOMS.index(s) if s in BDI_SYMPTOMS else 99
        return (-score, idx)

    ordered = sorted(BDI_SYMPTOMS, key=sort_key)
    # Only drill into symptoms the extractor already flags (>0).
    ordered = [s for s in ordered if int(symptom_signals.get(s, 0)) > 0]
    if not ordered:
        return None

    for sym in ordered:
        qs = bank.get(sym) or []
        if not qs:
            continue
        used = int(drilldown_counts.get(sym, 0))
        if used >= max_per:
            continue
        if used < len(qs):
            text = str(qs[used]).strip()
        else:
            continue
        if not text:
            continue
        if _norm_q(text) in asked_normalized:
            continue
        meta = {
            "phase": "drilldown",
            "screen_group": "",
            "group": get_symptom_group(sym),
            "symptoms": [sym],
            "topic": "",
        }
        return text, meta
    return None
