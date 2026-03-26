#!/usr/bin/env python3
"""Analyze per-symptom and per-group probing caps from saved interactions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.prober import infer_question_targets  # noqa: E402
from src.config import get_run_policy  # noqa: E402


def _load_interaction(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        raise ValueError(f"Empty interaction file: {path}")
    return data[0]


def _analyze_conversation(conversation: list[dict[str, str]], run_policy: dict) -> dict:
    symptom_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    topic_counts: dict[str, int] = {}
    forced_switches = 0
    prev_group = ""

    for msg in conversation:
        if msg.get("role") != "user":
            continue
        question = msg.get("message", "")
        route = infer_question_targets(question)
        group = str(route.get("group") or "")
        topic = str(route.get("topic") or "")
        symptoms = [str(s) for s in route.get("symptoms", [])]

        if prev_group and group and group != prev_group:
            forced_switches += 1
        prev_group = group or prev_group

        if topic:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        if group:
            group_counts[group] = group_counts.get(group, 0) + 1
        for symptom in symptoms:
            symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1

    max_q_symptom = int(run_policy.get("max_questions_per_symptom", 3))
    max_q_group = int(run_policy.get("max_questions_per_group", 6))
    symptom_violations = {k: v for k, v in symptom_counts.items() if v > max_q_symptom}
    group_violations = {k: v for k, v in group_counts.items() if v > max_q_group}

    return {
        "total_turns": len(conversation) // 2,
        "symptom_counts": symptom_counts,
        "group_counts": group_counts,
        "topic_counts": topic_counts,
        "forced_group_switches": forced_switches,
        "cap_config": {
            "max_questions_per_symptom": max_q_symptom,
            "max_questions_per_group": max_q_group,
        },
        "symptom_cap_violations": symptom_violations,
        "group_cap_violations": group_violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze cap behavior in saved interactions.")
    parser.add_argument("--persona", type=int, required=True, help="Persona ID, e.g. 8 or 9.")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID, e.g. 1 or 2.")
    parser.add_argument(
        "--submission-root",
        type=Path,
        default=ROOT / "outputs" / "submission",
        help="Root submission path containing persona folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "newVersion15Mar" / "reports" / "probe_caps_report.json",
        help="Output report path.",
    )
    args = parser.parse_args()

    interaction_path = (
        args.submission_root
        / f"persona{args.persona}"
        / f"run{args.run_id}"
        / f"interactions_run{args.run_id}.json"
    )
    if not interaction_path.exists():
        raise FileNotFoundError(f"Interaction file not found: {interaction_path}")

    item = _load_interaction(interaction_path)
    run_policy = get_run_policy(args.run_id)
    report = {
        "persona": args.persona,
        "run_id": args.run_id,
        "interaction_file": str(interaction_path),
        "analysis": _analyze_conversation(item.get("conversation", []), run_policy),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
