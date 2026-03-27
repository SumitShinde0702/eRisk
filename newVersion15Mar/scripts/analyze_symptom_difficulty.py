#!/usr/bin/env python3
"""Analyze symptom difficulty using extractor disagreement and transcript evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.extractor import extract_symptoms, extract_symptoms_fallback  # noqa: E402
from src.bdi_mapper import BDI_SYMPTOMS  # noqa: E402
from src.config import DEEPSEEK_API_KEY  # noqa: E402


def _load_reference(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    personas = data.get("personas", [])
    return {str(p["name"]): int(p["bdi_score"]) for p in personas if p.get("name")}


def _normalize_name(stem: str) -> str:
    return stem.replace("-final-conversation", "").strip().title()


def _extract_patient_utterances(transcript: str, persona_name: str) -> list[str]:
    pattern = re.compile(
        rf"^\s*(?:\d+\.\s*)?\*\*{re.escape(persona_name)}:\*\*\s*(.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    lines = [m.strip() for m in pattern.findall(transcript)]
    return [x for x in lines if x]


def _safe_score_map(raw: dict[str, int] | None) -> dict[str, int]:
    data = raw or {}
    out: dict[str, int] = {}
    for s in BDI_SYMPTOMS:
        out[s] = int(max(0, min(3, int(data.get(s, 0)))))
    return out


def _initial_aggregates() -> dict[str, dict[str, float]]:
    return {
        s: {
            "n": 0,
            "fallback_nonzero": 0,
            "model_nonzero": 0,
            "fallback_sum": 0,
            "model_sum": 0,
            "disagree_count": 0,
            "severe_miss_model": 0,     # fallback >=2, model ==0
            "severe_miss_fallback": 0,  # model >=2, fallback ==0
        }
        for s in BDI_SYMPTOMS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Symptom difficulty report from TalkDep transcripts.")
    parser.add_argument(
        "--talkdep-dir",
        type=Path,
        default=ROOT / "external" / "TalkDep" / "persona-development" / "conversation_generation" / "final_conversations",
        help="Folder with TalkDep final conversation text files.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "knowledge" / "talkdep_golden_truth.yaml",
        help="Reference YAML with TalkDep persona BDI scores.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "newVersion15Mar" / "reports" / "symptom_difficulty_report.json",
        help="Output JSON report.",
    )
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Use fallback extractor only (skip model extractor).",
    )
    args = parser.parse_args()

    if not args.talkdep_dir.exists():
        raise FileNotFoundError(f"TalkDep folder not found: {args.talkdep_dir}")
    transcript_files = sorted(args.talkdep_dir.glob("*.txt"))
    if not transcript_files:
        raise FileNotFoundError(f"No .txt files found in {args.talkdep_dir}")

    reference = _load_reference(args.reference) if args.reference.exists() else {}
    use_model = bool(DEEPSEEK_API_KEY) and not args.fallback_only

    aggregates = _initial_aggregates()
    per_persona: list[dict[str, Any]] = []

    for path in transcript_files:
        persona_name = _normalize_name(path.stem)
        text = path.read_text(encoding="utf-8", errors="ignore")
        patient_lines = _extract_patient_utterances(text, persona_name)
        if not patient_lines:
            continue

        conv = [{"role": "assistant", "message": ln} for ln in patient_lines]
        fallback = _safe_score_map(extract_symptoms_fallback(conv))
        if use_model:
            model = _safe_score_map(extract_symptoms(conv))
        else:
            model = dict(fallback)

        persona_disagreement = 0
        for s in BDI_SYMPTOMS:
            ag = aggregates[s]
            ag["n"] += 1
            fv = fallback[s]
            mv = model[s]
            ag["fallback_sum"] += fv
            ag["model_sum"] += mv
            if fv > 0:
                ag["fallback_nonzero"] += 1
            if mv > 0:
                ag["model_nonzero"] += 1
            if fv != mv:
                ag["disagree_count"] += 1
                persona_disagreement += 1
            if fv >= 2 and mv == 0:
                ag["severe_miss_model"] += 1
            if mv >= 2 and fv == 0:
                ag["severe_miss_fallback"] += 1

        per_persona.append(
            {
                "name": persona_name,
                "reference_bdi": reference.get(persona_name),
                "disagreement_symptom_count": persona_disagreement,
                "fallback_total": sum(fallback.values()),
                "model_total": sum(model.values()),
            }
        )

    symptom_rows = []
    for s in BDI_SYMPTOMS:
        ag = aggregates[s]
        n = max(int(ag["n"]), 1)
        row = {
            "symptom": s,
            "n": int(ag["n"]),
            "fallback_prevalence": round(ag["fallback_nonzero"] / n, 4),
            "model_prevalence": round(ag["model_nonzero"] / n, 4),
            "fallback_avg_score": round(ag["fallback_sum"] / n, 4),
            "model_avg_score": round(ag["model_sum"] / n, 4),
            "disagreement_rate": round(ag["disagree_count"] / n, 4),
            "severe_miss_model_rate": round(ag["severe_miss_model"] / n, 4),
            "severe_miss_fallback_rate": round(ag["severe_miss_fallback"] / n, 4),
        }
        symptom_rows.append(row)

    hardest = sorted(
        symptom_rows,
        key=lambda x: (
            -x["disagreement_rate"],
            -max(x["severe_miss_model_rate"], x["severe_miss_fallback_rate"]),
            -abs(x["fallback_avg_score"] - x["model_avg_score"]),
        ),
    )

    report = {
        "model_extractor_used": use_model,
        "api_key_present": bool(DEEPSEEK_API_KEY),
        "fallback_only": bool(args.fallback_only),
        "persona_count": len(per_persona),
        "hardest_symptoms_top10": hardest[:10],
        "sleep_symptom_detail": next(
            (x for x in symptom_rows if x["symptom"] == "Changes in Sleeping Pattern"),
            None,
        ),
        "agitation_symptom_detail": next(
            (x for x in symptom_rows if x["symptom"] == "Agitation"),
            None,
        ),
        "symptom_stats": symptom_rows,
        "per_persona": sorted(per_persona, key=lambda x: -x["disagreement_symptom_count"]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["hardest_symptoms_top10"], indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
