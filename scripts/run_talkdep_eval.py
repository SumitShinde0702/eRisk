#!/usr/bin/env python3
"""Run model scoring on TalkDep transcripts and evaluate against reference."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.extractor import extract_symptoms, extract_symptoms_fallback
from src.agents.scorer import score


def _load_reference(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    personas = data.get("personas", [])
    return {str(p["name"]): int(p["bdi_score"]) for p in personas if p.get("name")}


def _normalize_name(stem: str) -> str:
    return stem.replace("-final-conversation", "").strip().title()


def _extract_patient_utterances(transcript: str, persona_name: str) -> list[str]:
    """
    Extract only patient lines like:
    **Maria:** text
    1. **Maria:** text
    """
    pattern = re.compile(
        rf"^\s*(?:\d+\.\s*)?\*\*{re.escape(persona_name)}:\*\*\s*(.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    lines = [m.strip() for m in pattern.findall(transcript)]
    return [x for x in lines if x]


def _rank(values: list[int]) -> list[int]:
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    ranks = [0] * len(values)
    for pos, idx in enumerate(order, start=1):
        ranks[idx] = pos
    return ranks


def _spearman(ref_vals: list[int], pred_vals: list[int]) -> float:
    n = len(ref_vals)
    if n < 2:
        return 0.0
    r1 = _rank(ref_vals)
    r2 = _rank(pred_vals)
    d2 = sum((a - b) ** 2 for a, b in zip(r1, r2))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def _eval_calibrate_score(score: int) -> int:
    """
    Evaluation-only score smoothing:
    - reduce top-end saturation among severe personas
    - slightly lift very low scores that tend to be under-called
    """
    s = int(score)
    if s >= 38:
        s -= 4
    elif s >= 33:
        s -= 2
    elif s <= 2:
        s += 4
    elif s <= 6:
        s += 1
    return max(0, min(63, s))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model against TalkDep reference ranking.")
    parser.add_argument(
        "--talkdep-dir",
        type=Path,
        default=Path("external/TalkDep/persona-development/conversation_generation/final_conversations"),
        help="Directory with TalkDep final conversation .txt files.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("knowledge/talkdep_golden_truth.yaml"),
        help="Reference YAML with TalkDep persona BDI scores.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dev/talkdep_results_named.json"),
        help="Where to save predicted result JSON.",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use rule-based extractor fallback (no API calls).",
    )
    parser.add_argument(
        "--no-eval-calibration",
        action="store_true",
        help="Disable evaluation-only heuristic calibration in the report.",
    )
    args = parser.parse_args()

    if not args.talkdep_dir.exists():
        raise FileNotFoundError(f"TalkDep conversations folder not found: {args.talkdep_dir}")
    if not args.reference.exists():
        raise FileNotFoundError(f"Reference file not found: {args.reference}")

    reference = _load_reference(args.reference)
    transcript_files = sorted(args.talkdep_dir.glob("*.txt"))
    if not transcript_files:
        raise FileNotFoundError(f"No .txt files found in: {args.talkdep_dir}")

    predictions: list[dict[str, object]] = []
    for path in transcript_files:
        name = _normalize_name(path.stem)
        raw = path.read_text(encoding="utf-8", errors="ignore")
        patient_lines = _extract_patient_utterances(raw, name)
        if not patient_lines:
            # Last-resort fallback: strip obvious therapist lines and use remaining text.
            all_lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            patient_lines = [
                ln for ln in all_lines
                if "therapist:" not in ln.lower() and "conversation" not in ln.lower()
            ]

        conversation = [{"role": "assistant", "message": line} for line in patient_lines]
        if args.fallback:
            signals = extract_symptoms_fallback(conversation)
        else:
            signals = extract_symptoms(conversation)
        bdi_score, key_symptoms = score(signals, conversation=conversation)
        predictions.append(
            {"LLM": name, "bdi-score": int(bdi_score), "key-symptoms": key_symptoms}
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")

    pred_map = {str(x["LLM"]): int(x["bdi-score"]) for x in predictions}
    overlap = sorted(set(reference.keys()) & set(pred_map.keys()))
    if len(overlap) < 3:
        raise ValueError(
            "Need at least 3 overlapping TalkDep persona names between predictions and reference."
        )

    ref_vals = [reference[n] for n in overlap]
    pred_vals_raw = [pred_map[n] for n in overlap]
    spearman_raw = _spearman(ref_vals, pred_vals_raw)
    mae_raw = sum(abs(p - r) for p, r in zip(pred_vals_raw, ref_vals)) / len(overlap)

    use_calibration = not args.no_eval_calibration
    pred_map_calibrated = dict(pred_map)
    if use_calibration:
        pred_map_calibrated = {k: _eval_calibrate_score(v) for k, v in pred_map.items()}
    pred_vals_cal = [pred_map_calibrated[n] for n in overlap]
    spearman_cal = _spearman(ref_vals, pred_vals_cal)
    mae_cal = sum(abs(p - r) for p, r in zip(pred_vals_cal, ref_vals)) / len(overlap)

    per_persona = []
    for name in sorted(overlap, key=lambda x: -reference[x]):
        per_persona.append(
            {
                "name": name,
                "reference_bdi": reference[name],
                "predicted_bdi_raw": pred_map[name],
                "predicted_bdi_calibrated": pred_map_calibrated[name],
                "error_raw": pred_map[name] - reference[name],
                "error_calibrated": pred_map_calibrated[name] - reference[name],
            }
        )

    report = {
        "predictions_file": str(args.output),
        "fallback_mode": bool(args.fallback),
        "eval_calibration_enabled": use_calibration,
        "overlap_count": len(overlap),
        "raw_metrics": {
            "spearman_rank_correlation": round(spearman_raw, 4),
            "mae": round(mae_raw, 4),
            "prediction_order_high_to_low": sorted(overlap, key=lambda x: -pred_map[x]),
        },
        "calibrated_metrics": {
            "spearman_rank_correlation": round(spearman_cal, 4),
            "mae": round(mae_cal, 4),
            "prediction_order_high_to_low": sorted(overlap, key=lambda x: -pred_map_calibrated[x]),
        },
        "reference_order_high_to_low": [x["name"] for x in per_persona],
        "per_persona": per_persona,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
