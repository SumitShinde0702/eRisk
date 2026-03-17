#!/usr/bin/env python3
"""Evaluate ranking agreement against TalkDep external reference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_reference(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    personas = data.get("personas", [])
    return {str(p["name"]): int(p["bdi_score"]) for p in personas if p.get("name")}


def _rank(values: list[int]) -> list[int]:
    """Return rank positions (1=highest) with stable tie handling."""
    sorted_idx = sorted(range(len(values)), key=lambda i: (-values[i], i))
    rank = [0] * len(values)
    for pos, idx in enumerate(sorted_idx, start=1):
        rank[idx] = pos
    return rank


def _spearman(ref: list[int], pred: list[int]) -> float:
    n = len(ref)
    if n < 2:
        return 0.0
    r1 = _rank(ref)
    r2 = _rank(pred)
    d2 = sum((a - b) ** 2 for a, b in zip(r1, r2))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate outputs against TalkDep ranking reference")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to results JSON with LLM and bdi-score",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("knowledge/talkdep_golden_truth.yaml"),
        help="Path to TalkDep reference YAML",
    )
    args = parser.parse_args()

    results = _load_json(args.results)
    reference = _load_reference(args.reference)

    pred_scores: dict[str, int] = {}
    for item in results:
        name = str(item.get("LLM", "")).strip()
        if not name:
            continue
        pred_scores[name] = int(item.get("bdi-score", 0))

    common = sorted(set(reference.keys()) & set(pred_scores.keys()))
    if len(common) < 3:
        raise ValueError(
            "Need at least 3 overlapping persona names between results and reference. "
            "TalkDep reference uses names (e.g., Maria, Linda), not eRisk numeric IDs."
        )

    ref_vals = [reference[n] for n in common]
    pred_vals = [pred_scores[n] for n in common]
    spearman = _spearman(ref_vals, pred_vals)

    ordered_ref = sorted(common, key=lambda n: -reference[n])
    ordered_pred = sorted(common, key=lambda n: -pred_scores[n])

    report = {
        "overlap_count": len(common),
        "overlap_names": common,
        "spearman_rank_correlation": round(spearman, 4),
        "reference_order_high_to_low": ordered_ref,
        "prediction_order_high_to_low": ordered_pred,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
