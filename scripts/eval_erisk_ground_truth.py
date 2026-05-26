#!/usr/bin/env python3
"""Reproduce eRisk Task 1 official metrics (ADODL, DCHR, ASHR) for INSALyon-2 submissions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "task1-llms/golden-data/patients_data.jsonl"
SYMPTOM_MAP = ROOT / "task1-llms/golden-data/symptom_mappings.json"
DEFAULT_SUBMISSION = ROOT / "outputs/submission"


def _load_norm_map() -> dict[str, str]:
    data = json.loads(SYMPTOM_MAP.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for canon, variants in data.get("symptom_mappings", {}).items():
        out[canon.lower()] = canon
        for v in variants:
            out[str(v).strip().lower()] = canon
    return out


def _norm_symptom(text: str, norm_map: dict[str, str]) -> str:
    return norm_map.get((text or "").strip().lower(), (text or "").strip())


def _category(score: int) -> str:
    if score <= 9:
        return "minimal"
    if score <= 18:
        return "mild"
    if score <= 29:
        return "moderate"
    return "severe"


def _load_prediction(submission_dir: Path, persona_id: int, run_id: int) -> dict | None:
    path = submission_dir / f"persona{persona_id}" / f"run{run_id}" / f"results_run{run_id}.json"
    if not path.exists():
        return None
    item = json.loads(path.read_text(encoding="utf-8"))[0]
    return {
        "persona_id": int(item.get("LLM", persona_id)),
        "bdi": int(item.get("bdi-score", item.get("bdi_score", 0))),
        "symptoms": list(item.get("key-symptoms", item.get("key_symptoms", []))),
    }


def evaluate_run(
    submission_dir: Path,
    run_id: int,
    *,
    persona_ids: range | list[int] | None = None,
) -> dict:
    """
    Official Task 1 metrics per preliminary results report:
    - ADODL: mean((63 - |ADL - EDL|) / 63)
    - DCHR: fraction with correct BDI severity category
    - ASHR: mean fraction of reference key symptoms matched (up to 4)
    Ground-truth patient_id matches submission persona id / JSON LLM field directly.
    """
    golden = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    gt_by_id = {int(row["patient_id"]): row for row in golden}
    norm_map = _load_norm_map()

    if persona_ids is None:
        persona_ids = range(1, 21)

    adodls: list[float] = []
    dchrs: list[float] = []
    ashrs: list[float] = []
    rows: list[dict] = []

    for persona_id in persona_ids:
        pred = _load_prediction(submission_dir, persona_id, run_id)
        ref = gt_by_id.get(persona_id)
        if pred is None or ref is None:
            continue
        adl = int(ref["bdi_score"])
        edl = int(pred["bdi"])
        adodls.append((63 - abs(adl - edl)) / 63)
        dchrs.append(1.0 if _category(adl) == _category(edl) else 0.0)
        gt_syms = [_norm_symptom(s, norm_map) for s in ref.get("patient_key_symptoms", [])[:4]]
        pred_syms = [_norm_symptom(s, norm_map) for s in pred["symptoms"][:4]]
        hits = sum(1 for s in pred_syms if s in gt_syms)
        ashrs.append(hits / max(len(gt_syms), 1))
        rows.append(
            {
                "persona_id": persona_id,
                "patient_name": ref.get("patient_name", ""),
                "adl": adl,
                "edl": edl,
                "abs_error": abs(adl - edl),
                "symptom_hits": hits,
                "symptom_total": len(gt_syms),
            }
        )

    n = len(adodls)
    mae = sum(r["abs_error"] for r in rows) / n if n else float("nan")
    rmse = math.sqrt(sum(r["abs_error"] ** 2 for r in rows) / n) if n else float("nan")
    return {
        "run_id": run_id,
        "n": n,
        "ADODL": sum(adodls) / n if n else float("nan"),
        "DCHR": sum(dchrs) / n if n else float("nan"),
        "ASHR": sum(ashrs) / n if n else float("nan"),
        "MAE": mae,
        "RMSE": rmse,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--personas", type=str, default="1-19", help="Range like 1-19 or 1-20")
    args = parser.parse_args()

    if "-" in args.personas:
        start, end = map(int, args.personas.split("-", 1))
        persona_ids = list(range(start, end + 1))
    else:
        persona_ids = [int(args.personas)]

    print(f"Submission: {args.submission_dir}")
    print(f"Personas: {persona_ids[0]}..{persona_ids[-1]} (n={len(persona_ids)} requested)")
    print()
    for run_id in args.runs:
        result = evaluate_run(args.submission_dir, run_id, persona_ids=persona_ids)
        print(
            f"Run {run_id}: n={result['n']}  "
            f"ADODL={result['ADODL']:.4f}  DCHR={result['DCHR']:.4f}  ASHR={result['ASHR']:.4f}  "
            f"MAE={result['MAE']:.2f}  RMSE={result['RMSE']:.2f}"
        )


if __name__ == "__main__":
    main()
