#!/usr/bin/env python3
"""Batch-run personas 1-8 and report trend + latency diagnostics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _trend_report(scores: dict[int, int]) -> dict:
    ordered = [scores[i] for i in range(1, 9)]
    violations: list[tuple[int, int, int, int]] = []
    for i in range(1, 8):
        left = ordered[i - 1]
        right = ordered[i]
        if left < right:
            violations.append((i, i + 1, left, right))
    return {
        "ordered_scores": ordered,
        "monotonic_violations": violations,
        "is_monotonic_nonincreasing": len(violations) == 0,
    }


def _latency_report(interactions: list[dict]) -> dict:
    turns_by_persona: dict[int, int] = {}
    for item in interactions:
        pid = int(item["LLM"])
        conv = item.get("conversation", [])
        turns_by_persona[pid] = len(conv) // 2
    turns = list(turns_by_persona.values())
    avg_turns = (sum(turns) / len(turns)) if turns else 0.0
    return {
        "turns_by_persona": turns_by_persona,
        "avg_turns": round(avg_turns, 2),
        "min_turns": min(turns) if turns else 0,
        "max_turns": max(turns) if turns else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/evaluate personas 1-8 trend and latency")
    parser.add_argument("--run-id", type=str, default="98", help="Run id to execute/analyze")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dev"), help="Output root")
    parser.add_argument("--mock", action="store_true", help="Use mock personas")
    parser.add_argument("--skip-run", action="store_true", help="Analyze existing outputs only")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "run.py",
        "--personas",
        "1-8",
        "--run",
        args.run_id,
        "--output-dir",
        str(args.output_dir),
    ]
    if args.mock:
        cmd.append("--mock")

    if not args.skip_run:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    run_dir = args.output_dir / f"run{args.run_id}"
    results_path = run_dir / f"results_run{args.run_id}.json"
    interactions_path = run_dir / f"interactions_run{args.run_id}.json"

    if not results_path.exists() or not interactions_path.exists():
        raise FileNotFoundError(f"Missing run outputs under {run_dir}")

    results = _load_json(results_path)
    interactions = _load_json(interactions_path)

    scores = {int(item["LLM"]): int(item["bdi-score"]) for item in results}
    trend = _trend_report(scores)
    latency = _latency_report(interactions)

    report = {
        "run_id": args.run_id,
        "scores": scores,
        "trend": trend,
        "latency": latency,
    }
    report_path = run_dir / "sweep_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
