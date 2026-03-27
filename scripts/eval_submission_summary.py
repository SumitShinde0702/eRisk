#!/usr/bin/env python3
"""Summarize all submission result files under outputs/submission."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _fmt(val: float | int | str) -> str:
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def _print_table(headers: list[str], rows: list[list[object]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(_fmt(cell)))
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(_fmt(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize results_run*.json across submission outputs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/submission"),
        help="Root folder containing submission outputs.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    candidates = sorted(
        [
            p
            for p in root.rglob("*.json")
            if p.name.startswith("results_run") or p.name.startswith("manual_results_run")
        ]
    )
    if not candidates:
        print(f"No results files found under: {root}")
        return

    per_file_rows: list[list[object]] = []
    score_by_llm: dict[str, list[int]] = defaultdict(list)

    for path in candidates:
        try:
            data = _load_json(path)
        except Exception:
            continue
        if not isinstance(data, list) or not data:
            continue

        scores: list[int] = []
        file_scores: dict[str, int] = {}
        key_counts: list[int] = []
        for item in data:
            llm = str(item.get("LLM", "")).strip()
            if not llm:
                continue
            try:
                score = int(item.get("bdi-score", 0))
            except Exception:
                score = 0
            scores.append(score)
            file_scores[llm] = score
            score_by_llm[llm].append(score)
            key_symptoms = item.get("key-symptoms", [])
            key_counts.append(len(key_symptoms) if isinstance(key_symptoms, list) else 0)

        if not scores:
            continue

        rel = path.relative_to(root)
        profile = str(rel.parent.parent) if rel.parent.parent != Path(".") else "root"
        run_name = rel.parent.name
        p6 = file_scores.get("6", "")
        p7 = file_scores.get("7", "")
        d67 = (p7 - p6) if isinstance(p6, int) and isinstance(p7, int) else ""

        per_file_rows.append(
            [
                profile,
                run_name,
                rel.name,
                len(scores),
                min(scores),
                max(scores),
                _safe_float([float(x) for x in scores]),
                statistics.pstdev(scores) if len(scores) > 1 else 0.0,
                _safe_float([float(x) for x in key_counts]),
                p6,
                p7,
                d67,
            ]
        )

    if not per_file_rows:
        print("No valid result payloads found.")
        return

    per_file_rows.sort(key=lambda r: (str(r[0]), str(r[1]), str(r[2])))

    print("\n=== File-Level Summary ===")
    _print_table(
        [
            "profile",
            "run",
            "file",
            "n_personas",
            "min",
            "max",
            "mean",
            "std",
            "avg_key_symptoms",
            "p6",
            "p7",
            "d67",
        ],
        per_file_rows,
    )

    llm_rows: list[list[object]] = []
    for llm, vals in sorted(score_by_llm.items(), key=lambda x: x[0]):
        llm_rows.append(
            [
                llm,
                len(vals),
                min(vals),
                max(vals),
                _safe_float([float(x) for x in vals]),
                statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            ]
        )

    print("\n=== Per-LLM Aggregate Across Files ===")
    _print_table(["LLM", "n_files", "min", "max", "mean", "std"], llm_rows)


if __name__ == "__main__":
    main()
