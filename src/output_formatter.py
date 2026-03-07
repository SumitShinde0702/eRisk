"""Output formatter for eRisk 2026 Task 1 submission JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.bdi_mapper import validate_key_symptoms


def format_interactions(
    persona_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Format interactions for submission.
    Each item: {"LLM": "1", "conversation": [{"role": "user"|"assistant", "message": "..."}, ...]}
    """
    out = []
    for pr in persona_results:
        llm_id = str(pr.get("llm_id", pr.get("LLM", "")))
        conv = pr.get("conversation", [])
        out.append({"LLM": llm_id, "conversation": conv})
    return out


def format_results(
    persona_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Format classification results for submission.
    Each item: {"LLM": "1", "bdi-score": int, "key-symptoms": [...]}
    """
    out = []
    for pr in persona_results:
        llm_id = str(pr.get("llm_id", pr.get("LLM", "")))
        bdi = int(pr.get("bdi_score", pr.get("bdi-score", 0)))
        key_syms = pr.get("key_symptoms", pr.get("key-symptoms", []))
        key_syms = validate_key_symptoms(key_syms)
        out.append({"LLM": llm_id, "bdi-score": bdi, "key-symptoms": key_syms})
    return out


def save_run(
    persona_results: list[dict[str, Any]],
    run_id: int,
    output_dir: Path,
    manual: bool = False,
) -> tuple[Path, Path]:
    """
    Save interactions and results JSON files for a run.

    Returns:
        (interactions_path, results_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "manual_" if manual else ""
    interactions_path = output_dir / f"{prefix}interactions_run{run_id}.json"
    results_path = output_dir / f"{prefix}results_run{run_id}.json"

    interactions = format_interactions(persona_results)
    results = format_results(persona_results)

    with open(interactions_path, "w", encoding="utf-8") as f:
        json.dump(interactions, f, indent=2, ensure_ascii=False)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return interactions_path, results_path


def save_run_outputs(
    output_dir: Path,
    run_id: str,
    interactions: list[dict[str, Any]],
    results: list[dict[str, Any]],
    manual_prefix: str = "",
) -> tuple[Path, Path]:
    """
    Save pre-formatted interactions and results JSON files.

    Returns:
        (interactions_path, results_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    interactions_path = output_dir / f"{manual_prefix}interactions_run{run_id}.json"
    results_path = output_dir / f"{manual_prefix}results_run{run_id}.json"

    with open(interactions_path, "w", encoding="utf-8") as f:
        json.dump(interactions, f, indent=2, ensure_ascii=False)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return interactions_path, results_path
