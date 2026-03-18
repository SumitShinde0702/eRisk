#!/usr/bin/env python3
"""Run architecture ablations (memory/components) on personas and compare outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import DEEPSEEK_API_KEY, get_run_policy  # noqa: E402
from src.orchestrator import run_conversation  # noqa: E402
from src.persona_client import get_persona_client  # noqa: E402
import src.agents.extractor as extractor  # noqa: E402
import src.agents.prober as prober  # noqa: E402
import src.agents.risk_router as risk_router  # noqa: E402
import src.agents.scorer as scorer  # noqa: E402
import src.agents.stopper as stopper  # noqa: E402
import src.orchestrator as orchestrator  # noqa: E402


def _offline_memory_retrieval(
    conversation: list[dict[str, str]],
    query_text: str,
    top_k: int = 3,
) -> list[str]:
    """Simple token-overlap retrieval to avoid heavy embedding runtime in offline mode."""
    query_terms = {t for t in query_text.lower().split() if len(t) > 2}
    if not query_terms:
        return []
    patient_messages = [
        (m.get("message") or "").strip()
        for m in conversation
        if m.get("role") == "assistant" and (m.get("message") or "").strip()
    ]
    ranked: list[tuple[int, str]] = []
    for msg in patient_messages:
        msg_terms = {t for t in msg.lower().split() if len(t) > 2}
        ranked.append((len(query_terms & msg_terms), msg))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [m for score, m in ranked[:top_k] if score > 0]


@contextmanager
def _patched() -> Any:
    """Context manager for temporary monkey patches."""
    patches: list[tuple[object, str, Any]] = []

    def set_attr(obj: object, name: str, value: Any) -> None:
        old = getattr(obj, name)
        patches.append((obj, name, old))
        setattr(obj, name, value)

    try:
        yield set_attr
    finally:
        for obj, name, old in reversed(patches):
            setattr(obj, name, old)


def _parse_persona_ids(spec: str) -> list[int]:
    if "-" in spec:
        start, end = [int(x) for x in spec.split("-", maxsplit=1)]
        return list(range(start, end + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _trend(scores: dict[int, int]) -> dict[str, Any]:
    ordered_ids = sorted(scores.keys())
    ordered_scores = [scores[i] for i in ordered_ids]
    violations = []
    for i in range(len(ordered_scores) - 1):
        left = ordered_scores[i]
        right = ordered_scores[i + 1]
        if left < right:
            violations.append(
                {
                    "left_persona": ordered_ids[i],
                    "right_persona": ordered_ids[i + 1],
                    "left_score": left,
                    "right_score": right,
                }
            )
    return {
        "ordered_personas": ordered_ids,
        "ordered_scores": ordered_scores,
        "monotonic_nonincreasing": len(violations) == 0,
        "violations": violations,
    }


def _build_variants() -> list[dict[str, Any]]:
    return [
        {"name": "baseline"},
        {"name": "no_memory"},
        {"name": "no_template_risk"},
        {"name": "no_stopper"},
        {"name": "fallback_extractor"},
        {"name": "no_acute_calibration"},
    ]


def _run_variant(
    variant_name: str,
    *,
    persona_ids: list[int],
    use_mock: bool,
    use_ai_mock: bool,
    offline: bool,
    disable_memory_embeddings: bool,
    disable_template_embeddings: bool,
    quick_live: bool,
    run_policy: dict[str, Any],
) -> dict[str, Any]:
    scores: dict[int, int] = {}
    turns: dict[int, int] = {}
    conversations: dict[int, list[dict[str, str]]] = {}

    with _patched() as patch:
        policy = dict(run_policy)
        if quick_live:
            # Reduce number of turns for live smoke ablations.
            policy["min_exchanges_before_stop"] = min(int(policy.get("min_exchanges_before_stop", 10)), 2)
            policy["required_acute_ladder_steps"] = min(int(policy.get("required_acute_ladder_steps", 4)), 1)
            policy["risk_buffer_size"] = min(int(policy.get("risk_buffer_size", 6)), 2)

        if variant_name == "no_memory":
            patch(prober, "retrieve_relevant_patient_evidence", lambda *_a, **_k: [])
        elif variant_name == "no_template_risk":
            patch(orchestrator, "get_top_template_matches", lambda *_a, **_k: [])
            patch(orchestrator, "compute_turn_risk_score", lambda *_a, **_k: 0.0)
            policy["risk_buffer_size"] = 0
        elif variant_name == "no_stopper":
            patch(stopper, "should_stop", lambda *_a, **_k: (False, "disabled"))
            patch(orchestrator, "should_stop", lambda *_a, **_k: (False, "disabled"))
        elif variant_name == "fallback_extractor":
            patch(orchestrator, "extract_symptoms", extractor.extract_symptoms_fallback)
        elif variant_name == "no_acute_calibration":
            patch(scorer, "_calibrate_score_for_acute_risk", lambda base, *_a, **_k: int(base))

        # Offline mode: force local/fallback paths even if API key exists.
        if offline:
            os.environ["DISABLE_TEMPLATE_EMBEDDINGS"] = "1"
            patch(prober, "DEEPSEEK_API_KEY", "")
            patch(risk_router, "DEEPSEEK_API_KEY", "")
            patch(orchestrator, "extract_symptoms", extractor.extract_symptoms_fallback)
            patch(prober, "retrieve_relevant_patient_evidence", _offline_memory_retrieval)
            patch(orchestrator, "get_top_template_matches", lambda *_a, **_k: [])
            patch(orchestrator, "compute_turn_risk_score", lambda *_a, **_k: 0.0)
        else:
            if disable_template_embeddings:
                os.environ["DISABLE_TEMPLATE_EMBEDDINGS"] = "1"
                patch(orchestrator, "get_top_template_matches", lambda *_a, **_k: [])
                patch(orchestrator, "compute_turn_risk_score", lambda *_a, **_k: 0.0)
            if disable_memory_embeddings and variant_name != "no_memory":
                patch(prober, "retrieve_relevant_patient_evidence", _offline_memory_retrieval)
        # If no API key is present, default to fallback extractor for all variants.
        if (not offline) and (not DEEPSEEK_API_KEY) and variant_name != "fallback_extractor":
            patch(orchestrator, "extract_symptoms", extractor.extract_symptoms_fallback)

        for pid in persona_ids:
            persona = get_persona_client(
                pid,
                use_mock=use_mock,
                use_ai_mock=(use_ai_mock and not offline),
            )
            conv, bdi_score, _ = run_conversation(
                persona,
                str(pid),
                use_extractor=True,
                run_policy=policy,
            )
            scores[pid] = int(bdi_score)
            turns[pid] = len(conv) // 2
            conversations[pid] = conv

    avg_turns = sum(turns.values()) / max(len(turns), 1)
    return {
        "variant": variant_name,
        "scores": scores,
        "turns": turns,
        "avg_turns": round(avg_turns, 2),
        "trend": _trend(scores),
        "conversations": conversations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run memory/component ablation experiments.")
    parser.add_argument("--personas", type=str, default="1-8", help="Persona IDs: range (1-8) or CSV.")
    parser.add_argument("--run-id", type=str, default="1", help="Run policy id used as baseline.")
    parser.add_argument("--mock", action="store_true", help="Use mock personas.")
    parser.add_argument(
        "--static-mock-persona",
        action="store_true",
        help="With --mock, force static mock persona (no API in persona side).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force local mode (fallback extractor, static prober, no API calls).",
    )
    parser.add_argument(
        "--disable-memory-embeddings",
        action="store_true",
        help="Keep live API, but use lexical memory retrieval instead of embedding model.",
    )
    parser.add_argument(
        "--disable-template-embeddings",
        action="store_true",
        help="Keep live API, but disable template embedding model path.",
    )
    parser.add_argument(
        "--quick-live",
        action="store_true",
        help="Reduce turns for a fast live ablation smoke run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "newVersion15Mar" / "reports" / "component_ablation_report.json",
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    persona_ids = _parse_persona_ids(args.personas)
    for pid in persona_ids:
        if not 1 <= pid <= 20:
            raise ValueError(f"Persona id out of range 1-20: {pid}")

    run_policy = get_run_policy(args.run_id)
    variants = _build_variants()
    results: list[dict[str, Any]] = []

    for variant in variants:
        name = variant["name"]
        print(f"[ablation] Running variant: {name}", flush=True)
        results.append(
            _run_variant(
                name,
                persona_ids=persona_ids,
                use_mock=bool(args.mock),
                use_ai_mock=not bool(args.static_mock_persona),
                offline=bool(args.offline),
                disable_memory_embeddings=bool(args.disable_memory_embeddings),
                disable_template_embeddings=bool(args.disable_template_embeddings),
                quick_live=bool(args.quick_live),
                run_policy=run_policy,
            )
        )

    baseline = next((r for r in results if r["variant"] == "baseline"), None)
    if baseline is None:
        raise RuntimeError("Baseline result missing.")
    base_scores = baseline["scores"]

    summary = []
    for row in results:
        deltas = {}
        for pid, score in row["scores"].items():
            deltas[str(pid)] = int(score) - int(base_scores.get(pid, 0))
        mae_vs_baseline = sum(abs(v) for v in deltas.values()) / max(len(deltas), 1)
        summary.append(
            {
                "variant": row["variant"],
                "avg_turns": row["avg_turns"],
                "mae_vs_baseline": round(mae_vs_baseline, 3),
                "score_delta_vs_baseline": deltas,
                "monotonic_nonincreasing": row["trend"]["monotonic_nonincreasing"],
                "violation_count": len(row["trend"]["violations"]),
            }
        )

    report = {
        "personas": persona_ids,
        "run_id": args.run_id,
        "mock": bool(args.mock),
        "static_mock_persona": bool(args.static_mock_persona),
        "offline": bool(args.offline),
        "disable_memory_embeddings": bool(args.disable_memory_embeddings),
        "disable_template_embeddings": bool(args.disable_template_embeddings),
        "quick_live": bool(args.quick_live),
        "api_key_present": bool(DEEPSEEK_API_KEY),
        "summary": summary,
        "variants": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
