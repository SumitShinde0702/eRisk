#!/usr/bin/env python3
"""CLI entry point for eRisk 2026 Task 1 - run conversations with personas and produce submission JSON."""

import argparse
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEEPSEEK_API_KEY, MAX_MESSAGES, MIN_EXCHANGES_BEFORE_STOP, OUTPUT_DIR
from src.output_formatter import format_interactions, format_results, save_run_outputs
from src.persona_client import HumanPatientClient, MOCK_PERSONA_MODES, get_persona_client
from src.orchestrator import run_conversation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="eRisk 2026 Task 1: Run A2A depression detection on LLM personas"
    )
    parser.add_argument(
        "--persona",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="Run persona(s) 1-20. E.g. --persona 1 2 3 or --persona 1. If omitted, run all 20.",
    )
    parser.add_argument(
        "--personas",
        type=str,
        default=None,
        metavar="RANGE",
        help="Shorthand for persona range. E.g. --personas 1-8 runs personas 1 through 8. "
        "With --mock, 1-8 = suicidal, severe, moderate, mild, minimal, okay, good, happy.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="1",
        help="Run ID: 1, 2, 3, or 'all' for all three runs.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock persona (no GPU/HF needed).",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Mark as manual run (prefix filenames with manual_).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: you play the patient. AI asks questions, you type responses.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for JSON files.",
    )
    args = parser.parse_args()

    use_extractor = bool(DEEPSEEK_API_KEY)
    if not use_extractor and not args.mock and not args.interactive:
        parser.error(
            "DEEPSEEK_API_KEY is required for non-mock runs. "
            "Without extractor signals, BDI outputs may stay zero."
        )
    if MAX_MESSAGES < MIN_EXCHANGES_BEFORE_STOP:
        parser.error(
            f"Invalid config: MAX_MESSAGES ({MAX_MESSAGES}) must be >= "
            f"MIN_EXCHANGES_BEFORE_STOP ({MIN_EXCHANGES_BEFORE_STOP})."
        )

    if args.interactive:
        print("Interactive mode: You are the patient. Answer the doctor's questions.\n", flush=True)
        persona = HumanPatientClient()
        conv, bdi_score, key_symptoms = run_conversation(
            persona,
            "interactive",
            use_extractor=use_extractor,
        )
        print("\n" + "=" * 50, flush=True)
        print(f"BDI-II Score: {bdi_score}", flush=True)
        print(f"Key symptoms: {key_symptoms}", flush=True)
        print("=" * 50, flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_dir = args.output_dir / "interactive"
        out_dir.mkdir(parents=True, exist_ok=True)
        interactions = [{"LLM": "interactive", "conversation": conv}]
        results = [{"LLM": "interactive", "bdi-score": bdi_score, "key-symptoms": key_symptoms}]
        save_run_outputs(
            out_dir,
            "1",
            format_interactions(interactions),
            format_results(results),
            manual_prefix="manual_" if args.manual else "",
        )
        print(f"Saved to {out_dir}/", flush=True)
        print("Done.", flush=True)
        return

    persona_ids = list(range(1, 21))
    if args.personas:
        try:
            start, end = map(int, args.personas.split("-"))
            persona_ids = list(range(start, end + 1))
            for pid in persona_ids:
                if not 1 <= pid <= 20:
                    parser.error(f"--personas range must be 1-20, got {args.personas}")
        except ValueError:
            parser.error(f"--personas must be like 1-8, got {args.personas}")
    elif args.persona is not None and len(args.persona) > 0:
        ids = list(args.persona)
        for pid in ids:
            if not 1 <= pid <= 20:
                parser.error(f"--persona must be 1-20, got {pid}")
        persona_ids = ids

    run_ids: list[str] = []
    if args.run.lower() == "all":
        # Keep official submission convention for "all".
        run_ids = ["1", "2", "3"]
    elif args.run.isdigit() and int(args.run) >= 1:
        # Allow any positive run id for development (e.g., --run 9, --run 42).
        run_ids = [args.run]
    else:
        parser.error("--run must be a positive integer (e.g., 1, 9, 42) or 'all'")

    manual_prefix = "manual_" if args.manual else ""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for run_id in run_ids:
        interactions: list[dict] = []
        results: list[dict] = []

        for pid in persona_ids:
            mode_str = f" ({MOCK_PERSONA_MODES[pid - 1]})" if args.mock and 1 <= pid <= 8 else ""
            print(f"Run {run_id} | Persona {pid}{mode_str}...", flush=True)
            persona = get_persona_client(pid, use_mock=args.mock)
            conv, bdi_score, key_symptoms = run_conversation(
                persona,
                str(pid),
                use_extractor=use_extractor,
            )
            interactions.append({"LLM": str(pid), "conversation": conv})
            results.append(
                {"LLM": str(pid), "bdi-score": bdi_score, "key-symptoms": key_symptoms}
            )

        interactions_json = format_interactions(interactions)
        results_json = format_results(results)

        out_dir = args.output_dir / f"run{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_run_outputs(
            out_dir,
            run_id,
            interactions_json,
            results_json,
            manual_prefix=manual_prefix,
        )
        print(f"Saved to {out_dir}/", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
