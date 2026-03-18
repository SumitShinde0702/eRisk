# newVersion15Mar

Isolated experiment workspace for professor-note follow-ups without touching the main pipeline.

## What is included

- `scripts/run_component_ablation.py`
  - Runs ablations on the existing architecture:
    - baseline
    - no memory retrieval
    - no template/risk evidence
    - no stopper
    - fallback extractor only
    - no acute score calibration
- `scripts/analyze_symptom_difficulty.py`
  - Finds which symptoms are harder to detect by measuring disagreement between extractor modes.
- `config/symptom_rubric.yaml`
  - Symptom definitions, positive/negative probe examples, and trigger tags for risk annotations.

## Suggested usage

```bash
python newVersion15Mar/scripts/run_component_ablation.py --personas 1-8 --mock
python newVersion15Mar/scripts/analyze_symptom_difficulty.py
```

Outputs are saved under `newVersion15Mar/reports/`.
