# eRisk 2026 Task 1: Agent-to-Agent Depression Detection System

An A2A (agent-to-agent) system for **Conversational Depression Detection with LLM Personas** at CLEF eRisk 2026. Uses specialized agents to converse with simulated patient personas (Llama-3-8B + LoRA), infer BDI-II symptoms indirectly, and produce submission-ready JSON outputs.

## Architecture

- **Prober Agent** (DeepSeek): Selects/adapts indirect questions from a BDI-aligned question bank
- **Extractor Agent** (DeepSeek): Extracts BDI-II symptom scores (0–3) from conversation
- **Stopper Agent** (rule-based): Decides when to continue or classify
- **Scorer Agent** (rule-based): Computes BDI score and selects up to 4 key symptoms

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and set:

- `DEEPSEEK_API_KEY` – Required for Prober and Extractor (DeepSeek API)
- `HF_TOKEN` – Required for real personas (gated Llama + adapters)

### 3. Run with mock persona (no GPU/API needed)

```bash
python run.py --persona 4 --run 1 --mock
```

### 4. Run with real persona (requires GPU + HF access)

```bash
python run.py --persona 4 --run 1
```

### 5. Run all 20 personas for a run

```bash
python run.py --run 1
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--persona N` | Run single persona (1–20). Omit for all 20. |
| `--run N` | Run ID: 1, 2, 3, or `all` |
| `--mock` | Use mock persona (no GPU/HF) |
| `--manual` | Mark as manual run (prefix filenames with `manual_`) |
| `--output-dir` | Output directory (default: `outputs/`) |

## Output

For each run, outputs are saved to `outputs/run{N}/`:

- `interactions_run{N}.json` – Conversation log per persona
- `results_run{N}.json` – BDI score and key symptoms per persona

## Submission

- Run over all 20 personas (IDs 1–20)
- Submit both JSON files per run to FTP: `task1-llms-results/persona{N}/`
- Use `manual_` prefix for human-assisted runs (at most one)

### FTP Submission Steps (copy/paste)

#### 1) Generate official runs (automated)

```bash
cd ~/eRisk
source .venv/bin/activate
python run.py --run all --output-dir outputs/submission/final
```

This creates:

- `outputs/submission/final/run1/interactions_run1.json`
- `outputs/submission/final/run1/results_run1.json`
- `outputs/submission/final/run2/interactions_run2.json`
- `outputs/submission/final/run2/results_run2.json`
- `outputs/submission/final/run3/interactions_run3.json`
- `outputs/submission/final/run3/results_run3.json`

#### 2) Verify FTP access

```bash
lftp -u '<FTP_USER>','<FTP_TOKEN>' ftp://erisk.irlab.org -e "pwd; ls; bye"
```

#### 3) Upload a persona folder (example: persona6)

```bash
lftp -u '<FTP_USER>','<FTP_TOKEN>' ftp://erisk.irlab.org -e "mirror -R /Users/itzsihui/eRisk/outputs/submission/persona6 task1-llms-results/persona6; bye"
```

#### 4) Verify uploaded files

```bash
lftp -u '<FTP_USER>','<FTP_TOKEN>' ftp://erisk.irlab.org -e "ls task1-llms-results/persona6; bye"
```

#### 5) Upload all personas in one shot (if you prepared local `task1-llms-results/`)

```bash
lftp -u '<FTP_USER>','<FTP_TOKEN>' ftp://erisk.irlab.org -e "mirror -R /Users/itzsihui/eRisk/outputs/submission/task1-llms-results task1-llms-results; bye"
```

#### 6) Optional: mark a manual run

```bash
python run.py --run 1 --manual
```

Manual files are prefixed with `manual_` (at most one manual run is allowed by task rules).

#### Notes

- If `lftp` is missing: `brew install lftp`
- If `mkdir` returns FTP 550 but `mirror -R` reports files transferred, upload may still be successful
- Rotate/regenerate FTP token after sharing it in chats/logs

## External Ground-Truth Reference (TalkDep)

We include an external ranking reference from TalkDep for development-time calibration:

- `knowledge/talkdep_golden_truth.yaml`
- Source: `Anxo06/TalkDep` -> `golden_truth_BDI_depression_ranking.txt`

Important:

- This is an external reference for calibration/evaluation only.
- It is **not** a direct label source for eRisk Task 1 personas (IDs `1-20`).

Evaluate ranking agreement (when your results use TalkDep persona names):

```bash
python scripts/eval_talkdep_ranking.py --results /path/to/results.json
```

## References

- [eRisk 2026 Task 1](https://erisk.irlab.org/Task1LLMs.html)
- [Persona adapter (HuggingFace)](https://huggingface.co/Anxo/erisk26-task1-patient-04-adapter)
- [TalkDep repository](https://github.com/Anxo06/TalkDep)
