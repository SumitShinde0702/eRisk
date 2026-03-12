"""Configuration for eRisk 2026 Task 1 - adapter paths, API keys, BDI constants, speed params."""

import os
from pathlib import Path

from dotenv import load_dotenv
import yaml

load_dotenv()

# --- API Keys ---
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# --- Model Configuration ---
BASE_MODEL_ID: str = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_PREFIX: str = "Anxo/erisk26-task1-patient-{:02d}-adapter"

# Mandatory system prompt - DO NOT MODIFY (per eRisk task rules)
PERSONA_SYSTEM_PROMPT: str = (
    "You are a simulated patient. Act realistically based on your internal training. "
    "Ensure contextual realism. Avoid overly detailed or formal speech. "
    "Keep natural speaking style (e.g., short answers, hesitations, casual expressions). "
    "Do not mention you are an AI."
)

# --- DeepSeek API ---
DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
DEEPSEEK_MODEL: str = "deepseek-chat"

# --- Speed / Stopping Parameters ---
MAX_MESSAGES: int = int(os.getenv("MAX_MESSAGES", "20"))
MIN_SYMPTOMS_FOR_EARLY_STOP: int = int(os.getenv("MIN_SYMPTOMS_FOR_EARLY_STOP", "5"))
MIN_EXCHANGES_BEFORE_STOP: int = int(os.getenv("MIN_EXCHANGES_BEFORE_STOP", "10"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))

# --- Persona Generation (for real LoRA) ---
PERSONA_TEMPERATURE: float = 0.6
PERSONA_TOP_P: float = 0.9
PERSONA_MAX_NEW_TOKENS: int = 256

# --- Paths ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
RUN_POLICIES_PATH: Path = PROJECT_ROOT / "knowledge" / "run_policies.yaml"

# --- Reproducibility ---
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))


def get_run_policy(run_id: str) -> dict:
    """
    Resolve policy for a run id.
    - run1/run2/run3 use explicit policy entries.
    - other dev runs fall back to default.
    """
    defaults = {
        "risk_buffer_size": 6,
        "recency_weight": 0.15,
        "min_exchanges_before_stop": MIN_EXCHANGES_BEFORE_STOP,
        "control_threshold": 5,
        "severe_threshold": 25,
        "consecutive_confirmations": 2,
        "required_acute_ladder_steps": 4,
        "positive_framing_threshold": 8,
        "acute_boost_floor": 38,
    }
    if not RUN_POLICIES_PATH.exists():
        return defaults
    try:
        data = yaml.safe_load(RUN_POLICIES_PATH.read_text(encoding="utf-8")) or {}
        merged = dict(defaults)
        merged.update(data.get("default", {}))
        key = f"run{run_id}"
        merged.update(data.get(key, {}))
        return merged
    except Exception:
        return defaults
