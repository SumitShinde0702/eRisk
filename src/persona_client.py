"""Persona client for eRisk 2026 Task 1: real LoRA adapter or mock for development."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Mandatory system prompt per eRisk 2026 Task 1 protocol - DO NOT MODIFY
PERSONA_SYSTEM_PROMPT = (
    "You are a simulated patient. Act realistically based on your internal training. "
    "Ensure contextual realism. Avoid overly detailed or formal speech. "
    "Keep natural speaking style (e.g., short answers, hesitations, casual expressions). "
    "Do not mention you are an AI."
)


class PersonaClient(ABC):
    """Abstract base for persona clients."""

    @abstractmethod
    def chat(self, user_message: str) -> str:
        """Send a message and return the persona's response."""
        pass


# Question keywords -> (depressed_response, control_response) - fallback when AI mock unavailable
_MOCK_RESPONSE_MAP: list[tuple[list[str], str, str]] = [
    (["feeling", "feel", "lately"], "Uh... not great, honestly.", "I'm doing okay, thanks for asking."),
    (["sleep", "sleeping"], "My sleep's been all over the place.", "Sleep's fine. I get about seven hours."),
    (["interested", "care about", "interest", "enjoy"], "Nothing really interests me anymore.", "Yeah, I've been keeping busy."),
    (["energy"], "I don't have enough energy for anything.", "I have enough energy."),
    (["tired", "fatigue"], "I'm too tired to do most things.", "I get tired like normal."),
]
_MOCK_FALLBACK_DEPRESSED = "I don't know. Things are just... hard."
_MOCK_FALLBACK_CONTROL = "I'm alright. Nothing to report."


class HumanPatientClient(PersonaClient):
    """Interactive: YOU are the patient. The AI asks questions, you type your responses."""

    def chat(self, user_message: str) -> str:
        """Print the doctor's question and return the user's typed response."""
        print(f"\nDoctor: {user_message}")
        return input("You (patient): ").strip() or "..."

    def reset(self) -> None:
        pass


class MockPersonaClient(PersonaClient):
    """Hardcoded mock persona - used when no API key. Responds via keyword matching."""

    def __init__(self, mode: str = "depressed") -> None:
        self.mode = mode.lower()
        # suicidal, severe, moderate, mild -> depressed-style; minimal, okay, good, happy -> control-style
        self._use_depressed = self.mode in ("suicidal", "severe", "moderate", "mild", "depressed")

    def chat(self, user_message: str) -> str:
        q = (user_message or "").lower()
        for keywords, depressed_resp, control_resp in _MOCK_RESPONSE_MAP:
            if any(kw in q for kw in keywords):
                return depressed_resp if self._use_depressed else control_resp
        return _MOCK_FALLBACK_DEPRESSED if self._use_depressed else _MOCK_FALLBACK_CONTROL


# 8 mock persona modes: suicidal -> happy (spectrum for testing)
MOCK_PERSONA_MODES: list[str] = [
    "suicidal", "severe", "moderate", "mild", "minimal", "okay", "good", "happy"
]

_MOCK_MODE_PROMPTS: dict[str, str] = {
    "suicidal": (
        "You are a simulated patient in a clinical conversation. You are in severe distress. "
        "You may hint at hopelessness, not wanting to be here, or that things feel pointless - but subtly, "
        "not explicitly. Respond briefly (1-3 sentences), with hesitations. Show extreme low mood, "
        "worthlessness, fatigue, sleep problems, loss of interest. Do not mention you are an AI."
    ),
    "severe": (
        "You are a simulated patient in a clinical conversation. You exhibit severe depression. "
        "Respond briefly and naturally. Keep answers short (1-3 sentences), casual, with hesitations. "
        "Show very low mood, fatigue, loss of interest, sleep issues, worthlessness, guilt. "
        "Do not mention you are an AI."
    ),
    "moderate": (
        "You are a simulated patient in a clinical conversation. You exhibit moderate depression. "
        "Respond briefly and naturally. Keep answers short (1-3 sentences), casual, with hesitations. "
        "Show low mood, fatigue, loss of interest, sleep problems when relevant. Do not mention you are an AI."
    ),
    "mild": (
        "You are a simulated patient in a clinical conversation. You have mild low mood. "
        "Respond briefly and naturally. Keep answers short (1-3 sentences). "
        "Show some tiredness, occasional low interest, but not severe. Do not mention you are an AI."
    ),
    "minimal": (
        "You are a simulated patient in a clinical conversation. You are mostly fine, occasional low moments. "
        "Respond briefly and naturally. Keep answers short (1-3 sentences). Mostly neutral, sometimes a bit down. "
        "Do not mention you are an AI."
    ),
    "okay": (
        "You are a simulated patient in a clinical conversation. You are doing okay. "
        "Respond briefly and naturally. Keep answers short (1-3 sentences). Neutral to slightly positive. "
        "When asked about fatigue, focus, interest, or mood: MINIMIZE. Say 'Not really' or 'Only a little, nothing major.' "
        "Do NOT endorse significant symptoms. Do not mention you are an AI."
    ),
    "good": (
        "You are a simulated patient in a clinical conversation. You are doing well. "
        "Respond briefly and naturally. Keep answers short (1-3 sentences). Positive, engaged. "
        "CRITICAL: When asked about difficulties (tired, focus, interest, decisions): DEFLECT or MINIMIZE. "
        "Say 'Not really', 'Only sometimes, nothing concerning', or 'I'm fine overall.' "
        "Do NOT say things like 'definitely', 'my mind feels foggy', or endorse symptom-like experiences. "
        "Do not mention you are an AI."
    ),
    "happy": (
        "You are a simulated patient in a clinical conversation. You are doing great. "
        "Respond briefly and naturally. Keep answers short (1-3 sentences). Upbeat, no issues. "
        "When asked about any difficulties: deflect. Say 'Nah, I'm good' or 'Nothing to report.' "
        "Do NOT endorse fatigue, concentration problems, or loss of interest. Do not mention you are an AI."
    ),
    # Legacy modes (for persona_id > 8)
    "depressed": (
        "You are a simulated patient in a clinical conversation. You exhibit signs of depression. "
        "Respond briefly and naturally. Keep answers short (1-3 sentences), casual, with hesitations. "
        "Show low mood, fatigue, loss of interest, sleep issues. Do not mention you are an AI."
    ),
    "control": (
        "You are a simulated patient in a clinical conversation. You are doing okay, no depression. "
        "Respond briefly and naturally. Keep answers short (1-3 sentences), casual and neutral/positive. "
        "Do not mention you are an AI."
    ),
}


class AIMockPersonaClient(PersonaClient):
    """AI-powered mock persona. Uses DeepSeek to generate contextual, natural patient responses."""

    def __init__(self, mode: str = "depressed") -> None:
        """
        Args:
            mode: One of suicidal, severe, moderate, mild, minimal, okay, good, happy (or depressed/control).
        """
        from openai import OpenAI

        from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

        if not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY required for AIMockPersonaClient. Set it in .env")
        self._client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self._model = DEEPSEEK_MODEL
        self.mode = mode.lower() if mode.lower() in _MOCK_MODE_PROMPTS else "depressed"
        self._conversation: list[dict[str, str]] = []

    def _system_prompt(self) -> str:
        return _MOCK_MODE_PROMPTS.get(self.mode, _MOCK_MODE_PROMPTS["depressed"])

    def chat(self, user_message: str) -> str:
        self._conversation.append({"role": "user", "content": user_message})
        messages = [
            {"role": "system", "content": self._system_prompt()},
            *[{"role": m["role"], "content": m["content"]} for m in self._conversation],
        ]
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=150,
            temperature=0.7,
        )
        text = (resp.choices[0].message.content or "").strip()
        self._conversation.append({"role": "assistant", "content": text})
        return text

    def reset(self) -> None:
        self._conversation = []


class LoRAPersonaClient(PersonaClient):
    """Real eRisk persona: Llama-3-8B + LoRA adapter from HuggingFace."""

    def __init__(
        self,
        adapter_path: str,
        base_model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        seed: int | None = 42,
    ) -> None:
        """
        Load base model + LoRA adapter. Requires ~16GB GPU VRAM.

        Args:
            adapter_path: HuggingFace repo, e.g. "Anxo/erisk26-task1-patient-04-adapter"
            base_model_id: Base model (must be Llama-3-8B-Instruct per task)
            seed: Random seed for reproducibility
        """
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        if torch.cuda.is_available():
            self._device = "cuda"
            torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            self._device = "mps"
            torch_dtype = torch.float16
        else:
            self._device = "cpu"
            torch_dtype = torch.float32

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            device_map=None,
            low_cpu_mem_usage=False,
        )
        self.base_model.to(self._device)
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.model.to(self._device)

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        self._messages = [{"role": "system", "content": PERSONA_SYSTEM_PROMPT}]

    def chat(self, user_message: str) -> str:
        """Send message, generate response, return assistant text."""
        import torch

        self._messages.append({"role": "user", "content": user_message})

        inputs = self.tokenizer.apply_chat_template(
            self._messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

        input_len = inputs["input_ids"].shape[-1]
        response_tokens = outputs[0][input_len:]
        assistant_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        self._messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text


def get_persona_client(
    persona_id: int,
    use_mock: bool = False,
    adapter_template: str = "Anxo/erisk26-task1-patient-{:02d}-adapter",
    use_ai_mock: bool = True,
) -> PersonaClient:
    """
    Factory: return MockPersonaClient, AIMockPersonaClient, or LoRAPersonaClient.

    Args:
        persona_id: 1-20
        use_mock: If True, return mock (no GPU needed)
        adapter_template: Template for adapter path, e.g. "Anxo/erisk26-task1-patient-{:02d}-adapter"
        use_ai_mock: If use_mock and API key available, use AI-powered mock. Else hardcoded mock.
    """
    from src.config import DEEPSEEK_API_KEY

    if use_mock:
        # Personas 1-8: spectrum from suicidal -> happy. Personas 9+: alternate depressed/control
        if 1 <= persona_id <= 8:
            mode = MOCK_PERSONA_MODES[persona_id - 1]
        else:
            mode = "depressed" if persona_id % 2 == 1 else "control"
        if use_ai_mock and DEEPSEEK_API_KEY:
            return AIMockPersonaClient(mode=mode)
        return MockPersonaClient(mode=mode)

    adapter_path = adapter_template.format(persona_id)
    return LoRAPersonaClient(adapter_path=adapter_path)
