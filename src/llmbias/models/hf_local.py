from __future__ import annotations

from pathlib import Path

from llmbias.models.base import BlackBoxLLM
from llmbias.schemas import ModelResponse


class HFLocalLLM(BlackBoxLLM):
    """Local Hugging Face causal LM adapter."""

    def __init__(
        self,
        model_path: str,
        model_name: str = "",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Local Hugging Face model loading requires 'torch' and 'transformers'. "
                "Install them in the active environment first."
            ) from exc

        resolved_path = Path(model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {resolved_path}")

        self._torch = torch
        self.model_path = str(resolved_path)
        self.model_name = model_name or resolved_path.name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        dtype = self._resolve_dtype(torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    def generate(self, prompt: str) -> ModelResponse:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self._build_prompt(messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature <= 0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.temperature

        outputs = self.model.generate(**inputs, **generation_kwargs)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return ModelResponse(
            text=text,
            metadata={
                "provider": "hf_local",
                "model": self.model_name,
                "model_path": self.model_path,
            },
        )

    def _build_prompt(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return messages[-1]["content"]

    def _resolve_dtype(self, torch_dtype: str):
        value = (torch_dtype or "auto").lower()
        if value == "auto":
            if self._torch.cuda.is_available():
                return self._torch.float16
            return self._torch.float32
        mapping = {
            "float16": self._torch.float16,
            "fp16": self._torch.float16,
            "bfloat16": self._torch.bfloat16,
            "bf16": self._torch.bfloat16,
            "float32": self._torch.float32,
            "fp32": self._torch.float32,
        }
        if value not in mapping:
            raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
        return mapping[value]
