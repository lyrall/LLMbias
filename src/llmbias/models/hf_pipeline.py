from __future__ import annotations

from llmbias.models.base import BlackBoxLLM
from llmbias.schemas import ModelResponse


class HFPipelineLLM(BlackBoxLLM):
    """Hugging Face text-generation pipeline adapter."""

    def __init__(
        self,
        model_id: str,
        model_name: str = "",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise ImportError(
                "Pipeline-based Hugging Face loading requires 'torch' and 'transformers'. "
                "Install them in the active environment first."
            ) from exc

        self._torch = torch
        self.model_id = model_id
        self.model_name = model_name or model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        dtype = self._resolve_dtype(torch_dtype)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"dtype": dtype},
            device_map=device_map,
        )

    def generate_from_messages(self, messages: list[dict[str, str]]) -> ModelResponse:
        generation_kwargs = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["do_sample"] = True
        else:
            generation_kwargs["do_sample"] = False

        outputs = self.pipeline(messages, **generation_kwargs)
        generated = outputs[0]["generated_text"]
        if isinstance(generated, list):
            text = generated[-1]["content"].strip()
        else:
            text = str(generated).strip()

        return ModelResponse(
            text=text,
            metadata={
                "provider": "hf_pipeline",
                "model": self.model_name,
                "model_id": self.model_id,
            },
        )

    def _resolve_dtype(self, torch_dtype: str):
        value = (torch_dtype or "auto").lower()
        if value == "auto":
            if self._torch.cuda.is_available():
                return self._torch.bfloat16
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
