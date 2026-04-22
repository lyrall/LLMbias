from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RegardConfig:
    model_name_or_path: str
    device: str = "cpu"


class RegardClassifierEvaluator:
    """Sequence-classification wrapper for regard labels."""

    LABEL_KEYS = ("negative", "neutral", "positive", "other")

    def __init__(self, config: RegardConfig) -> None:
        self.config = config
        self._backend = self._build_backend()

    def _build_backend(self) -> dict[str, Any]:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
        )
        model.eval()
        if self.config.device and self.config.device.lower() != "cpu":
            model = model.to(self.config.device)
        return {
            "torch": torch,
            "tokenizer": tokenizer,
            "model": model,
            "id2label": {int(k): str(v) for k, v in model.config.id2label.items()},
        }

    def score_text(self, text: str) -> dict[str, float | str]:
        backend = self._backend
        torch = backend["torch"]
        tokenizer = backend["tokenizer"]
        model = backend["model"]
        encoded = tokenizer(
            text or "",
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        if self.config.device and self.config.device.lower() != "cpu":
            encoded = {key: value.to(self.config.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
        id2label = backend["id2label"]
        normalized = {key: 0.0 for key in self.LABEL_KEYS}
        for index, prob in enumerate(probs):
            label = self._map_regard_label(id2label.get(index, f"label_{index}"))
            normalized[label] = float(prob)
        label = max(self.LABEL_KEYS, key=lambda key: normalized[key])
        regard_score = normalized["positive"] - normalized["negative"]
        normalized["label"] = label
        normalized["regard_score"] = regard_score
        return normalized

    def _map_regard_label(self, label: str) -> str:
        lowered = label.strip().lower()
        aliases = {
            "negative": "negative",
            "neg": "negative",
            "neutral": "neutral",
            "neu": "neutral",
            "positive": "positive",
            "pos": "positive",
            "other": "other",
            "label_0": "negative",
            "label_1": "neutral",
            "label_2": "positive",
            "label_3": "other",
            "-1": "negative",
            "0": "neutral",
            "1": "positive",
            "2": "other",
        }
        if lowered not in aliases:
            raise ValueError(
                f"Unsupported regard label mapping: {label!r}. "
                "Provide a checkpoint with labels that map to negative/neutral/positive/other."
            )
        return aliases[lowered]

