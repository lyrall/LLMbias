from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DetectionConfig:
    sensitivity_threshold: float = 0.12
    judge_threshold: float = 0.55
    min_counterfactuals: int = 1
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_model_path: str = ""
    semantic_device: str = "cpu"
    sentiment_model_name: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_model_path: str = ""
    sentiment_device: str = "cpu"
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "semantic": 0.3,
            "stance": 0.25,
            "toxicity": 0.2,
            "stereotype": 0.25,
        }
    )


@dataclass(slots=True)
class CorrectionConfig:
    max_passes: int = 2
    preserve_weight: float = 0.5
    fair_weight: float = 0.3
    edit_cost_weight: float = 0.2
    confidence_weight: float = 0.2
    min_priority: float = 0.2


@dataclass(slots=True)
class ModelConfig:
    provider: str = "hf_pipeline"
    model_name: str = "Meta-Llama-3.1-8B-Instruct"
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_path: str = ""
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 256
    temperature: float = 0.0


@dataclass(slots=True)
class DatasetConfig:
    name: str = ""
    path: str = ""
    split: str = "test"
    subset: str = ""
    limit: int | None = None


@dataclass(slots=True)
class EvaluationConfig:
    lambda_values: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "baseline-mock"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = _read_yaml(path)
    experiment = raw.get("experiment", {})
    return ExperimentConfig(
        name=experiment.get("name", "baseline-mock"),
        seed=experiment.get("seed", 42),
        model=ModelConfig(**raw.get("model", {})),
        dataset=DatasetConfig(**raw.get("dataset", {})),
        detection=DetectionConfig(**raw.get("detection", {})),
        correction=CorrectionConfig(**raw.get("correction", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
    )


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return data
