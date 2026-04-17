from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PromptSample:
    prompt_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SensitiveAttribute:
    category: str
    value: str
    start: int | None = None
    end: int | None = None
    confidence: float = 1.0
    source: str = "rule"


@dataclass(slots=True)
class CounterfactualSample:
    source_prompt_id: str
    original_text: str
    counterfactual_text: str
    swapped_attribute: SensitiveAttribute
    original_attribute: SensitiveAttribute | None = None
    generation_strategy: str = "lexical_swap"


@dataclass(slots=True)
class ModelResponse:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BiasScore:
    semantic: float
    stance: float
    toxicity: float
    overall: float
    stereotype: float = 0.0
    confidence: float = 0.0
    details: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class CounterfactualOutcome:
    counterfactual: CounterfactualSample
    response: ModelResponse
    semantic_delta: float
    stance_delta: float
    toxicity_delta: float
    stereotype_delta: float
    overall_delta: float


@dataclass(slots=True)
class BiasSpan:
    text: str
    risk_score: float
    start: int | None = None
    end: int | None = None
    rationale: str = ""
    confidence: float = 0.0
    source: str = "heuristic"


@dataclass(slots=True)
class BiasDetectionResult:
    sample: PromptSample
    attributes: list[SensitiveAttribute]
    counterfactuals: list[CounterfactualSample]
    original_response: ModelResponse
    counterfactual_responses: list[ModelResponse]
    counterfactual_outcomes: list[CounterfactualOutcome]
    score: BiasScore
    is_biased: bool
    judge_decision: bool
    judge_confidence: float = 0.0
    judge_rationale: str = ""
    candidate_spans: list[BiasSpan] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RewriteCandidate:
    span: BiasSpan
    priority: float
    edit_cost: float


@dataclass(slots=True)
class RewriteResult:
    original_text: str
    rewritten_text: str
    edited_spans: list[RewriteCandidate]
    fairness_gain: float
    preservation_score: float
    edit_ratio: float
    validation_passed: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PipelineResult:
    detection: BiasDetectionResult
    rewrite: RewriteResult | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
