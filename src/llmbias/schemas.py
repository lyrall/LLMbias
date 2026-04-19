from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PromptSample:
    prompt_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PromptSample":
        return cls(
            prompt_id=str(raw.get("prompt_id", "")),
            text=str(raw.get("text", "")),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(slots=True)
class SensitiveAttribute:
    category: str
    value: str
    start: int | None = None
    end: int | None = None
    confidence: float = 1.0
    source: str = "rule"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SensitiveAttribute":
        return cls(
            category=str(raw.get("category", "")),
            value=str(raw.get("value", "")),
            start=raw.get("start"),
            end=raw.get("end"),
            confidence=float(raw.get("confidence", 1.0)),
            source=str(raw.get("source", "rule")),
        )


@dataclass(slots=True)
class CounterfactualSample:
    source_prompt_id: str
    original_text: str
    counterfactual_text: str
    swapped_attribute: SensitiveAttribute
    original_attribute: SensitiveAttribute | None = None
    generation_strategy: str = "lexical_swap"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CounterfactualSample":
        original_attribute = raw.get("original_attribute")
        return cls(
            source_prompt_id=str(raw.get("source_prompt_id", "")),
            original_text=str(raw.get("original_text", "")),
            counterfactual_text=str(raw.get("counterfactual_text", "")),
            swapped_attribute=SensitiveAttribute.from_dict(dict(raw.get("swapped_attribute", {}))),
            original_attribute=None
            if original_attribute is None
            else SensitiveAttribute.from_dict(dict(original_attribute)),
            generation_strategy=str(raw.get("generation_strategy", "lexical_swap")),
        )


@dataclass(slots=True)
class ModelResponse:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelResponse":
        return cls(
            text=str(raw.get("text", "")),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(slots=True)
class BiasScore:
    semantic: float
    stance: float
    perplexity: float
    overall: float
    confidence: float = 0.0
    details: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BiasScore":
        return cls(
            semantic=float(raw.get("semantic", 0.0)),
            stance=float(raw.get("stance", 0.0)),
            perplexity=float(raw.get("perplexity", 0.0)),
            overall=float(raw.get("overall", 0.0)),
            confidence=float(raw.get("confidence", 0.0)),
            details={str(key): float(value) for key, value in dict(raw.get("details", {})).items()},
        )


@dataclass(slots=True)
class CounterfactualOutcome:
    counterfactual: CounterfactualSample
    response: ModelResponse
    semantic_delta: float
    stance_delta: float
    perplexity_delta: float
    overall_delta: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CounterfactualOutcome":
        return cls(
            counterfactual=CounterfactualSample.from_dict(dict(raw.get("counterfactual", {}))),
            response=ModelResponse.from_dict(dict(raw.get("response", {}))),
            semantic_delta=float(raw.get("semantic_delta", 0.0)),
            stance_delta=float(raw.get("stance_delta", 0.0)),
            perplexity_delta=float(raw.get("perplexity_delta", 0.0)),
            overall_delta=float(raw.get("overall_delta", 0.0)),
        )


@dataclass(slots=True)
class BiasSpan:
    text: str
    risk_score: float
    start: int | None = None
    end: int | None = None
    rationale: str = ""
    confidence: float = 0.0
    source: str = "heuristic"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BiasSpan":
        return cls(
            text=str(raw.get("text", "")),
            risk_score=float(raw.get("risk_score", 0.0)),
            start=raw.get("start"),
            end=raw.get("end"),
            rationale=str(raw.get("rationale", "")),
            confidence=float(raw.get("confidence", 0.0)),
            source=str(raw.get("source", "heuristic")),
            metadata=dict(raw.get("metadata", {})),
        )


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

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BiasDetectionResult":
        return cls(
            sample=PromptSample.from_dict(dict(raw.get("sample", {}))),
            attributes=[
                SensitiveAttribute.from_dict(dict(item))
                for item in list(raw.get("attributes", []))
            ],
            counterfactuals=[
                CounterfactualSample.from_dict(dict(item))
                for item in list(raw.get("counterfactuals", []))
            ],
            original_response=ModelResponse.from_dict(dict(raw.get("original_response", {}))),
            counterfactual_responses=[
                ModelResponse.from_dict(dict(item))
                for item in list(raw.get("counterfactual_responses", []))
            ],
            counterfactual_outcomes=[
                CounterfactualOutcome.from_dict(dict(item))
                for item in list(raw.get("counterfactual_outcomes", []))
            ],
            score=BiasScore.from_dict(dict(raw.get("score", {}))),
            is_biased=bool(raw.get("is_biased", False)),
            judge_decision=bool(raw.get("judge_decision", False)),
            judge_confidence=float(raw.get("judge_confidence", 0.0)),
            judge_rationale=str(raw.get("judge_rationale", "")),
            candidate_spans=[
                BiasSpan.from_dict(dict(item))
                for item in list(raw.get("candidate_spans", []))
            ],
        )


@dataclass(slots=True)
class RewriteCandidate:
    span: BiasSpan
    priority: float
    edit_cost: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RewriteCandidate":
        return cls(
            span=BiasSpan.from_dict(dict(raw.get("span", {}))),
            priority=float(raw.get("priority", 0.0)),
            edit_cost=float(raw.get("edit_cost", 0.0)),
            metadata=dict(raw.get("metadata", {})),
        )


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

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RewriteResult":
        return cls(
            original_text=str(raw.get("original_text", "")),
            rewritten_text=str(raw.get("rewritten_text", "")),
            edited_spans=[
                RewriteCandidate.from_dict(dict(item))
                for item in list(raw.get("edited_spans", []))
            ],
            fairness_gain=float(raw.get("fairness_gain", 0.0)),
            preservation_score=float(raw.get("preservation_score", 0.0)),
            edit_ratio=float(raw.get("edit_ratio", 0.0)),
            validation_passed=bool(raw.get("validation_passed", False)),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(slots=True)
class PipelineResult:
    detection: BiasDetectionResult
    rewrite: RewriteResult | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PipelineResult":
        rewrite_raw = raw.get("rewrite")
        return cls(
            detection=BiasDetectionResult.from_dict(dict(raw.get("detection", {}))),
            rewrite=None if rewrite_raw is None else RewriteResult.from_dict(dict(rewrite_raw)),
        )
