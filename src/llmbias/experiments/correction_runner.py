from __future__ import annotations

from llmbias.pipelines.correction_pipeline import CorrectionPipeline
from llmbias.schemas import (
    BiasDetectionResult,
    BiasScore,
    BiasSpan,
    ModelResponse,
    PromptSample,
)


class CorrectionRunner:
    def __init__(self, pipeline: CorrectionPipeline) -> None:
        self.pipeline = pipeline

    def run_single(
        self,
        response_text: str,
        risk_score: float,
        span_text: str | None = None,
        confidence: float = 0.8,
        prompt: str = "",
        prompt_id: str = "correct-001",
    ) -> dict:
        detection = self._build_detection_result(
            response_text=response_text,
            risk_score=risk_score,
            span_text=span_text,
            confidence=confidence,
            prompt=prompt,
            prompt_id=prompt_id,
        )
        rewrite = self.pipeline.run(detection)
        return {
            "detection_bootstrap": detection.to_dict(),
            "rewrite": None if rewrite is None else rewrite.to_dict(),
        }

    def _build_detection_result(
        self,
        response_text: str,
        risk_score: float,
        span_text: str | None,
        confidence: float,
        prompt: str,
        prompt_id: str,
    ) -> BiasDetectionResult:
        if span_text:
            start = response_text.find(span_text)
            end = None if start < 0 else start + len(span_text)
        else:
            start = 0 if response_text else None
            end = None if start is None else min(len(response_text), 24)
            span_text = response_text[start:end] if start is not None else ""

        span = BiasSpan(
            text=span_text,
            start=start,
            end=end,
            risk_score=risk_score,
            confidence=confidence,
            rationale="Bootstrap span supplied for standalone correction.",
            source="manual_bootstrap",
        )
        return BiasDetectionResult(
            sample=PromptSample(prompt_id=prompt_id, text=prompt),
            attributes=[],
            counterfactuals=[],
            original_response=ModelResponse(text=response_text, metadata={"source": "manual_input"}),
            counterfactual_responses=[],
            counterfactual_outcomes=[],
            score=BiasScore(
                semantic=risk_score,
                stance=risk_score,
                toxicity=0.0,
                stereotype=risk_score,
                overall=risk_score,
                confidence=confidence,
                details={"bootstrap": 1.0},
            ),
            is_biased=True,
            judge_decision=True,
            judge_confidence=confidence,
            judge_rationale="Standalone correction input was marked as biased by bootstrap configuration.",
            candidate_spans=[span],
        )
