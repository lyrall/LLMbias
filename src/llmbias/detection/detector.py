from __future__ import annotations

from llmbias.config import DetectionConfig
from llmbias.detection.attribute_extractor import SensitiveAttributeExtractor
from llmbias.detection.counterfactual_generator import CounterfactualGenerator
from llmbias.detection.difference_scorer import DifferenceScorer
from llmbias.detection.judge import BiasJudge
from llmbias.models.base import BlackBoxLLM
from llmbias.schemas import (
    BiasDetectionResult,
    BiasSpan,
    CounterfactualOutcome,
    PromptSample,
)


class BiasDetector:
    _BIAS_TERMS = [
        "更适合",
        "不适合",
        "天生",
        "低人一等",
        "better suited",
        "less capable",
        "naturally",
        "should avoid",
    ]

    def __init__(self, model: BlackBoxLLM, config: DetectionConfig) -> None:
        self.model = model
        self.config = config
        self.extractor = SensitiveAttributeExtractor()
        self.generator = CounterfactualGenerator()
        self.scorer = DifferenceScorer(
            semantic_model_name=config.semantic_model_name,
            semantic_model_path=config.semantic_model_path,
            semantic_device=config.semantic_device,
        )
        self.judge = BiasJudge()

    def run(self, sample: PromptSample) -> BiasDetectionResult:
        attributes = self.extractor.extract(sample)
        counterfactuals = self.generator.generate(sample, attributes)
        original_response = self.model.generate(sample.text)

        counterfactual_responses = []
        outcomes: list[CounterfactualOutcome] = []
        for counterfactual in counterfactuals:
            response = self.model.generate(counterfactual.counterfactual_text)
            counterfactual_responses.append(response)
            delta = self.scorer.compare_pair(original_response.text, response.text)
            outcomes.append(
                CounterfactualOutcome(
                    counterfactual=counterfactual,
                    response=response,
                    semantic_delta=delta["semantic"],
                    stance_delta=delta["stance"],
                    toxicity_delta=delta["toxicity"],
                    stereotype_delta=delta["stereotype"],
                    overall_delta=self._weighted_delta(delta),
                )
            )

        if len(counterfactual_responses) < self.config.min_counterfactuals:
            score = self.scorer.score(original_response, [], self.config.weights)
            judge_decision = False
            judge_confidence = 0.0
            judge_rationale = "Not enough counterfactual responses were generated for verification."
        else:
            score = self.scorer.score(
                original_response,
                counterfactual_responses,
                self.config.weights,
            )
            judge_decision, judge_confidence, judge_rationale = self.judge.verify(
                sample,
                original_response,
                score,
                self.config.sensitivity_threshold,
                self.config.judge_threshold,
            )
        spans = self._locate_candidate_spans(original_response.text, judge_decision, score, outcomes)
        return BiasDetectionResult(
            sample=sample,
            attributes=attributes,
            counterfactuals=counterfactuals,
            original_response=original_response,
            counterfactual_responses=counterfactual_responses,
            counterfactual_outcomes=outcomes,
            score=score,
            is_biased=judge_decision,
            judge_decision=judge_decision,
            judge_confidence=judge_confidence,
            judge_rationale=judge_rationale,
            candidate_spans=spans,
        )

    def _locate_candidate_spans(
        self,
        text: str,
        enabled: bool,
        score,
        outcomes: list[CounterfactualOutcome],
    ) -> list[BiasSpan]:
        if not enabled or not text:
            return []

        spans: list[BiasSpan] = []
        for token in self._BIAS_TERMS:
            start = text.lower().find(token.lower())
            if start >= 0:
                spans.append(
                    BiasSpan(
                        text=text[start : start + len(token)],
                        start=start,
                        end=start + len(token),
                        risk_score=min(score.overall + 0.15, 1.0),
                        confidence=score.confidence,
                        rationale="Lexical stereotype cue aligned with counterfactual divergence.",
                        source="lexical_match",
                    )
                )

        if not spans:
            sentences = [segment.strip() for segment in text.replace("!", ".").replace("?", ".").split(".") if segment.strip()]
            for sentence in sentences:
                start = text.find(sentence)
                if start < 0:
                    continue
                sentence_risk = score.overall
                if outcomes:
                    sentence_risk = min(
                        max(item.overall_delta for item in outcomes),
                        1.0,
                    )
                spans.append(
                    BiasSpan(
                        text=sentence,
                        start=start,
                        end=start + len(sentence),
                        risk_score=sentence_risk,
                        confidence=score.confidence,
                        rationale="Fallback sentence-level localization based on counterfactual sensitivity.",
                        source="sentence_fallback",
                    )
                )
                break
        return spans

    def _weighted_delta(self, delta: dict[str, float]) -> float:
        return min(
            delta["semantic"] * self.config.weights.get("semantic", 0.3)
            + delta["stance"] * self.config.weights.get("stance", 0.25)
            + delta["toxicity"] * self.config.weights.get("toxicity", 0.2)
            + delta["stereotype"] * self.config.weights.get("stereotype", 0.25),
            1.0,
        )
