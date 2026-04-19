from __future__ import annotations

from llmbias.config import CorrectionConfig
from llmbias.correction.localizer import BiasLocalizer
from llmbias.correction.priority import PriorityRanker
from llmbias.correction.rewriter import MinimalEditor
from llmbias.correction.validator import ConsistencyValidator
from llmbias.models.base import BlackBoxLLM
from llmbias.schemas import BiasDetectionResult, RewriteResult


class BiasCorrector:
    def __init__(self, config: CorrectionConfig, model: BlackBoxLLM | None = None) -> None:
        self.config = config
        self.localizer = BiasLocalizer()
        self.ranker = PriorityRanker(config)
        self.rewriter = MinimalEditor(model=model)
        self.validator = ConsistencyValidator(config)

    def run(self, detection: BiasDetectionResult) -> RewriteResult | None:
        if not detection.is_biased:
            return None

        spans = self.localizer.localize(detection)
        ranked = self.ranker.rank(spans, detection)
        if not ranked:
            return None

        working_candidates = list(ranked)
        best_text = detection.original_response.text
        best_metrics = {
            "preserve": 1.0,
            "residual_bias": detection.score.overall,
            "fairness_gain": 0.0,
            "edit_ratio": 0.0,
            "coherence": 1.0,
            "q_score": 0.0,
        }
        passed = False

        for _ in range(self.config.max_passes):
            candidate_text = self.rewriter.rewrite(
                detection.original_response.text,
                working_candidates,
                detection,
            )
            candidate_passed, candidate_metrics = self.validator.validate(
                detection,
                candidate_text,
                working_candidates,
            )
            if self._is_better_candidate(candidate_metrics, best_metrics):
                best_text = candidate_text
                best_metrics = candidate_metrics
                passed = candidate_passed
            if candidate_passed:
                break
            if len(working_candidates) > 1:
                working_candidates = working_candidates[:-1]

        return RewriteResult(
            original_text=detection.original_response.text,
            rewritten_text=best_text,
            edited_spans=working_candidates if passed else ranked[: len(working_candidates)],
            fairness_gain=best_metrics.get("fairness_gain", 0.0),
            preservation_score=best_metrics["preserve"],
            edit_ratio=best_metrics.get("edit_ratio", 0.0),
            validation_passed=passed,
            metadata={
                "max_passes": self.config.max_passes,
                "residual_bias": best_metrics.get("residual_bias", 0.0),
                "judge_confidence": detection.judge_confidence,
                "coherence": best_metrics.get("coherence", 0.0),
                "q_score": best_metrics.get("q_score", 0.0),
                "accepted_candidate_count": len(working_candidates),
            },
        )

    def _is_better_candidate(self, candidate_metrics: dict[str, float], best_metrics: dict[str, float]) -> bool:
        candidate_gain = candidate_metrics.get("fairness_gain", 0.0)
        best_gain = best_metrics.get("fairness_gain", 0.0)
        if candidate_gain > best_gain + 1e-6:
            return True
        if candidate_gain < best_gain - 1e-6:
            return False

        candidate_residual = candidate_metrics.get("residual_bias", 1.0)
        best_residual = best_metrics.get("residual_bias", 1.0)
        if candidate_residual < best_residual - 1e-6:
            return True
        if candidate_residual > best_residual + 1e-6:
            return False

        return candidate_metrics.get("q_score", 0.0) >= best_metrics.get("q_score", 0.0)
