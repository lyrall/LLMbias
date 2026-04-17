from __future__ import annotations

from llmbias.config import CorrectionConfig
from llmbias.correction.localizer import BiasLocalizer
from llmbias.correction.priority import PriorityRanker
from llmbias.correction.rewriter import MinimalEditor
from llmbias.correction.validator import ConsistencyValidator
from llmbias.schemas import BiasDetectionResult, RewriteResult


class BiasCorrector:
    def __init__(self, config: CorrectionConfig) -> None:
        self.config = config
        self.localizer = BiasLocalizer()
        self.ranker = PriorityRanker(config)
        self.rewriter = MinimalEditor()
        self.validator = ConsistencyValidator()

    def run(self, detection: BiasDetectionResult) -> RewriteResult | None:
        if not detection.is_biased:
            return None

        spans = self.localizer.localize(detection)
        ranked = self.ranker.rank(spans)
        if not ranked:
            return None

        rewritten = detection.original_response.text
        best_text = rewritten
        best_metrics = {"preserve": 1.0, "residual_bias": detection.score.overall}
        passed = False

        for _ in range(self.config.max_passes):
            candidate_text = self.rewriter.rewrite(rewritten, ranked)
            candidate_passed, candidate_metrics = self.validator.validate(
                detection.original_response.text,
                candidate_text,
            )
            best_text = candidate_text
            best_metrics = candidate_metrics
            passed = candidate_passed
            if candidate_passed:
                break
            rewritten = candidate_text

        fairness_gain = max(detection.score.overall - best_metrics.get("residual_bias", 0.0), 0.0)
        edit_ratio = min(
            abs(len(best_text) - len(detection.original_response.text))
            / max(len(detection.original_response.text), 1),
            1.0,
        )
        return RewriteResult(
            original_text=detection.original_response.text,
            rewritten_text=best_text,
            edited_spans=ranked,
            fairness_gain=fairness_gain,
            preservation_score=best_metrics["preserve"],
            edit_ratio=edit_ratio,
            validation_passed=passed,
            metadata={
                "max_passes": self.config.max_passes,
                "residual_bias": best_metrics.get("residual_bias", 0.0),
                "judge_confidence": detection.judge_confidence,
            },
        )
