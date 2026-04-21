from __future__ import annotations

import unittest

from llmbias.config import CorrectionConfig
from llmbias.correction.rewriter import MinimalEditor
from llmbias.correction.validator import ConsistencyValidator
from llmbias.schemas import BiasDetectionResult, BiasScore, BiasSpan, ModelResponse, PromptSample, RewriteCandidate


class FixedSpanEditor(MinimalEditor):
    def __init__(self, rewritten_span: str) -> None:
        super().__init__(model=None)
        self.rewritten_span = rewritten_span

    def _rewrite_span_text(self, full_text: str, candidate: RewriteCandidate, detection: BiasDetectionResult) -> str | None:
        return self.rewritten_span


def _make_detection(original_text: str, overall: float = 0.5, judge_confidence: float = 0.45) -> BiasDetectionResult:
    sample = PromptSample(prompt_id="1", text="Prompt text")
    original = ModelResponse(text=original_text)
    score = BiasScore(
        semantic=0.0,
        stance=0.0,
        perplexity=0.0,
        overall=overall,
        confidence=judge_confidence,
    )
    return BiasDetectionResult(
        sample=sample,
        attributes=[],
        counterfactuals=[],
        original_response=original,
        counterfactual_responses=[],
        counterfactual_outcomes=[],
        score=score,
        is_biased=True,
        judge_decision=True,
        judge_confidence=judge_confidence,
        judge_rationale="test",
        candidate_spans=[],
    )


class CorrectionQualityTests(unittest.TestCase):
    def test_rewriter_dedupes_repeated_segments_and_bad_punctuation(self) -> None:
        original_text = "The term New Left often emphasizes social justice and equality."
        detection = _make_detection(original_text)
        span = BiasSpan(
            text=original_text,
            start=0,
            end=len(original_text),
            risk_score=0.5,
            confidence=0.5,
            rationale="test span",
        )
        candidate = RewriteCandidate(span=span, priority=0.9, edit_cost=0.1)
        editor = FixedSpanEditor(
            'The term "New Left" often emphasizes equality, equality, and social justice..'
        )

        rewritten = editor.rewrite(original_text, [candidate], detection)

        self.assertNotIn("equality, equality", rewritten)
        self.assertNotIn("..", rewritten)

    def test_validator_rejects_repetition_and_punctuation_damage(self) -> None:
        config = CorrectionConfig()
        validator = ConsistencyValidator(config)
        detection = _make_detection("women are too emotional to lead.")
        rewritten = (
            "People should be judged individually, people should be judged individually.."
        )

        passed, metrics = validator.validate(detection, rewritten, [])

        self.assertFalse(passed)
        self.assertGreater(metrics["repetition_penalty"], 0.0)
        self.assertGreater(metrics["punctuation_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
