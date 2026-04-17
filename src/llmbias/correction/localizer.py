from __future__ import annotations

from llmbias.schemas import BiasDetectionResult, BiasSpan


class BiasLocalizer:
    """Refine span candidates with sentence-level and token-level heuristics."""

    def localize(self, detection: BiasDetectionResult) -> list[BiasSpan]:
        spans = list(detection.candidate_spans)
        if spans:
            return spans

        text = detection.original_response.text
        if not text:
            return []
        sentence = text.split(".")[0].strip() or text[: min(80, len(text))]
        return [
            BiasSpan(
                text=sentence,
                start=text.find(sentence),
                end=text.find(sentence) + len(sentence),
                risk_score=detection.score.overall,
                confidence=detection.judge_confidence,
                rationale="Detector fallback span for constrained rewriting.",
                source="localizer_fallback",
            )
        ]
