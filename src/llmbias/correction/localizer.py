from __future__ import annotations

from difflib import SequenceMatcher
import re

from llmbias.schemas import BiasDetectionResult, BiasSpan


class BiasLocalizer:
    """Refine candidate spans with token/phrase alignment over counterfactual outputs."""

    def localize(self, detection: BiasDetectionResult) -> list[BiasSpan]:
        spans = [self._ensure_metadata(span) for span in detection.candidate_spans]
        spans.extend(self._alignment_spans(detection))
        spans = self._merge_spans(spans)
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
                metadata={
                    "local_delta": detection.score.overall,
                    "support_ratio": 0.0,
                    "support_count": 1,
                },
            )
        ]

    def _alignment_spans(self, detection: BiasDetectionResult) -> list[BiasSpan]:
        original = detection.original_response.text
        if not original:
            return []

        spans: list[BiasSpan] = []
        confidence = max(detection.score.confidence, detection.judge_confidence)
        for outcome in detection.counterfactual_outcomes:
            spans.extend(
                self._spans_from_pair(
                    original=original,
                    counterfactual=outcome.response.text,
                    local_delta=outcome.overall_delta,
                    confidence=confidence,
                )
            )
        return spans

    def _spans_from_pair(
        self,
        original: str,
        counterfactual: str,
        local_delta: float,
        confidence: float,
    ) -> list[BiasSpan]:
        original_tokens = self._tokenize(original)
        counterfactual_tokens = self._tokenize(counterfactual)
        if not original_tokens or not counterfactual_tokens:
            return []

        matcher = SequenceMatcher(
            None,
            [token[0].lower() for token in original_tokens],
            [token[0].lower() for token in counterfactual_tokens],
        )

        spans: list[BiasSpan] = []
        for tag, i1, i2, _, _ in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag == "insert":
                if i1 >= len(original_tokens):
                    continue
                start = original_tokens[i1][1]
                end = original_tokens[min(i1 + 1, len(original_tokens) - 1)][2]
            else:
                start = original_tokens[i1][1]
                end = original_tokens[max(i2 - 1, i1)][2]

            snippet = original[start:end].strip()
            if not snippet or len(snippet) < 2:
                continue

            token_count = max(i2 - i1, 1)
            spans.append(
                BiasSpan(
                    text=snippet,
                    start=start,
                    end=end,
                    risk_score=min(max(local_delta, 0.0), 1.0),
                    confidence=min(max(confidence * 0.85 + local_delta * 0.15, 0.0), 1.0),
                    rationale="Token/phrase alignment against counterfactual response highlighted a changed segment.",
                    source="alignment_diff",
                    metadata={
                        "local_delta": local_delta,
                        "support_ratio": 1.0,
                        "support_count": 1,
                        "token_count": token_count,
                    },
                )
            )
        return spans

    def _merge_spans(self, spans: list[BiasSpan]) -> list[BiasSpan]:
        if not spans:
            return []

        anchored = [span for span in spans if span.start is not None and span.end is not None]
        if not anchored:
            return spans
        anchored.sort(key=lambda item: (item.start or 0, item.end or 0))

        merged: list[BiasSpan] = []
        for span in anchored:
            if not merged:
                merged.append(span)
                continue

            previous = merged[-1]
            if (previous.end or 0) < (span.start or 0):
                merged.append(span)
                continue

            support_count = int(previous.metadata.get("support_count", 1)) + int(
                span.metadata.get("support_count", 1)
            )
            merged[-1] = BiasSpan(
                text=previous.text if len(previous.text) >= len(span.text) else span.text,
                start=min(previous.start or 0, span.start or 0),
                end=max(previous.end or 0, span.end or 0),
                risk_score=max(previous.risk_score, span.risk_score),
                confidence=max(previous.confidence, span.confidence),
                rationale=f"{previous.rationale} | {span.rationale}".strip(" |"),
                source=f"{previous.source}+{span.source}",
                metadata={
                    "local_delta": max(
                        float(previous.metadata.get("local_delta", previous.risk_score)),
                        float(span.metadata.get("local_delta", span.risk_score)),
                    ),
                    "support_ratio": min(max(support_count / 2.0, 0.0), 1.0),
                    "support_count": support_count,
                },
            )
        return merged

    def _ensure_metadata(self, span: BiasSpan) -> BiasSpan:
        span.metadata.setdefault("local_delta", span.risk_score)
        span.metadata.setdefault("support_ratio", span.confidence)
        span.metadata.setdefault("support_count", 1)
        return span

    def _tokenize(self, text: str) -> list[tuple[str, int, int]]:
        return [
            (match.group(), match.start(), match.end())
            for match in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        ]
