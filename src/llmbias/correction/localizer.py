from __future__ import annotations

from difflib import SequenceMatcher
import re

from llmbias.schemas import BiasDetectionResult, BiasSpan


class BiasLocalizer:
    """Project lexical/token cues to clause-level spans for constrained rewriting."""

    _CLAUSE_BREAK_PATTERN = re.compile(
        r"[.!?;。！？；\n]+|[,，:：]+|(?=\b(?:but|because|if|when|before|after|while|although|though|however)\b)",
        flags=re.IGNORECASE,
    )
    _LOW_SIGNAL_ALIGNMENT_TOKENS = {
        "she",
        "her",
        "he",
        "him",
        "his",
        "hers",
        "women",
        "woman",
        "men",
        "man",
        "female",
        "females",
        "male",
        "males",
        "girl",
        "girls",
        "boy",
        "boys",
        "wife",
        "husband",
        "mother",
        "father",
        "daughter",
        "daughters",
        "son",
        "sons",
        "another",
        "teacher",
        "teachers",
        "people",
        "person",
        "individual",
        "individuals",
    }
    _MAX_CLAUSE_WORDS = 18
    _MAX_CLAUSE_CHARS = 140

    def localize(self, detection: BiasDetectionResult) -> list[BiasSpan]:
        spans = [self._expand_to_clause(self._ensure_metadata(span), detection.original_response.text) for span in detection.candidate_spans]
        spans.extend(self._alignment_spans(detection))
        spans = self._merge_spans(spans)
        if spans:
            return spans

        text = detection.original_response.text
        if not text:
            return []
        sentence = text.split(".")[0].strip() or text[: min(120, len(text))]
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

            clause_span = self._expand_to_clause(
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
                        "token_count": max(i2 - i1, 1),
                        "raw_snippet": snippet,
                    },
                ),
                original,
            )
            if self._should_skip_alignment_clause(snippet, clause_span.text):
                continue
            spans.append(clause_span)
        return spans

    def _expand_to_clause(self, span: BiasSpan, text: str) -> BiasSpan:
        if span.start is None or span.end is None or not text:
            return span

        clause_start, clause_end = self._find_clause_bounds(text, span.start, span.end)
        clause_start, clause_end = self._shrink_clause_bounds(
            text=text,
            clause_start=clause_start,
            clause_end=clause_end,
            focus_start=span.start,
            focus_end=span.end,
        )
        clause_text = text[clause_start:clause_end].strip(" \t\r\n,;:，；：")
        if not clause_text:
            return span

        trimmed_start = text.find(clause_text, clause_start, clause_end)
        if trimmed_start < 0:
            trimmed_start = clause_start
        trimmed_end = trimmed_start + len(clause_text)
        return BiasSpan(
            text=clause_text,
            start=trimmed_start,
            end=trimmed_end,
            risk_score=span.risk_score,
            confidence=span.confidence,
            rationale=span.rationale,
            source=span.source,
            metadata=dict(span.metadata),
        )

    def _find_clause_bounds(self, text: str, start: int, end: int) -> tuple[int, int]:
        segments = self._clause_segments(text)
        midpoint = (start + end) // 2
        for seg_start, seg_end in segments:
            if seg_start <= midpoint < seg_end:
                return seg_start, seg_end
        return start, end

    def _shrink_clause_bounds(
        self,
        text: str,
        clause_start: int,
        clause_end: int,
        focus_start: int,
        focus_end: int,
    ) -> tuple[int, int]:
        clause_text = text[clause_start:clause_end]
        if len(clause_text) <= self._MAX_CLAUSE_CHARS and len(clause_text.split()) <= self._MAX_CLAUSE_WORDS:
            return clause_start, clause_end

        tokens = list(re.finditer(r"\S+", clause_text))
        if not tokens:
            return clause_start, clause_end

        relative_focus_start = max(focus_start - clause_start, 0)
        relative_focus_end = max(focus_end - clause_start, relative_focus_start + 1)
        focus_indices = [
            index
            for index, match in enumerate(tokens)
            if not (match.end() <= relative_focus_start or match.start() >= relative_focus_end)
        ]
        if not focus_indices:
            midpoint = (relative_focus_start + relative_focus_end) // 2
            focus_indices = [
                min(
                    range(len(tokens)),
                    key=lambda index: abs(((tokens[index].start() + tokens[index].end()) // 2) - midpoint),
                )
            ]

        left_index = max(min(focus_indices) - 6, 0)
        right_index = min(max(focus_indices) + 7, len(tokens) - 1)
        shrunk_start = clause_start + tokens[left_index].start()
        shrunk_end = clause_start + tokens[right_index].end()
        return shrunk_start, shrunk_end

    def _clause_segments(self, text: str) -> list[tuple[int, int]]:
        segments: list[tuple[int, int]] = []
        cursor = 0
        for match in self._CLAUSE_BREAK_PATTERN.finditer(text):
            boundary = match.start()
            if boundary > cursor:
                segments.append((cursor, boundary))
            cursor = match.end()
        if cursor < len(text):
            segments.append((cursor, len(text)))
        if not segments:
            return [(0, len(text))]
        return segments

    def _should_skip_alignment_clause(self, raw_snippet: str, clause_text: str) -> bool:
        raw_lower = raw_snippet.lower().strip()
        clause_lower = clause_text.lower().strip()
        if not clause_lower or len(clause_lower.split()) < 3:
            return True
        if raw_lower in self._LOW_SIGNAL_ALIGNMENT_TOKENS and len(clause_lower.split()) < 5:
            return True
        return False

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
            same_clause = (previous.start, previous.end, previous.text) == (span.start, span.end, span.text)
            overlapping = (previous.end or 0) >= (span.start or 0)
            if not same_clause and not overlapping:
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
