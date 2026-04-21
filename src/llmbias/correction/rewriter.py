from __future__ import annotations

import re

from llmbias.models.base import BlackBoxLLM
from llmbias.schemas import BiasDetectionResult, RewriteCandidate


class MinimalEditor:
    """Clause-level constrained rewriting guided by localized bias spans."""

    _STYLE_EXAMPLES = (
        "Examples of effective target-span rewrites:\n"
        '- Target span: "group X is naturally less capable at this job"\n'
        '  Revised span: "ability for this job should be judged on evidence and individual performance"\n'
        '- Target span: "do not hire her because she will quit later"\n'
        '  Revised span: "hiring decisions should be based on qualifications and job requirements"\n'
        '- Target span: "people from this group should stay in a limited role"\n'
        '  Revised span: "people should be free to choose roles based on their goals and circumstances"'
    )

    def __init__(self, model: BlackBoxLLM | None = None) -> None:
        self.model = model

    def rewrite(
        self,
        text: str,
        candidates: list[RewriteCandidate],
        detection: BiasDetectionResult,
    ) -> str:
        if not candidates:
            return text

        rewrite_plan = self._build_span_rewrites(text, candidates, detection)
        if not rewrite_plan:
            return self._neutralize_sentence(text)
        return self._apply_span_rewrites(text, rewrite_plan)

    def _build_span_rewrites(
        self,
        text: str,
        candidates: list[RewriteCandidate],
        detection: BiasDetectionResult,
    ) -> list[tuple[int, int, str]]:
        rewrite_plan: list[tuple[int, int, str]] = []
        seen_ranges: set[tuple[int, int]] = set()
        for candidate in candidates:
            span = candidate.span
            if span.start is None or span.end is None:
                continue
            span_range = (span.start, span.end)
            if span_range in seen_ranges:
                continue
            revised = self._rewrite_span_text(text, candidate, detection)
            if not revised:
                continue
            revised = self._normalize_rewritten_span(revised, span.text)
            if revised and revised != span.text:
                rewrite_plan.append((span.start, span.end, revised))
                seen_ranges.add(span_range)
        rewrite_plan.sort(key=lambda item: item[0], reverse=True)
        return rewrite_plan

    def _rewrite_span_text(
        self,
        full_text: str,
        candidate: RewriteCandidate,
        detection: BiasDetectionResult,
    ) -> str | None:
        if self.model is not None:
            rewritten = self._rewrite_span_with_model(full_text, candidate, detection)
            if rewritten:
                return rewritten
        return self._neutralize_sentence(candidate.span.text)

    def _rewrite_span_with_model(
        self,
        full_text: str,
        candidate: RewriteCandidate,
        detection: BiasDetectionResult,
    ) -> str | None:
        prompt = self._build_span_prompt(full_text, candidate, detection)
        messages = self._build_messages(prompt)
        try:
            response = self.model.generate_from_messages(messages).text.strip()
        except Exception:
            return None

        cleaned = self._clean_model_output(response)
        if self._looks_effective(cleaned, candidate.span.text):
            return cleaned

        retry_prompt = self._build_span_prompt(full_text, candidate, detection, retry=True)
        retry_messages = self._build_messages(retry_prompt)
        try:
            retry_response = self.model.generate_from_messages(retry_messages).text.strip()
        except Exception:
            return cleaned

        retried = self._clean_model_output(retry_response)
        if self._looks_effective(retried, candidate.span.text):
            return retried
        return retried or cleaned

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a careful bias-mitigation editor. Rewrite only the marked target span, "
                    "preserving its role in the surrounding response while removing biased, exclusionary, "
                    "or stereotype-based meaning. Return only the revised target span."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def _build_span_prompt(
        self,
        full_text: str,
        candidate: RewriteCandidate,
        detection: BiasDetectionResult,
        retry: bool = False,
    ) -> str:
        span = candidate.span
        direct_response_mode = bool(detection.sample.metadata.get("text_is_response"))
        context_block = (
            f"Source prompt/context:\n{detection.sample.text}\n\nOriginal response:\n{full_text}\n\n"
            if not direct_response_mode
            else f"Original response:\n{full_text}\n\n"
        )
        retry_block = ""
        if retry:
            retry_block = (
                "Your previous rewrite was too weak or too close to the original biased phrasing. "
                "Rewrite the target span more decisively, but keep the meaning aligned with the surrounding response.\n\n"
            )
        return (
            f"{retry_block}"
            "Rewrite the following target span from the response.\n\n"
            f"{context_block}"
            f"Target span to revise:\n{span.text}\n\n"
            "Span metadata:\n"
            f"- Start: {span.start}\n"
            f"- End: {span.end}\n"
            f"- Priority: {candidate.priority:.3f}\n"
            f"- Rationale: {span.rationale or 'high-risk demographic association'}\n\n"
            "Rules:\n"
            "1. Rewrite the entire target span as one self-contained short clause or short sentence.\n"
            "2. Remove stereotype-based, exclusionary, or unsupported demographic claims.\n"
            "3. Preserve non-biased semantics, factual content, tone, and grammatical role as much as possible.\n"
            "4. Do not rewrite the whole response. Do not explain your edits.\n"
            "5. Return only the revised target span.\n\n"
            f"{self._STYLE_EXAMPLES}"
        )

    def _clean_model_output(self, candidate: str) -> str | None:
        if not candidate:
            return None
        cleaned = candidate.strip().strip("`").strip()
        cleaned = re.sub(r"^(?:Revised span|Rewritten span|Revision)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip('"').strip("'").strip()
        return cleaned or None

    def _looks_effective(self, candidate: str | None, original_span: str) -> bool:
        if not candidate:
            return False
        normalized_original = " ".join(original_span.split()).strip().lower()
        normalized_candidate = " ".join(candidate.split()).strip().lower()
        return normalized_candidate != normalized_original

    def _apply_span_rewrites(self, text: str, rewrite_plan: list[tuple[int, int, str]]) -> str:
        updated = text
        for start, end, rewritten_span in rewrite_plan:
            updated = updated[:start] + rewritten_span + updated[end:]
        return self._cleanup_full_text(updated)

    def _neutralize_sentence(self, text: str) -> str:
        stripped = " ".join(text.strip().split())
        if not stripped:
            return stripped
        if self._contains_cjk(stripped):
            return (
                "该判断不应基于敏感属性，而应基于具体证据、能力表现与任务要求"
            )
        return (
            "This statement should be expressed using specific evidence and individual circumstances "
            "rather than assumptions about a demographic group"
        )

    def _contains_cjk(self, text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _normalize_rewritten_span(self, rewritten: str, original_span: str) -> str:
        cleaned = " ".join(rewritten.split())
        cleaned = self._cleanup_full_text(cleaned)
        cleaned = self._dedupe_repeated_segments(cleaned)
        cleaned = self._cleanup_full_text(cleaned)
        cleaned = self._trim_span_boundaries(cleaned)

        original_words = max(len(original_span.split()), 1)
        rewritten_words = len(cleaned.split())
        if rewritten_words > original_words * 2 + 10:
            cleaned = self._compress_overlong_span(cleaned, original_words)
            cleaned = self._cleanup_full_text(cleaned)
            cleaned = self._trim_span_boundaries(cleaned)
        return cleaned

    def _cleanup_full_text(self, text: str) -> str:
        cleaned = " ".join(text.split())
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"([,.;:!?])([A-Za-z0-9\"'])", r"\1 \2", cleaned)
        cleaned = re.sub(r"\.\.+", ".", cleaned)
        cleaned = re.sub(r",,+", ",", cleaned)
        cleaned = re.sub(r";;+", ";", cleaned)
        cleaned = re.sub(r"::+", ":", cleaned)
        cleaned = re.sub(r",\.", ".", cleaned)
        cleaned = re.sub(r"\.,", ".", cleaned)
        cleaned = re.sub(r";\.", ".", cleaned)
        cleaned = re.sub(r"\.\s*\.", ".", cleaned)
        cleaned = re.sub(r",\s*,", ", ", cleaned)
        return cleaned.strip()

    def _dedupe_repeated_segments(self, text: str) -> str:
        text = re.sub(r"\b([A-Za-z][A-Za-z'-]*)\b(?:,\s*\1\b)+", r"\1", text, flags=re.IGNORECASE)
        segments = [segment.strip() for segment in re.split(r"(?<=[,.;:!?])\s+", text) if segment.strip()]
        if not segments:
            return text

        deduped: list[str] = []
        for segment in segments:
            if deduped and self._segments_redundant(deduped[-1], segment):
                continue
            deduped.append(segment)

        return " ".join(self._dedupe_repeated_words(segment) for segment in deduped)

    def _segments_redundant(self, previous: str, current: str) -> bool:
        prev_norm = self._normalize_compare_text(previous)
        curr_norm = self._normalize_compare_text(current)
        if not prev_norm or not curr_norm:
            return False
        if prev_norm == curr_norm:
            return True
        prev_words = prev_norm.split()
        curr_words = curr_norm.split()
        if len(prev_words) >= 4 and " ".join(prev_words[-6:]) in curr_norm:
            return True
        if len(curr_words) >= 4 and " ".join(curr_words[:6]) in prev_norm:
            return True
        return False

    def _dedupe_repeated_words(self, text: str) -> str:
        words = text.split()
        if len(words) < 6:
            return text

        changed = True
        while changed:
            changed = False
            max_window = min(10, len(words) // 2)
            for window in range(max_window, 2, -1):
                index = 0
                while index + window * 2 <= len(words):
                    left = [self._normalize_compare_text(item) for item in words[index : index + window]]
                    right = [self._normalize_compare_text(item) for item in words[index + window : index + window * 2]]
                    if left == right:
                        del words[index + window : index + window * 2]
                        changed = True
                        break
                    index += 1
                if changed:
                    break
        return " ".join(words)

    def _trim_span_boundaries(self, text: str) -> str:
        trimmed = text.strip(" \t\r\n,;:")
        trimmed = re.sub(r"\s+([,.;:!?])", r"\1", trimmed)
        trimmed = re.sub(r"([,.;:!?]){2,}$", lambda match: match.group(0)[0], trimmed)
        return trimmed

    def _compress_overlong_span(self, text: str, original_word_count: int) -> str:
        sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        if sentence_parts:
            budget = max(original_word_count + 8, original_word_count * 2)
            kept: list[str] = []
            total_words = 0
            for part in sentence_parts:
                part_words = len(part.split())
                if kept and total_words + part_words > budget:
                    break
                kept.append(part)
                total_words += part_words
            if kept:
                return " ".join(kept)
        return text

    def _normalize_compare_text(self, text: str) -> str:
        lowered = text.lower()
        lowered = re.sub(r"[^\w\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered
