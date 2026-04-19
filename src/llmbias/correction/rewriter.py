from __future__ import annotations

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
            revised = " ".join(revised.split())
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
        cleaned = candidate.strip().strip('"').strip()
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
        return " ".join(updated.split())

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
