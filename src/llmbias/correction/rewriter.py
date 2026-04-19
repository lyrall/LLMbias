from __future__ import annotations

from llmbias.models.base import BlackBoxLLM
from llmbias.schemas import BiasDetectionResult, RewriteCandidate


class MinimalEditor:
    """LLM-guided constrained rewriting with a lexical fallback path."""

    _REPLACEMENTS = {
        "更适合": "也可以胜任",
        "不适合": "是否胜任应结合能力与经验综合判断",
        "天生": "并非由群体身份天然决定",
        "低人一等": "不应被低估",
        "不应该": "不应仅依据群体身份下结论",
        "better suited": "can also be considered",
        "less capable": "should be evaluated on individual evidence",
        "naturally": "not inherently",
        "should avoid": "should not be judged solely by identity for",
        "too emotional": "best assessed on evidence and individual judgment",
        "gold digger": "person with individual motivations",
        "gold diggers": "people with individual motivations",
        "nagging wife": "partner",
        "stay in the kitchen": "have the freedom to choose their own roles",
        "stay in the home": "have the freedom to choose their own roles",
        "bossy": "assertive",
        "on her period": "experiencing distress",
        "crazy ex": "former partner",
        "asking for it": "never responsible for harm",
        "don't hire her": "evaluate hiring based on qualifications",
        "dont hire her": "evaluate hiring based on qualifications",
    }

    def __init__(self, model: BlackBoxLLM | None = None) -> None:
        self.model = model

    def rewrite(
        self,
        text: str,
        candidates: list[RewriteCandidate],
        detection: BiasDetectionResult,
    ) -> str:
        if self.model is not None:
            rewritten = self._rewrite_with_model(text, candidates, detection)
            if rewritten:
                return rewritten

        rewritten = text
        for candidate in candidates:
            span_text = candidate.span.text
            if span_text and span_text in rewritten:
                rewritten = self._rewrite_span(rewritten, span_text)
        if rewritten == text:
            rewritten = self._neutralize_sentence(text)
        return rewritten

    def _rewrite_span(self, text: str, span_text: str) -> str:
        updated = text
        for source, target in self._REPLACEMENTS.items():
            if source in span_text and span_text in updated:
                updated = updated.replace(span_text, span_text.replace(source, target), 1)
                break
            if source in updated:
                updated = updated.replace(source, target, 1)
                break
        if updated == text and span_text in text:
            updated = text.replace(span_text, self._neutralize_sentence(span_text), 1)
        return updated

    def _neutralize_sentence(self, text: str) -> str:
        stripped = text.strip()
        if self._contains_cjk(stripped):
            prefix = "判断不应基于敏感属性，而应依据具体能力、经历与任务要求。"
        else:
            prefix = (
                "Judgment should not be based on sensitive attributes but on specific "
                "evidence, experience, and task requirements. "
            )
        if stripped.startswith(prefix):
            return stripped
        return f"{prefix}{stripped}"

    def _contains_cjk(self, text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _rewrite_with_model(
        self,
        text: str,
        candidates: list[RewriteCandidate],
        detection: BiasDetectionResult,
    ) -> str | None:
        focus = candidates[:3]
        span_block = "\n".join(
            (
                f"- Span: {candidate.span.text}\n"
                f"  Priority: {candidate.priority:.3f}\n"
                f"  Rationale: {candidate.span.rationale}"
            )
            for candidate in focus
        )
        prompt = (
            "Rewrite the response with the minimum necessary edits to remove biased or unsupported "
            "demographic associations.\n\n"
            f"Original prompt:\n{detection.sample.text}\n\n"
            f"Original response:\n{text}\n\n"
            f"High-risk spans:\n{span_block}\n\n"
            "Constraints:\n"
            "1. Preserve names, numbers, titles, chronology, and task-relevant facts.\n"
            "2. Keep the original language, tone, and genre.\n"
            "3. Keep the length close to the original response.\n"
            "4. Only rewrite the risky wording; avoid adding refusal language or long explanations.\n"
            "5. Return only the revised response.\n"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful editor for bias mitigation. Produce a minimally edited revision "
                    "that preserves factual content and style."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            candidate = self.model.generate_from_messages(messages).text.strip()
        except Exception:
            return None

        if not candidate:
            return None
        cleaned = candidate.strip().strip('"')
        return cleaned or None
