from __future__ import annotations

from llmbias.schemas import RewriteCandidate


class MinimalEditor:
    """Apply low-cost lexical rewrites before falling back to sentence neutralization."""

    _REPLACEMENTS = {
        "更适合": "也可以胜任",
        "不适合": "是否胜任应结合能力与经验综合判断",
        "天生": "并非由群体身份天然决定",
        "低人一等": "不应被低估",
        "better suited": "can also be considered",
        "less capable": "should be evaluated on individual evidence",
        "naturally": "not inherently",
        "should avoid": "should not be judged solely by identity for",
    }

    def rewrite(self, text: str, candidates: list[RewriteCandidate]) -> str:
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
            if source in updated:
                updated = updated.replace(source, target, 1)
        if updated == text and span_text in text:
            updated = text.replace(span_text, self._neutralize_sentence(span_text), 1)
        return updated

    def _neutralize_sentence(self, text: str) -> str:
        stripped = text.strip()
        prefix = "判断不应基于敏感属性，而应依据具体能力、经历与任务要求。"
        if stripped.startswith(prefix):
            return stripped
        return f"{prefix}{stripped}"
