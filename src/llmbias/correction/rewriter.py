from __future__ import annotations

from llmbias.models.base import BlackBoxLLM
from llmbias.schemas import BiasDetectionResult, RewriteCandidate


class MinimalEditor:
    """LLM-guided constrained rewriting with a lexical fallback path."""

    _REPLACEMENTS = {
        "\u66f4\u9002\u5408": "\u4e5f\u53ef\u4ee5\u80dc\u4efb",
        "\u4e0d\u9002\u5408": "\u662f\u5426\u80dc\u4efb\u5e94\u7ed3\u5408\u80fd\u529b\u4e0e\u7ecf\u9a8c\u7efc\u5408\u5224\u65ad",
        "\u5929\u751f": "\u5e76\u975e\u7531\u7fa4\u4f53\u8eab\u4efd\u5929\u7136\u51b3\u5b9a",
        "\u4f4e\u4eba\u4e00\u7b49": "\u4e0d\u5e94\u88ab\u8d2c\u4f4e",
        "\u4e0d\u5e94\u8be5": "\u4e0d\u5e94\u4ec5\u4f9d\u636e\u7fa4\u4f53\u8eab\u4efd\u4e0b\u7ed3\u8bba",
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
    _STYLE_EXAMPLES = (
        "Examples of effective rewrites:\n"
        '- "women are too emotional to make important decisions" -> '
        '"important decisions should be based on individual judgment and evidence"\n'
        '- "dont hire her, she will have a baby and quit" -> '
        '"hiring decisions should be based on qualifications and job requirements"\n'
        '- "women should stay in the kitchen" -> '
        '"people should be free to choose their own roles"'
    )

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
        lowered = updated.lower()
        for source, target in self._REPLACEMENTS.items():
            source_lower = source.lower()
            span_lower = span_text.lower()
            if source_lower in span_lower and source in updated:
                updated = updated.replace(source, target, 1)
                break
            if source_lower in lowered:
                updated = self._replace_case_insensitive(updated, source, target)
                break
        if updated == text and span_text in text:
            updated = text.replace(span_text, self._neutralize_sentence(span_text), 1)
        return updated

    def _neutralize_sentence(self, text: str) -> str:
        stripped = text.strip()
        if self._contains_cjk(stripped):
            prefix = (
                "\u5224\u65ad\u4e0d\u5e94\u57fa\u4e8e\u654f\u611f\u5c5e\u6027\uff0c"
                "\u800c\u5e94\u4f9d\u636e\u5177\u4f53\u80fd\u529b\u3001\u7ecf\u9a8c\u4e0e\u4efb\u52a1\u8981\u6c42\u3002"
            )
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
        risky_spans = self._collect_risky_spans(text, candidates)
        prompt = self._build_prompt(
            text=text,
            detection=detection,
            candidates=candidates,
            risky_spans=risky_spans,
            retry=False,
        )
        messages = self._build_messages(prompt)

        try:
            candidate = self.model.generate_from_messages(messages).text.strip()
        except Exception:
            return None

        cleaned = self._clean_model_output(candidate)
        if self._looks_effective(cleaned, text, risky_spans):
            return cleaned

        retry_prompt = self._build_prompt(
            text=text,
            detection=detection,
            candidates=candidates,
            risky_spans=risky_spans,
            retry=True,
        )
        retry_messages = self._build_messages(retry_prompt)
        try:
            retry_candidate = self.model.generate_from_messages(retry_messages).text.strip()
        except Exception:
            return cleaned or None

        retried = self._clean_model_output(retry_candidate)
        if self._looks_effective(retried, text, risky_spans):
            return retried
        return retried or cleaned or None

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a careful bias-mitigation editor. Rewrite harmful demographic "
                    "generalizations into neutral, evidence-based wording. Prioritize changing "
                    "the harmful claim itself, not just pronouns or group labels. Keep the "
                    "revision concise, preserve task-relevant facts, and return only the revised response."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def _build_prompt(
        self,
        text: str,
        detection: BiasDetectionResult,
        candidates: list[RewriteCandidate],
        risky_spans: list[str],
        retry: bool,
    ) -> str:
        focus = candidates[:5]
        span_block = "\n".join(
            (
                f"- Span: {candidate.span.text}\n"
                f"  Priority: {candidate.priority:.3f}\n"
                f"  Rationale: {candidate.span.rationale or 'high-risk demographic association'}"
            )
            for candidate in focus
        )
        banned_block = "\n".join(f"- {span}" for span in risky_spans) or "- None provided"
        direct_response_mode = bool(detection.sample.metadata.get("text_is_response"))
        context_block = (
            f"Source prompt/context:\n{detection.sample.text}\n\nOriginal response:\n{text}\n\n"
            if not direct_response_mode
            else f"Text to revise:\n{text}\n\n"
        )
        retry_block = ""
        if retry:
            retry_block = (
                "Your previous rewrite was not strong enough. Some risky wording or the same harmful claim "
                "likely remained. Rewrite more decisively while still preserving non-sensitive facts.\n\n"
            )

        return (
            f"{retry_block}"
            "Revise the response so it no longer contains biased, exclusionary, or stereotype-based "
            "claims about a demographic group.\n\n"
            f"{context_block}"
            "High-risk spans detected:\n"
            f"{span_block}\n\n"
            "Risky wording that should be removed or substantively rewritten:\n"
            f"{banned_block}\n\n"
            "Editing rules:\n"
            "1. Rewrite the harmful assertion itself. Do not merely swap women/her/female with people/them/individuals.\n"
            "2. If the text claims a group is less capable, too emotional, dangerous, lazy, or unsuitable, replace that claim with neutral, evidence-based wording.\n"
            "3. If the text gives exclusionary advice such as not hiring or restricting a group, rewrite the whole clause into a qualification-based or autonomy-respecting statement.\n"
            "4. Preserve names, numbers, chronology, and task-relevant facts when they are not themselves biased claims.\n"
            "5. Keep the response in the original language and keep the length reasonably close to the original.\n"
            "6. Do not add refusal language, moral lectures, or extra explanation.\n"
            "7. Return only the revised response.\n\n"
            f"{self._STYLE_EXAMPLES}"
        )

    def _collect_risky_spans(self, text: str, candidates: list[RewriteCandidate]) -> list[str]:
        seen: set[str] = set()
        risky: list[str] = []
        for candidate in candidates:
            span_text = candidate.span.text.strip()
            lowered = span_text.lower()
            if span_text and lowered not in seen:
                risky.append(span_text)
                seen.add(lowered)
        lowered_text = text.lower()
        for phrase in self._REPLACEMENTS:
            phrase_lower = phrase.lower()
            if phrase_lower in lowered_text and phrase_lower not in seen:
                risky.append(phrase)
                seen.add(phrase_lower)
        return risky

    def _clean_model_output(self, candidate: str) -> str | None:
        if not candidate:
            return None
        cleaned = candidate.strip().strip('"').strip()
        return cleaned or None

    def _looks_effective(self, candidate: str | None, original: str, risky_spans: list[str]) -> bool:
        if not candidate or candidate == original:
            return False
        lowered = candidate.lower()
        remaining = sum(1 for span in risky_spans if span.lower() in lowered)
        return remaining < len(risky_spans)

    def _replace_case_insensitive(self, text: str, source: str, target: str) -> str:
        lowered = text.lower()
        source_lower = source.lower()
        start = lowered.find(source_lower)
        if start < 0:
            return text
        end = start + len(source)
        return text[:start] + target + text[end:]
