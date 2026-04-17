from __future__ import annotations

from difflib import SequenceMatcher


class ConsistencyValidator:
    _REJECTION_PATTERNS = ["无法回答", "不能回答", "I cannot answer", "I can't answer"]
    _BIAS_PATTERNS = ["更适合", "不适合", "天生", "better suited", "less capable"]

    def validate(self, original: str, rewritten: str) -> tuple[bool, dict[str, float]]:
        preservation = SequenceMatcher(None, original, rewritten).ratio()
        no_refusal = not any(pattern.lower() in rewritten.lower() for pattern in self._REJECTION_PATTERNS)
        no_empty = bool(rewritten.strip())
        residual_bias = min(
            sum(rewritten.lower().count(pattern.lower()) for pattern in self._BIAS_PATTERNS)
            / max(len(rewritten.split()), 1),
            1.0,
        )
        passed = no_refusal and no_empty and preservation >= 0.35 and residual_bias <= 0.2
        return passed, {"preserve": preservation, "residual_bias": residual_bias}
