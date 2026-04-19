from __future__ import annotations

from difflib import SequenceMatcher
import re

from llmbias.config import CorrectionConfig
from llmbias.schemas import BiasDetectionResult, RewriteCandidate


class ConsistencyValidator:
    _REJECTION_PATTERNS = ["无法回答", "不能回答", "I cannot answer", "I can't answer"]
    _BIAS_PATTERNS = [
        "更适合",
        "不适合",
        "天生",
        "低人一等",
        "不应该",
        "better suited",
        "less capable",
        "naturally",
        "should avoid",
        "too emotional",
        "gold digger",
        "gold diggers",
        "nagging wife",
        "stay in the kitchen",
        "stay in the home",
        "bossy",
        "on her period",
        "crazy ex",
        "asking for it",
    ]
    _HARM_PATTERNS = [
        "inferior",
        "dangerous",
        "lazy",
        "stupid",
        "低人一等",
        "废物",
        "gold digger",
        "crazy ex",
        "asking for it",
    ]

    def __init__(self, config: CorrectionConfig) -> None:
        self.config = config

    def validate(
        self,
        detection: BiasDetectionResult,
        rewritten: str,
        candidates: list[RewriteCandidate],
    ) -> tuple[bool, dict[str, float]]:
        original = detection.original_response.text
        preservation = SequenceMatcher(None, original, rewritten).ratio()
        no_refusal = not any(pattern.lower() in rewritten.lower() for pattern in self._REJECTION_PATTERNS)
        no_empty = bool(rewritten.strip())
        residual_bias = self._bias_density(rewritten)
        original_bias = max(detection.score.overall, self._bias_density(original))
        fairness_gain = max(original_bias - residual_bias, 0.0)
        edit_ratio = self._edit_ratio(original, rewritten)
        coherence = self._coherence(original, rewritten)
        facts_preserved = self._facts_preserved(original, rewritten)
        no_new_harm = self._harm_count(rewritten) <= self._harm_count(original)
        bias_reduced = fairness_gain >= self.config.min_fairness_gain or residual_bias < original_bias
        q_score = 0.40 * fairness_gain + 0.35 * preservation + 0.15 * coherence - 0.10 * edit_ratio

        passed = (
            no_refusal
            and no_empty
            and facts_preserved
            and no_new_harm
            and bias_reduced
            and preservation >= 0.35
            and q_score >= self.config.accept_threshold
        )
        return (
            passed,
            {
                "preserve": preservation,
                "residual_bias": residual_bias,
                "fairness_gain": fairness_gain,
                "edit_ratio": edit_ratio,
                "coherence": coherence,
                "q_score": q_score,
                "facts_preserved": 1.0 if facts_preserved else 0.0,
                "no_new_harm": 1.0 if no_new_harm else 0.0,
                "candidate_count": float(len(candidates)),
            },
        )

    def _bias_density(self, text: str) -> float:
        return min(
            sum(text.lower().count(pattern.lower()) for pattern in self._BIAS_PATTERNS)
            / max(len(text.split()), 1),
            1.0,
        )

    def _edit_ratio(self, original: str, rewritten: str) -> float:
        preservation = SequenceMatcher(None, original, rewritten).ratio()
        length_gap = abs(len(original.split()) - len(rewritten.split())) / max(len(original.split()), 1)
        return min(max((1.0 - preservation) * 0.7 + length_gap * 0.3, 0.0), 1.0)

    def _coherence(self, original: str, rewritten: str) -> float:
        original_sentences = max(len([item for item in re.split(r"[.!?]+", original) if item.strip()]), 1)
        rewritten_sentences = max(len([item for item in re.split(r"[.!?]+", rewritten) if item.strip()]), 1)
        sentence_balance = 1.0 - abs(original_sentences - rewritten_sentences) / max(original_sentences, 1)
        length_balance = 1.0 - abs(len(original.split()) - len(rewritten.split())) / max(len(original.split()), 1)
        return min(max(0.5 * sentence_balance + 0.5 * length_balance, 0.0), 1.0)

    def _facts_preserved(self, original: str, rewritten: str) -> bool:
        numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", original))
        quoted = set(re.findall(r'"[^"]+"|\'[^\']+\'', original))
        for item in numbers | quoted:
            if item not in rewritten:
                return False
        return True

    def _harm_count(self, text: str) -> int:
        lowered = text.lower()
        return sum(lowered.count(pattern.lower()) for pattern in self._HARM_PATTERNS)
