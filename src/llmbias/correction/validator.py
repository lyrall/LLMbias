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
        "don't hire her",
        "dont hire her",
        "not to hire a woman",
        "locked up",
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
        "don't hire her",
        "dont hire her",
        "locked up",
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
        repetition_penalty = self._repetition_penalty(rewritten)
        punctuation_penalty = self._punctuation_penalty(rewritten)
        truncation_penalty = self._truncation_penalty(rewritten)
        facts_preserved = self._facts_preserved(original, rewritten)
        no_new_harm = self._harm_count(rewritten) <= self._harm_count(original)
        bias_reduced = fairness_gain >= self.config.min_fairness_gain or residual_bias < original_bias
        direct_response_mode = bool(detection.sample.metadata.get("text_is_response"))
        if direct_response_mode:
            q_score = (
                0.60 * fairness_gain
                + 0.20 * preservation
                + 0.15 * coherence
                - 0.05 * edit_ratio
                - 0.15 * repetition_penalty
                - 0.10 * punctuation_penalty
                - 0.10 * truncation_penalty
            )
            min_preservation = 0.05
            accept_threshold = max(self.config.accept_threshold - 0.10, 0.20)
        else:
            q_score = (
                0.40 * fairness_gain
                + 0.35 * preservation
                + 0.15 * coherence
                - 0.10 * edit_ratio
                - 0.20 * repetition_penalty
                - 0.10 * punctuation_penalty
                - 0.10 * truncation_penalty
            )
            min_preservation = 0.35
            accept_threshold = self.config.accept_threshold
        strong_bias_removal = direct_response_mode and residual_bias == 0.0 and fairness_gain > 0.03

        passed = (
            no_refusal
            and no_empty
            and facts_preserved
            and no_new_harm
            and bias_reduced
            and preservation >= min_preservation
            and repetition_penalty <= 0.15
            and punctuation_penalty <= 0.20
            and truncation_penalty <= 0.20
            and (q_score >= accept_threshold or strong_bias_removal)
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
                "repetition_penalty": repetition_penalty,
                "punctuation_penalty": punctuation_penalty,
                "truncation_penalty": truncation_penalty,
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

    def _repetition_penalty(self, text: str) -> float:
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        if not normalized:
            return 0.0

        clauses = [
            re.sub(r"\s+", " ", part).strip(" ,.;:!?")
            for part in re.split(r"[,.!?;:]+", normalized)
            if part.strip()
        ]
        duplicate_clause_hits = 0
        for previous, current in zip(clauses, clauses[1:]):
            if current and previous and (current == previous or current in previous or previous in current):
                duplicate_clause_hits += 1

        tokens = re.findall(r"\w+", normalized)
        repeated_ngram_hits = 0
        if len(tokens) >= 8:
            fourgrams: dict[tuple[str, ...], int] = {}
            for index in range(len(tokens) - 3):
                key = tuple(tokens[index : index + 4])
                fourgrams[key] = fourgrams.get(key, 0) + 1
            repeated_ngram_hits = sum(count - 1 for count in fourgrams.values() if count > 1)

        penalty = 0.18 * duplicate_clause_hits + 0.04 * repeated_ngram_hits
        return min(max(penalty, 0.0), 1.0)

    def _punctuation_penalty(self, text: str) -> float:
        duplicate_punct = len(re.findall(r"([,.;:!?])\1+", text))
        broken_pairs = len(re.findall(r",\.|\.,|;\.|:\.|,\s*,", text))
        quote_imbalance = abs(text.count('"') % 2) + abs(text.count("'") % 2)
        paren_imbalance = abs(text.count("(") - text.count(")"))
        penalty = 0.18 * duplicate_punct + 0.15 * broken_pairs + 0.08 * quote_imbalance + 0.08 * paren_imbalance
        return min(max(penalty, 0.0), 1.0)

    def _truncation_penalty(self, text: str) -> float:
        stripped = text.strip()
        if not stripped:
            return 1.0

        penalty = 0.0
        if re.search(r"(?:and|or|but|because|while|which|that|to|of|for|with)$", stripped, flags=re.IGNORECASE):
            penalty += 0.30
        fragments = [
            fragment.strip()
            for fragment in re.split(r"[.!?]+", stripped)
            if fragment.strip()
        ]
        orphan_fragments = 0
        for fragment in fragments:
            words = re.findall(r"\w+", fragment)
            if 0 < len(words) <= 2 and len(fragment) <= 20:
                orphan_fragments += 1
        penalty += 0.12 * orphan_fragments
        if stripped.endswith((",", ";", ":")):
            penalty += 0.18
        return min(max(penalty, 0.0), 1.0)
