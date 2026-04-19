from __future__ import annotations

import re

from llmbias.config import CorrectionConfig
from llmbias.schemas import BiasDetectionResult, BiasSpan, RewriteCandidate


class PriorityRanker:
    def __init__(self, config: CorrectionConfig) -> None:
        self.config = config

    def rank(self, spans: list[BiasSpan], detection: BiasDetectionResult) -> list[RewriteCandidate]:
        candidates: list[RewriteCandidate] = []
        factual_support = self._factual_support(detection.judge_rationale)
        for span in spans:
            local_delta = float(span.metadata.get("local_delta", span.risk_score))
            support_ratio = float(span.metadata.get("support_ratio", span.confidence))
            toxicity = self._toxicity_proxy(span.text)
            bias_strength = min(
                max(
                    0.45 * span.risk_score
                    + 0.25 * local_delta
                    + 0.20 * detection.judge_confidence
                    + 0.10 * toxicity
                    - 0.20 * factual_support,
                    0.0,
                ),
                1.0,
            )
            consistency = min(max(0.55 * span.confidence + 0.45 * support_ratio, 0.0), 1.0)
            key_info_cost = self._key_info_cost(span.text)
            structure_cost = self._structure_cost(span, detection.original_response.text)
            length_cost = min(max(max(len(span.text) / 80.0, len(span.text.split()) / 20.0), 0.05), 1.0)
            edit_cost = min(0.45 * length_cost + 0.35 * key_info_cost + 0.20 * structure_cost, 1.0)
            preserve_bonus = min(max(1.0 - 0.6 * key_info_cost - 0.4 * structure_cost, 0.0), 1.0)
            priority = (
                self.config.fair_weight * bias_strength
                + self.config.confidence_weight * consistency
                + self.config.preserve_weight * preserve_bonus
                - self.config.edit_cost_weight * edit_cost
            )
            if priority < self.config.min_priority:
                continue
            candidates.append(
                RewriteCandidate(
                    span=span,
                    priority=min(priority, 1.0),
                    edit_cost=edit_cost,
                    metadata={
                        "bias_strength": bias_strength,
                        "consistency": consistency,
                        "preserve_bonus": preserve_bonus,
                        "factual_support": factual_support,
                        "length_cost": length_cost,
                        "key_info_cost": key_info_cost,
                        "structure_cost": structure_cost,
                    },
                )
            )
        ranked = sorted(candidates, key=lambda item: item.priority, reverse=True)
        return ranked[: self.config.max_candidate_spans]

    def _factual_support(self, rationale: str) -> float:
        match = re.search(r"\b(10|[0-9])\s*/\s*10\b", rationale)
        if not match:
            return 0.5
        return int(match.group(1)) / 10.0

    def _toxicity_proxy(self, text: str) -> float:
        lowered = text.lower()
        cues = [
            "不适合",
            "低人一等",
            "lazy",
            "dangerous",
            "inferior",
            "unfit",
            "incapable",
            "too emotional",
            "gold digger",
            "nagging wife",
            "bossy",
            "on her period",
        ]
        count = sum(lowered.count(item.lower()) for item in cues)
        return min(count / 2.0, 1.0)

    def _key_info_cost(self, text: str) -> float:
        has_number = bool(re.search(r"\b\d+(?:\.\d+)?\b", text))
        has_quote = '"' in text or "'" in text
        has_title_case = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text))
        score = 0.0
        if has_number:
            score += 0.4
        if has_quote:
            score += 0.3
        if has_title_case:
            score += 0.3
        return min(score, 1.0)

    def _structure_cost(self, span: BiasSpan, original_text: str) -> float:
        if span.start is None or not original_text:
            return 0.2
        position = span.start / max(len(original_text), 1)
        if position >= 0.65:
            return 0.8
        if position <= 0.2:
            return 0.4
        return 0.2
