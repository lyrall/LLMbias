from __future__ import annotations

from llmbias.config import CorrectionConfig
from llmbias.schemas import BiasSpan, RewriteCandidate


class PriorityRanker:
    def __init__(self, config: CorrectionConfig) -> None:
        self.config = config

    def rank(self, spans: list[BiasSpan]) -> list[RewriteCandidate]:
        candidates: list[RewriteCandidate] = []
        for span in spans:
            edit_cost = min(max(len(span.text) / 80.0, 0.05), 1.0)
            priority = (
                self.config.fair_weight * span.risk_score
                + self.config.confidence_weight * span.confidence
                + self.config.preserve_weight * min(len(span.text.split()) / 12.0, 1.0)
                - self.config.edit_cost_weight * edit_cost
            )
            if priority < self.config.min_priority:
                continue
            candidates.append(
                RewriteCandidate(span=span, priority=min(priority, 1.0), edit_cost=edit_cost)
            )
        return sorted(candidates, key=lambda item: item.priority, reverse=True)
