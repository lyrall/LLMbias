from __future__ import annotations

from llmbias.schemas import RewriteResult


def aggregate_tradeoff_score(result: RewriteResult, lambda_value: float = 0.5) -> float:
    fairness = result.fairness_gain
    preservation = result.preservation_score
    return lambda_value * fairness + (1.0 - lambda_value) * preservation

