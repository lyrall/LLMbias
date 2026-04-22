from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    return [
        json.loads(line)
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def scalar_delta(original: float, counterfactual: float) -> dict[str, float]:
    signed_delta = counterfactual - original
    return {
        "signed_delta": signed_delta,
        "abs_delta": abs(signed_delta),
    }


def probability_deltas(
    original: dict[str, float],
    counterfactual: dict[str, float],
    keys: list[str],
) -> dict[str, dict[str, float]]:
    return {
        key: scalar_delta(float(original.get(key, 0.0)), float(counterfactual.get(key, 0.0)))
        for key in keys
    }


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)

