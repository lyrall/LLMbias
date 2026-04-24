from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "matplotlib is required for plotting. Install it with: pip install matplotlib"
    ) from exc


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

HISTOGRAM_COUNT_SCALE = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sentiment/toxicity experiment figures from a detection JSONL file."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the evaluation JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for figures and summary files. Defaults to outputs/figures/<input_stem>.",
    )
    return parser.parse_args()


def safe_get(obj: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = obj
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        evaluation = row.get("evaluation", {})
        summary = evaluation.get("summary", {})

        orig_sent = safe_get(
            evaluation, "original_metrics", "sentiment_vader", "compound"
        )
        orig_tox = safe_get(
            evaluation, "original_metrics", "toxicity_bert", "max_toxicity"
        )
        sent_mean_signed = safe_get(
            summary, "sentiment_vader", "mean_signed_delta", default=None
        )
        sent_mean_abs = safe_get(
            summary, "sentiment_vader", "mean_abs_delta", default=None
        )
        tox_mean_signed = safe_get(
            summary, "toxicity_bert", "mean_signed_delta", default=None
        )
        tox_mean_abs = safe_get(
            summary, "toxicity_bert", "mean_abs_delta", default=None
        )

        records.append(
            {
                "prompt_id": safe_get(row, "sample", "prompt_id"),
                "category": safe_get(row, "sample", "metadata", "category"),
                "is_biased": bool(row.get("is_biased", False)),
                "judge_confidence": row.get("judge_confidence", 0.0),
                "orig_sent": orig_sent,
                "orig_tox": orig_tox,
                "sent_mean_signed_delta": sent_mean_signed,
                "sent_mean_abs_delta": sent_mean_abs,
                "tox_mean_signed_delta": tox_mean_signed,
                "tox_mean_abs_delta": tox_mean_abs,
            }
        )
    return records


def summarize(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    p90_index = int(0.9 * (len(ordered) - 1))
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": ordered[0],
        "max": ordered[-1],
        "p90": ordered[p90_index],
    }


def split_present(records: list[dict[str, Any]], key: str) -> list[float]:
    return [record[key] for record in records if isinstance(record.get(key), (int, float))]


def scaled_hist_weights(values: list[float]) -> list[float]:
    return [HISTOGRAM_COUNT_SCALE] * len(values)


def save_summary(records: list[dict[str, Any]], output_dir: Path) -> None:
    paired = [r for r in records if r["sent_mean_abs_delta"] is not None]
    biased = [r for r in paired if r["is_biased"]]
    non_biased = [r for r in paired if not r["is_biased"]]

    summary = {
        "total_samples": len(records),
        "paired_counterfactual_samples": len(paired),
        "biased_samples": sum(1 for r in records if r["is_biased"]),
        "original_sentiment_compound": summarize(split_present(records, "orig_sent")),
        "original_max_toxicity": summarize(split_present(records, "orig_tox")),
        "sentiment_mean_signed_delta": summarize(
            split_present(paired, "sent_mean_signed_delta")
        ),
        "sentiment_mean_abs_delta": summarize(split_present(paired, "sent_mean_abs_delta")),
        "toxicity_mean_signed_delta": summarize(
            split_present(paired, "tox_mean_signed_delta")
        ),
        "toxicity_mean_abs_delta": summarize(split_present(paired, "tox_mean_abs_delta")),
        "biased_sentiment_abs_delta": summarize(split_present(biased, "sent_mean_abs_delta")),
        "non_biased_sentiment_abs_delta": summarize(
            split_present(non_biased, "sent_mean_abs_delta")
        ),
        "biased_toxicity_abs_delta": summarize(split_present(biased, "tox_mean_abs_delta")),
        "non_biased_toxicity_abs_delta": summarize(
            split_present(non_biased, "tox_mean_abs_delta")
        ),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_top_cases(records: list[dict[str, Any]], output_dir: Path) -> None:
    paired = [r for r in records if r["sent_mean_abs_delta"] is not None]
    top_sent = sorted(
        paired, key=lambda item: item["sent_mean_abs_delta"], reverse=True
    )[:10]
    top_tox = sorted(
        paired, key=lambda item: item["tox_mean_abs_delta"], reverse=True
    )[:10]

    with (output_dir / "top_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "list_name",
                "prompt_id",
                "category",
                "is_biased",
                "judge_confidence",
                "sent_mean_signed_delta",
                "sent_mean_abs_delta",
                "tox_mean_signed_delta",
                "tox_mean_abs_delta",
            ]
        )
        for list_name, rows in [
            ("top_sentiment_delta", top_sent),
            ("top_toxicity_delta", top_tox),
        ]:
            for row in rows:
                writer.writerow(
                    [
                        list_name,
                        row["prompt_id"],
                        row["category"],
                        row["is_biased"],
                        row["judge_confidence"],
                        row["sent_mean_signed_delta"],
                        row["sent_mean_abs_delta"],
                        row["tox_mean_signed_delta"],
                        row["tox_mean_abs_delta"],
                    ]
                )


def plot_original_distributions(records: list[dict[str, Any]], output_dir: Path) -> None:
    sentiment = split_present(records, "orig_sent")
    toxicity = split_present(records, "orig_tox")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(
        sentiment,
        bins=30,
        weights=scaled_hist_weights(sentiment),
        color="#2563eb",
        alpha=0.85,
        edgecolor="white",
    )
    axes[0].set_title("原始回复情感分布")
    axes[0].set_xlabel("VADER 综合得分")
    axes[0].set_ylabel("数量")

    axes[1].hist(
        toxicity,
        bins=30,
        weights=scaled_hist_weights(toxicity),
        color="#dc2626",
        alpha=0.85,
        edgecolor="white",
    )
    axes[1].set_title("原始回复毒性分布")
    axes[1].set_xlabel("最大毒性")
    axes[1].set_ylabel("数量")

    fig.tight_layout()
    fig.savefig(output_dir / "original_metric_distributions.png", dpi=220)
    plt.close(fig)


def plot_delta_distributions(records: list[dict[str, Any]], output_dir: Path) -> None:
    paired = [r for r in records if r["sent_mean_abs_delta"] is not None]
    sent_signed = split_present(paired, "sent_mean_signed_delta")
    sent_abs = split_present(paired, "sent_mean_abs_delta")
    tox_signed = split_present(paired, "tox_mean_signed_delta")
    tox_abs = split_present(paired, "tox_mean_abs_delta")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(
        sent_signed,
        bins=30,
        weights=scaled_hist_weights(sent_signed),
        color="#0f766e",
        alpha=0.85,
        edgecolor="white",
    )
    axes[0, 0].set_title("情感变化量（有符号）")
    axes[0, 0].set_xlabel("反事实 - 原始回复")
    axes[0, 0].set_ylabel("数量")

    axes[0, 1].hist(
        sent_abs,
        bins=30,
        weights=scaled_hist_weights(sent_abs),
        color="#0891b2",
        alpha=0.85,
        edgecolor="white",
    )
    axes[0, 1].set_title("情感变化量（绝对值）")
    axes[0, 1].set_xlabel("|反事实 - 原始回复|")
    axes[0, 1].set_ylabel("数量")

    axes[1, 0].hist(
        tox_signed,
        bins=30,
        weights=scaled_hist_weights(tox_signed),
        color="#b45309",
        alpha=0.85,
        edgecolor="white",
    )
    axes[1, 0].set_title("毒性变化量（有符号）")
    axes[1, 0].set_xlabel("反事实 - 原始回复")
    axes[1, 0].set_ylabel("数量")

    tox_abs_log = [math.log10(value + 1e-8) for value in tox_abs]
    axes[1, 1].hist(
        tox_abs_log,
        bins=30,
        weights=scaled_hist_weights(tox_abs_log),
        color="#7c3aed",
        alpha=0.85,
        edgecolor="white",
    )
    axes[1, 1].set_title("毒性变化量绝对值（log10）")
    axes[1, 1].set_xlabel("log10(|反事实 - 原始回复| + 1e-8)")
    axes[1, 1].set_ylabel("数量")

    fig.tight_layout()
    fig.savefig(output_dir / "delta_distributions.png", dpi=220)
    plt.close(fig)


def plot_group_comparison(records: list[dict[str, Any]], output_dir: Path) -> None:
    paired = [r for r in records if r["sent_mean_abs_delta"] is not None]
    biased = [r for r in paired if r["is_biased"]]
    non_biased = [r for r in paired if not r["is_biased"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].boxplot(
        [
            split_present(non_biased, "sent_mean_abs_delta"),
            split_present(biased, "sent_mean_abs_delta"),
        ],
        labels=["无偏见", "有偏见"],
        patch_artist=True,
        boxprops={"facecolor": "#93c5fd"},
        medianprops={"color": "#1e3a8a", "linewidth": 1.8},
    )
    axes[0].set_title("情感变化量绝对值")
    axes[0].set_ylabel("|反事实 - 原始回复|")

    axes[1].boxplot(
        [
            split_present(non_biased, "tox_mean_abs_delta"),
            split_present(biased, "tox_mean_abs_delta"),
        ],
        labels=["无偏见", "有偏见"],
        patch_artist=True,
        boxprops={"facecolor": "#fca5a5"},
        medianprops={"color": "#7f1d1d", "linewidth": 1.8},
    )
    axes[1].set_title("毒性变化量绝对值")
    axes[1].set_ylabel("|反事实 - 原始回复|")
    axes[1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_dir / "bias_group_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (Path("outputs") / "figures" / input_path.stem).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    records = build_records(rows)

    save_summary(records, output_dir)
    save_top_cases(records, output_dir)
    plot_original_distributions(records, output_dir)
    plot_delta_distributions(records, output_dir)
    plot_group_comparison(records, output_dir)

    print(f"Saved figures and summaries to: {output_dir}")


if __name__ == "__main__":
    main()
