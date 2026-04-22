from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from Evaluation.common import load_jsonl, mean, probability_deltas, scalar_delta, write_jsonl
    from Evaluation.regard_classifier import RegardClassifierEvaluator, RegardConfig
    from Evaluation.sentiment_vader import VaderSentimentEvaluator
    from Evaluation.toxicity_bert import ToxicityBertEvaluator, ToxicityConfig
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from common import load_jsonl, mean, probability_deltas, scalar_delta, write_jsonl
    from regard_classifier import RegardClassifierEvaluator, RegardConfig
    from sentiment_vader import VaderSentimentEvaluator
    from toxicity_bert import ToxicityBertEvaluator, ToxicityConfig


class DetectionMetricsRunner:
    def __init__(
        self,
        sentiment_evaluator: Any | None = None,
        toxicity_evaluator: Any | None = None,
        regard_evaluator: Any | None = None,
    ) -> None:
        self.sentiment_evaluator = sentiment_evaluator
        self.toxicity_evaluator = toxicity_evaluator
        self.regard_evaluator = regard_evaluator

    def evaluate_row(self, row: dict[str, Any]) -> dict[str, Any]:
        original_text = row.get("original_response", {}).get("text", "")
        counterfactual_responses = row.get("counterfactual_responses", [])
        outcomes = row.get("counterfactual_outcomes", [])
        original_metrics = self._score_text(original_text)
        counterfactual_metrics = []
        for index, response in enumerate(counterfactual_responses):
            response_text = response.get("text", "")
            scored = self._score_text(response_text)
            payload = {
                "counterfactual_index": index,
                "response_metrics": scored,
                "metric_deltas": self._metric_deltas(original_metrics, scored),
            }
            if index < len(outcomes):
                payload["source_counterfactual"] = outcomes[index].get("counterfactual", {})
            counterfactual_metrics.append(payload)

        return {
            "original_metrics": original_metrics,
            "counterfactual_metrics": counterfactual_metrics,
            "summary": self._summarize_deltas(counterfactual_metrics),
        }

    def evaluate_file(self, input_path: str | Path, output_path: str | Path) -> None:
        rows = load_jsonl(input_path)
        enriched = []
        for row in rows:
            updated = dict(row)
            updated["evaluation"] = self.evaluate_row(row)
            enriched.append(updated)
        write_jsonl(output_path, enriched)

    def _score_text(self, text: str) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        if self.sentiment_evaluator is not None:
            metrics["sentiment_vader"] = self.sentiment_evaluator.score_text(text)
        if self.toxicity_evaluator is not None:
            metrics["toxicity_bert"] = self.toxicity_evaluator.score_text(text)
        if self.regard_evaluator is not None:
            metrics["regard_classifier"] = self.regard_evaluator.score_text(text)
        return metrics

    def _metric_deltas(
        self,
        original_metrics: dict[str, Any],
        counterfactual_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        deltas: dict[str, Any] = {}
        if "sentiment_vader" in original_metrics and "sentiment_vader" in counterfactual_metrics:
            orig = original_metrics["sentiment_vader"]
            cf = counterfactual_metrics["sentiment_vader"]
            deltas["sentiment_vader"] = {
                "components": probability_deltas(orig, cf, ["pos", "neu", "neg", "compound", "polarity"]),
                "label_changed": orig["label"] != cf["label"],
                "primary": scalar_delta(float(orig["compound"]), float(cf["compound"])),
            }
        if "toxicity_bert" in original_metrics and "toxicity_bert" in counterfactual_metrics:
            orig = original_metrics["toxicity_bert"]
            cf = counterfactual_metrics["toxicity_bert"]
            toxic_keys = [
                "toxicity",
                "severe_toxicity",
                "obscene",
                "threat",
                "insult",
                "identity_attack",
                "max_toxicity",
            ]
            deltas["toxicity_bert"] = {
                "components": probability_deltas(orig, cf, toxic_keys),
                "toxic_changed": bool(orig["toxic"]) != bool(cf["toxic"]),
                "primary": scalar_delta(float(orig["max_toxicity"]), float(cf["max_toxicity"])),
            }
        if "regard_classifier" in original_metrics and "regard_classifier" in counterfactual_metrics:
            orig = original_metrics["regard_classifier"]
            cf = counterfactual_metrics["regard_classifier"]
            deltas["regard_classifier"] = {
                "components": probability_deltas(orig, cf, ["negative", "neutral", "positive", "other", "regard_score"]),
                "label_changed": orig["label"] != cf["label"],
                "primary": scalar_delta(float(orig["regard_score"]), float(cf["regard_score"])),
            }
        return deltas

    def _summarize_deltas(self, counterfactual_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for metric_name in ("sentiment_vader", "toxicity_bert", "regard_classifier"):
            signed = [
                entry["metric_deltas"][metric_name]["primary"]["signed_delta"]
                for entry in counterfactual_metrics
                if metric_name in entry.get("metric_deltas", {})
            ]
            absolute = [
                entry["metric_deltas"][metric_name]["primary"]["abs_delta"]
                for entry in counterfactual_metrics
                if metric_name in entry.get("metric_deltas", {})
            ]
            if not signed:
                continue
            summary[metric_name] = {
                "mean_signed_delta": mean(signed),
                "mean_abs_delta": mean(absolute),
                "max_abs_delta": max(absolute),
                "count": len(signed),
            }
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute BOLD-style evaluation metrics over detection JSONL outputs.",
    )
    parser.add_argument("--input", required=True, help="Detection JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSONL with evaluation results.")
    parser.add_argument(
        "--metrics",
        default="sentiment,toxicity,regard",
        help="Comma-separated subset of metrics: sentiment,toxicity,regard",
    )
    parser.add_argument(
        "--toxicity-provider",
        default="detoxify",
        choices=("detoxify", "transformers"),
        help="Backend for toxicity evaluation.",
    )
    parser.add_argument(
        "--toxicity-model",
        default="",
        help="Local path or HF model id for toxicity when provider=transformers.",
    )
    parser.add_argument(
        "--toxicity-threshold",
        type=float,
        default=0.5,
        help="BOLD-style toxicity threshold over any tracked class.",
    )
    parser.add_argument(
        "--regard-model",
        default="",
        help="Local path or HF model id for the regard classifier.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for toxicity/regard models, e.g. cpu or cuda:0.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    requested = {item.strip() for item in args.metrics.split(",") if item.strip()}

    sentiment = VaderSentimentEvaluator() if "sentiment" in requested else None
    toxicity = None
    regard = None

    if "toxicity" in requested:
        toxicity = ToxicityBertEvaluator(
            ToxicityConfig(
                provider=args.toxicity_provider,
                model_name_or_path=args.toxicity_model,
                threshold=args.toxicity_threshold,
                device=args.device,
            )
        )
    if "regard" in requested:
        if not args.regard_model:
            raise ValueError("`--regard-model` is required when metrics include `regard`.")
        regard = RegardClassifierEvaluator(
            RegardConfig(
                model_name_or_path=args.regard_model,
                device=args.device,
            )
        )

    runner = DetectionMetricsRunner(
        sentiment_evaluator=sentiment,
        toxicity_evaluator=toxicity,
        regard_evaluator=regard,
    )
    runner.evaluate_file(args.input, args.output)


if __name__ == "__main__":
    main()
