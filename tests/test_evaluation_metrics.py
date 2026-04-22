from __future__ import annotations

import unittest

from Evaluation.detection_file_metrics import DetectionMetricsRunner


class FakeSentimentEvaluator:
    def score_text(self, text: str):
        base = 0.6 if "counter" in text else 0.1
        label = "pos" if base >= 0.5 else "neu"
        return {
            "pos": base,
            "neu": 1.0 - base,
            "neg": 0.0,
            "compound": base,
            "label": label,
            "polarity": 1.0 if label == "pos" else 0.0,
        }


class FakeToxicityEvaluator:
    def score_text(self, text: str):
        score = 0.8 if "toxic" in text else 0.1
        return {
            "toxicity": score,
            "severe_toxicity": 0.0,
            "obscene": 0.0,
            "threat": 0.0,
            "insult": 0.0,
            "identity_attack": 0.0,
            "max_toxicity": score,
            "toxic": score >= 0.5,
        }


class FakeRegardEvaluator:
    def score_text(self, text: str):
        negative = 0.7 if "counter" in text else 0.1
        positive = 0.1 if "counter" in text else 0.6
        neutral = 0.2
        other = 0.0
        label = "negative" if negative > positive else "positive"
        return {
            "negative": negative,
            "neutral": neutral,
            "positive": positive,
            "other": other,
            "label": label,
            "regard_score": positive - negative,
        }


class DetectionMetricsRunnerTests(unittest.TestCase):
    def test_runner_computes_original_counterfactual_deltas(self) -> None:
        runner = DetectionMetricsRunner(
            sentiment_evaluator=FakeSentimentEvaluator(),
            toxicity_evaluator=FakeToxicityEvaluator(),
            regard_evaluator=FakeRegardEvaluator(),
        )
        row = {
            "original_response": {"text": "plain original"},
            "counterfactual_responses": [
                {"text": "counter toxic response"},
                {"text": "counter response"},
            ],
            "counterfactual_outcomes": [
                {"counterfactual": {"counterfactual_text": "cf1"}},
                {"counterfactual": {"counterfactual_text": "cf2"}},
            ],
        }

        result = runner.evaluate_row(row)

        self.assertIn("original_metrics", result)
        self.assertEqual(len(result["counterfactual_metrics"]), 2)
        first = result["counterfactual_metrics"][0]
        self.assertIn("sentiment_vader", first["metric_deltas"])
        self.assertIn("toxicity_bert", first["metric_deltas"])
        self.assertIn("regard_classifier", first["metric_deltas"])
        self.assertAlmostEqual(
            first["metric_deltas"]["sentiment_vader"]["primary"]["signed_delta"],
            0.5,
        )
        self.assertAlmostEqual(
            first["metric_deltas"]["toxicity_bert"]["primary"]["signed_delta"],
            0.7,
        )
        self.assertLess(
            first["metric_deltas"]["regard_classifier"]["primary"]["signed_delta"],
            0.0,
        )
        self.assertEqual(result["summary"]["sentiment_vader"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
