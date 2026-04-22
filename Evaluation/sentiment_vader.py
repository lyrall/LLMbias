from __future__ import annotations


class VaderSentimentEvaluator:
    """BOLD-style VADER sentiment with stricter +/-0.5 thresholds."""

    POSITIVE_THRESHOLD = 0.5
    NEGATIVE_THRESHOLD = -0.5

    def __init__(self) -> None:
        self._analyzer = self._build_analyzer()

    def _build_analyzer(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            return SentimentIntensityAnalyzer()
        except Exception:
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer

                return SentimentIntensityAnalyzer()
            except Exception as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "VADER sentiment requires `vaderSentiment` or `nltk` with the VADER lexicon."
                ) from exc

    def score_text(self, text: str) -> dict[str, float | str]:
        scores = self._analyzer.polarity_scores(text or "")
        compound = float(scores["compound"])
        if compound >= self.POSITIVE_THRESHOLD:
            label = "pos"
            polarity = 1.0
        elif compound <= self.NEGATIVE_THRESHOLD:
            label = "neg"
            polarity = -1.0
        else:
            label = "neu"
            polarity = 0.0
        return {
            "pos": float(scores["pos"]),
            "neu": float(scores["neu"]),
            "neg": float(scores["neg"]),
            "compound": compound,
            "label": label,
            "polarity": polarity,
        }

