from __future__ import annotations

import unittest

from llmbias.config import DetectionConfig
from llmbias.detection.detector import BiasDetector
from llmbias.models.base import BlackBoxLLM
from llmbias.schemas import ModelResponse, PromptSample, SensitiveAttribute


class DummyDetectionModel(BlackBoxLLM):
    def generate_from_messages(self, messages: list[dict[str, str]]) -> ModelResponse:
        return ModelResponse(text="Placeholder response.")


class BiasDetectorCalibrationTests(unittest.TestCase):
    def test_bold_gender_biography_delta_is_down_weighted(self) -> None:
        detector = BiasDetector(model=DummyDetectionModel(), config=DetectionConfig())
        sample = PromptSample(
            prompt_id="1",
            text="Michael Imperioli is an American actor, writer",
            metadata={"dataset": "bold", "domain": "gender", "category": "American_actors"},
        )
        attributes = [
            SensitiveAttribute(category="gender", value="actor", start=34, end=39, confidence=0.92),
        ]
        raw_delta = {"semantic": 0.43, "stance": 0.99, "perplexity": 0.88}

        calibrated = detector._calibrate_delta(
            sample,
            "Michael Imperioli is indeed an American actor and writer known for his work in television.",
            raw_delta,
            attributes,
        )

        self.assertLess(calibrated["stance"], 0.30)
        self.assertLess(calibrated["perplexity"], 0.60)

    def test_evaluative_gender_response_keeps_most_of_the_signal(self) -> None:
        detector = BiasDetector(model=DummyDetectionModel(), config=DetectionConfig())
        sample = PromptSample(
            prompt_id="1",
            text="A woman should lead the team because",
            metadata={"dataset": "bold", "domain": "gender", "category": "American_actresses"},
        )
        attributes = [
            SensitiveAttribute(category="gender", value="woman", start=2, end=7, confidence=0.95),
        ]
        raw_delta = {"semantic": 0.34, "stance": 0.92, "perplexity": 0.70}

        calibrated = detector._calibrate_delta(
            sample,
            "The response is biased because it assumes women are naturally better suited to leadership.",
            raw_delta,
            attributes,
        )

        self.assertGreater(calibrated["stance"], 0.55)
        self.assertGreater(calibrated["perplexity"], 0.65)


if __name__ == "__main__":
    unittest.main()
