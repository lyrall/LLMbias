from __future__ import annotations

from llmbias.config import DetectionConfig
from llmbias.detection.detector import BiasDetector
from llmbias.models.base import BlackBoxLLM
from llmbias.schemas import BiasDetectionResult, PromptSample


class DetectionPipeline:
    """Independent pipeline for research content one."""

    def __init__(self, model: BlackBoxLLM, config: DetectionConfig) -> None:
        self.detector = BiasDetector(model=model, config=config)

    def run(self, sample: PromptSample) -> BiasDetectionResult:
        return self.detector.run(sample)
