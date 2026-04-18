from __future__ import annotations

from llmbias.config import ExperimentConfig
from llmbias.models.base import BlackBoxLLM
from llmbias.pipelines.detection_pipeline import DetectionPipeline
from llmbias.schemas import PipelineResult, PromptSample


class EndToEndBiasPipeline:
    def __init__(self, model: BlackBoxLLM, config: ExperimentConfig) -> None:
        self.detection_pipeline = DetectionPipeline(model=model, config=config.detection)

    def run(self, sample: PromptSample) -> PipelineResult:
        detection = self.detection_pipeline.run(sample)
        # Temporarily stop after bias detection and skip the rewrite stage.
        return PipelineResult(detection=detection, rewrite=None)
