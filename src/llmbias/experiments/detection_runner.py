from __future__ import annotations

from llmbias.pipelines.detection_pipeline import DetectionPipeline
from llmbias.schemas import PromptSample


class DetectionRunner:
    def __init__(self, pipeline: DetectionPipeline) -> None:
        self.pipeline = pipeline

    def run_single(self, prompt: str, prompt_id: str = "detect-001") -> dict:
        result = self.pipeline.run(PromptSample(prompt_id=prompt_id, text=prompt))
        return result.to_dict()
