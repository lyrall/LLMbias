from __future__ import annotations

from llmbias.config import ExperimentConfig
from llmbias.evaluation.metrics import aggregate_tradeoff_score
from llmbias.pipelines.end_to_end_pipeline import EndToEndBiasPipeline
from llmbias.schemas import PromptSample


class EndToEndRunner:
    def __init__(self, pipeline: EndToEndBiasPipeline, config: ExperimentConfig) -> None:
        self.pipeline = pipeline
        self.config = config

    def run_single(self, prompt: str, prompt_id: str = "demo-001") -> dict:
        result = self.pipeline.run(PromptSample(prompt_id=prompt_id, text=prompt))
        return self._serialize_result(result)

    def run_sample(self, sample: PromptSample) -> dict:
        result = self.pipeline.run(sample)
        return self._serialize_result(result)

    def _serialize_result(self, result) -> dict:
        payload = result.to_dict()
        if result.rewrite is not None:
            payload["tradeoff_scores"] = {
                f"lambda_{value}": aggregate_tradeoff_score(result.rewrite, value)
                for value in self.config.evaluation.lambda_values
            }
        else:
            payload["tradeoff_scores"] = {}
        return payload
