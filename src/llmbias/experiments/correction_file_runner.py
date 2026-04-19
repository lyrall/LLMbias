from __future__ import annotations

import json
from pathlib import Path

from llmbias.evaluation.metrics import aggregate_tradeoff_score
from llmbias.pipelines.correction_pipeline import CorrectionPipeline
from llmbias.schemas import BiasDetectionResult


class CorrectionFileRunner:
    def __init__(self, pipeline: CorrectionPipeline, lambda_values: list[float] | None = None) -> None:
        self.pipeline = pipeline
        self.lambda_values = lambda_values or [0.3, 0.5, 0.7]

    def run_file(self, input_path: str, output_path: str | None = None) -> list[dict]:
        path = Path(input_path)
        results: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                results.append(self.run_record(payload))

        if output_path:
            self._write_jsonl(output_path, results)
        return results

    def run_record(self, payload: dict) -> dict:
        detection_payload = payload.get("detection", payload)
        detection = BiasDetectionResult.from_dict(dict(detection_payload))
        rewrite = self.pipeline.run(detection)

        if "detection" in payload:
            result = dict(payload)
            result["detection"] = detection.to_dict()
        else:
            result = {"detection": detection.to_dict()}

        result["rewrite"] = None if rewrite is None else rewrite.to_dict()
        result["tradeoff_scores"] = {}
        if rewrite is not None:
            result["tradeoff_scores"] = {
                f"lambda_{value}": aggregate_tradeoff_score(rewrite, value)
                for value in self.lambda_values
            }
        return result

    def _write_jsonl(self, output_path: str, results: list[dict]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
