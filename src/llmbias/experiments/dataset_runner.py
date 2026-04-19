from __future__ import annotations

import json
from pathlib import Path

from llmbias.datasets import BBQDatasetLoader, BOLDDatasetLoader, ToxiGenDatasetLoader
from llmbias.experiments.end_to_end_runner import EndToEndRunner


class DatasetRunner:
    def __init__(self, runner: EndToEndRunner) -> None:
        self.runner = runner

    def run_bbq(
        self,
        dataset_path: str,
        split: str = "test",
        subset: str = "",
        limit: int | None = None,
        output_path: str | None = None,
    ) -> list[dict]:
        loader = BBQDatasetLoader(dataset_path)
        samples = loader.load(split=split, subset=subset, limit=limit)
        results = [self.runner.run_sample(sample) for sample in samples]
        if output_path:
            self._write_jsonl(output_path, results)
        return results

    def run_bold(
        self,
        dataset_path: str,
        subset: str = "",
        limit: int | None = None,
        output_path: str | None = None,
    ) -> list[dict]:
        loader = BOLDDatasetLoader(dataset_path)
        samples = loader.load(subset=subset, limit=limit)
        results = [self.runner.run_sample(sample) for sample in samples]
        if output_path:
            self._write_jsonl(output_path, results)
        return results

    def run_toxigen(
        self,
        dataset_path: str,
        subset: str = "",
        limit: int | None = None,
        output_path: str | None = None,
    ) -> list[dict]:
        loader = ToxiGenDatasetLoader(dataset_path)
        samples = loader.load(subset=subset, limit=limit)
        results = [self.runner.run_sample(sample) for sample in samples]
        if output_path:
            self._write_jsonl(output_path, results)
        return results

    def _write_jsonl(self, output_path: str, results: list[dict]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
