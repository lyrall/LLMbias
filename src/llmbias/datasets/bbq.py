from __future__ import annotations

import json
from pathlib import Path

from llmbias.schemas import PromptSample


class BBQDatasetLoader:
    """Load local BBQ JSONL files and convert them into prompt samples."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load(self, split: str = "test", subset: str = "", limit: int | None = None) -> list[PromptSample]:
        files = self._resolve_files(split=split, subset=subset)
        samples: list[PromptSample] = []
        for file_path in files:
            samples.extend(self._load_file(file_path))
            if limit is not None and len(samples) >= limit:
                return samples[:limit]
        return samples

    def _resolve_files(self, split: str, subset: str) -> list[Path]:
        if not self.root.exists():
            raise FileNotFoundError(f"BBQ dataset directory does not exist: {self.root}")

        if subset:
            candidates = [
                self.root / f"{subset}.jsonl",
                self.root / split / f"{subset}.jsonl",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return [candidate]
            matched = sorted(self.root.rglob(f"{subset}.jsonl"))
            if matched:
                return matched
            raise FileNotFoundError(f"Could not find BBQ subset '{subset}' under {self.root}")

        split_dir = self.root / split
        if split_dir.exists():
            files = sorted(split_dir.glob("*.jsonl"))
            if files:
                return files

        files = sorted(self.root.glob("*.jsonl"))
        if files:
            return files

        raise FileNotFoundError(
            f"No BBQ JSONL files found under {self.root}. Expected *.jsonl or a '{split}' subdirectory."
        )

    def _load_file(self, path: Path) -> list[PromptSample]:
        samples: list[PromptSample] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                samples.append(self._record_to_sample(record, path, line_number))
        return samples

    def _record_to_sample(self, record: dict, source: Path, line_number: int) -> PromptSample:
        context = str(record.get("context", "")).strip()
        question = str(record.get("question", "")).strip()
        answers = [str(record.get(f"ans{i}", "")).strip() for i in range(3)]
        prompt_text = (
            "Please answer the multiple-choice question based on the context.\n\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"A. {answers[0]}\n"
            f"B. {answers[1]}\n"
            f"C. {answers[2]}\n"
            "Respond with the best answer and a short explanation."
        )
        example_id = record.get("example_id", f"{source.stem}-{line_number}")
        metadata = {
            "dataset": "bbq",
            "source_file": str(source),
            "line_number": line_number,
            "question_index": record.get("question_index"),
            "category": record.get("category"),
            "context_condition": record.get("context_condition"),
            "question_polarity": record.get("question_polarity"),
            "label": record.get("label"),
            "answers": answers,
            "raw_record": record,
        }
        return PromptSample(prompt_id=f"bbq-{example_id}", text=prompt_text, metadata=metadata)
