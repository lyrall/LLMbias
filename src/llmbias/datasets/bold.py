from __future__ import annotations

import json
from pathlib import Path

from llmbias.schemas import PromptSample


class BOLDDatasetLoader:
    """Load local BOLD prompt files and convert them into prompt samples."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load(self, subset: str = "", limit: int | None = None) -> list[PromptSample]:
        files = self._resolve_files(subset=subset)
        samples: list[PromptSample] = []
        for file_path in files:
            samples.extend(self._load_file(file_path))
            if limit is not None and len(samples) >= limit:
                return samples[:limit]
        return samples

    def _resolve_files(self, subset: str) -> list[Path]:
        if not self.root.exists():
            raise FileNotFoundError(f"BOLD dataset path does not exist: {self.root}")

        if self.root.is_file():
            return [self.root]

        if subset:
            normalized = subset.strip().lower()
            candidates = [
                self.root / f"{normalized}_prompt.json",
                self.root / f"{normalized}_prompt_wiki.json",
                self.root / f"{normalized}.json",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return [candidate]

            matched = sorted(self.root.rglob(f"{normalized}*.json"))
            if matched:
                return matched

            raise FileNotFoundError(f"Could not find BOLD subset '{subset}' under {self.root}")

        files = sorted(self.root.glob("*prompt*.json"))
        if files:
            return files

        files = sorted(self.root.glob("*.json"))
        if files:
            return files

        raise FileNotFoundError(f"No BOLD JSON files found under {self.root}")

    def _load_file(self, path: Path) -> list[PromptSample]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            return self._load_nested_prompt_map(payload, path)
        if isinstance(payload, list):
            return self._load_flat_records(payload, path)
        raise ValueError(f"Unsupported BOLD file format in {path}")

    def _load_nested_prompt_map(self, payload: dict, source: Path) -> list[PromptSample]:
        samples: list[PromptSample] = []
        domain = self._infer_domain(source)
        for category, entries in payload.items():
            if not isinstance(entries, dict):
                continue
            for name, prompts in entries.items():
                if not isinstance(prompts, list):
                    continue
                for prompt_index, prompt in enumerate(prompts):
                    prompt_text = str(prompt).strip()
                    if not prompt_text:
                        continue
                    sample_id = f"bold-{domain}-{category}-{name}-{prompt_index}"
                    metadata = {
                        "dataset": "bold",
                        "domain": domain,
                        "category": category,
                        "name": name,
                        "prompt_index": prompt_index,
                        "source_file": str(source),
                    }
                    samples.append(
                        PromptSample(
                            prompt_id=self._sanitize_id(sample_id),
                            text=prompt_text,
                            metadata=metadata,
                        )
                    )
        return samples

    def _load_flat_records(self, payload: list, source: Path) -> list[PromptSample]:
        samples: list[PromptSample] = []
        domain = self._infer_domain(source)
        for row_index, record in enumerate(payload):
            if not isinstance(record, dict):
                continue
            prompts = record.get("prompts") or record.get("prompt") or []
            if isinstance(prompts, str):
                prompts = [prompts]
            if not isinstance(prompts, list):
                continue
            for prompt_index, prompt in enumerate(prompts):
                prompt_text = str(prompt).strip()
                if not prompt_text:
                    continue
                record_name = str(record.get("name", f"row-{row_index}"))
                category = str(record.get("category", ""))
                sample_id = f"bold-{domain}-{category}-{record_name}-{prompt_index}"
                metadata = {
                    "dataset": "bold",
                    "domain": record.get("domain", domain),
                    "category": category,
                    "name": record_name,
                    "prompt_index": prompt_index,
                    "source_file": str(source),
                    "raw_record": record,
                }
                samples.append(
                    PromptSample(
                        prompt_id=self._sanitize_id(sample_id),
                        text=prompt_text,
                        metadata=metadata,
                    )
                )
        return samples

    def _infer_domain(self, source: Path) -> str:
        stem = source.stem.lower()
        return stem.replace("_prompt_wiki", "").replace("_prompt", "")

    def _sanitize_id(self, value: str) -> str:
        return value.replace(" ", "_").replace("/", "_")
