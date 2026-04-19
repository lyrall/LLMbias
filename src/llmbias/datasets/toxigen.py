from __future__ import annotations

from pathlib import Path

from llmbias.schemas import PromptSample


class ToxiGenDatasetLoader:
    """Load local ToxiGen text files and convert them into prompt samples."""

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
            raise FileNotFoundError(f"ToxiGen dataset path does not exist: {self.root}")
        if self.root.is_file():
            return [self.root]
        if subset:
            normalized = subset.strip().lower()
            candidates = [
                self.root / f"{normalized}.txt",
                self.root / f"{normalized}_1k.txt",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return [candidate]
            matched = sorted(self.root.rglob(f"{normalized}*.txt"))
            if matched:
                return matched
            raise FileNotFoundError(f"Could not find ToxiGen subset '{subset}' under {self.root}")
        files = sorted(self.root.glob("*.txt"))
        if files:
            return files
        raise FileNotFoundError(f"No ToxiGen text files found under {self.root}")

    def _load_file(self, path: Path) -> list[PromptSample]:
        samples: list[PromptSample] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                normalized = self._normalize_line(line)
                if not normalized:
                    continue
                stem = path.stem.replace(" ", "_")
                samples.append(
                    PromptSample(
                        prompt_id=f"toxigen-{stem}-{line_number}",
                        text=normalized,
                        metadata={
                            "dataset": "toxigen",
                            "subset": stem,
                            "source_file": str(path),
                            "line_number": line_number,
                            "known_biased": True,
                            "text_is_response": True,
                        },
                    )
                )
        return samples

    def _normalize_line(self, line: str) -> str:
        raw = line.strip()
        if not raw:
            return ""
        raw = raw.replace("\\n", "\n")
        segments: list[str] = []
        for piece in raw.splitlines():
            cleaned = piece.strip()
            if cleaned.startswith("-"):
                cleaned = cleaned[1:].strip()
            if cleaned:
                segments.append(cleaned)
        return " ".join(segments)
