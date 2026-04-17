from __future__ import annotations

from llmbias.config import CorrectionConfig
from llmbias.correction.corrector import BiasCorrector
from llmbias.schemas import BiasDetectionResult, RewriteResult


class CorrectionPipeline:
    """Independent pipeline for research content two."""

    def __init__(self, config: CorrectionConfig) -> None:
        self.corrector = BiasCorrector(config=config)

    def run(self, detection: BiasDetectionResult) -> RewriteResult | None:
        return self.corrector.run(detection)
