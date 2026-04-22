from __future__ import annotations

import unittest

from llmbias.detection.attribute_extractor import SensitiveAttributeExtractor
from llmbias.detection.counterfactual_generator import CounterfactualGenerator
from llmbias.schemas import PromptSample


class PoliticalCounterfactualTests(unittest.TestCase):
    def test_extractor_covers_democracy_and_liberalism_terms(self) -> None:
        extractor = SensitiveAttributeExtractor()
        sample = PromptSample(
            prompt_id="1",
            text="Social democracy and liberalism both shaped modern politics.",
        )

        attributes = extractor.extract(sample)
        values = {attribute.value.lower() for attribute in attributes if attribute.category == "political_ideology"}

        self.assertIn("social democracy", values)
        self.assertIn("liberalism", values)

    def test_extractor_covers_left_right_variants(self) -> None:
        extractor = SensitiveAttributeExtractor()
        sample = PromptSample(
            prompt_id="1",
            text="The Left Bloc and Jewish left were discussed alongside Right Realism.",
        )

        attributes = extractor.extract(sample)
        values = {attribute.value.lower() for attribute in attributes if attribute.category == "political_ideology"}

        self.assertIn("left bloc", values)
        self.assertIn("jewish left", values)
        self.assertIn("right realism", values)

    def test_generator_builds_counterfactual_for_christian_democracy(self) -> None:
        extractor = SensitiveAttributeExtractor()
        generator = CounterfactualGenerator()
        sample = PromptSample(
            prompt_id="1",
            text="Christian democracy is a political ideology that values social welfare.",
        )

        attributes = extractor.extract(sample)
        counterfactuals = generator.generate(sample, attributes)
        texts = {counterfactual.counterfactual_text for counterfactual in counterfactuals}

        self.assertIn(
            "Muslim democracy is a political ideology that values social welfare.",
            texts,
        )


if __name__ == "__main__":
    unittest.main()
