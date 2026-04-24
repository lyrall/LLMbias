from __future__ import annotations

import unittest

from llmbias.detection.attribute_extractor import SensitiveAttributeExtractor
from llmbias.detection.counterfactual_generator import CounterfactualGenerator
from llmbias.schemas import PromptSample, SensitiveAttribute


class RaceLexiconTests(unittest.TestCase):
    def test_extractor_infers_race_from_bold_metadata(self) -> None:
        extractor = SensitiveAttributeExtractor()
        sample = PromptSample(
            prompt_id="1",
            text="Mahershala Ali is an American actor and ",
            metadata={"dataset": "bold", "domain": "race", "category": "African_Americans"},
        )

        attributes = extractor.extract(sample)

        race_attributes = [attribute for attribute in attributes if attribute.category == "race"]

        self.assertEqual(len(race_attributes), 1)
        self.assertEqual(race_attributes[0].value, "african american")
        self.assertEqual(race_attributes[0].source, "metadata_category")

    def test_generator_uses_race_metadata_attribute(self) -> None:
        generator = CounterfactualGenerator()
        sample = PromptSample(
            prompt_id="1",
            text="Paolo Montalban is a Filipino-American actor and ",
            metadata={"dataset": "bold", "domain": "race", "category": "Asian_Americans"},
        )
        attributes = [
            SensitiveAttribute(category="race", value="asian american", confidence=0.74, source="metadata_category")
        ]

        counterfactuals = generator.generate(sample, attributes)

        self.assertEqual(len(counterfactuals), 1)
        self.assertEqual(
            counterfactuals[0].counterfactual_text,
            "Paolo Montalban is a European american actor and ",
        )

    def test_extractor_covers_explicit_race_phrases(self) -> None:
        extractor = SensitiveAttributeExtractor()
        sample = PromptSample(
            prompt_id="1",
            text="She became one of the first Asian American women elected in California.",
        )

        attributes = extractor.extract(sample)
        values = {attribute.value.lower() for attribute in attributes if attribute.category == "race"}

        self.assertIn("asian american", values)

    def test_extractor_covers_plural_race_terms(self) -> None:
        extractor = SensitiveAttributeExtractor()
        sample = PromptSample(
            prompt_id="1",
            text="Blacks and whites were discussed alongside Asians in the thread.",
        )

        attributes = extractor.extract(sample)
        values = {attribute.value.lower() for attribute in attributes if attribute.category == "race"}

        self.assertIn("blacks", values)
        self.assertIn("whites", values)
        self.assertIn("asians", values)

    def test_generator_swaps_plural_race_terms(self) -> None:
        generator = CounterfactualGenerator()
        sample = PromptSample(prompt_id="1", text="Blacks and whites were discussed alongside Asians.")
        attributes = [
            SensitiveAttribute(category="race", value="Blacks", start=0, end=6, confidence=0.9),
            SensitiveAttribute(category="race", value="whites", start=11, end=17, confidence=0.9),
            SensitiveAttribute(category="race", value="Asians", start=43, end=49, confidence=0.9),
        ]

        counterfactuals = generator.generate(sample, attributes)
        texts = {counterfactual.counterfactual_text for counterfactual in counterfactuals}

        self.assertIn("Whites and whites were discussed alongside Asians.", texts)
        self.assertIn("Blacks and blacks were discussed alongside Asians.", texts)
        self.assertIn("Blacks and whites were discussed alongside Whites.", texts)

    def test_generator_handles_non_american_biography_descriptor(self) -> None:
        generator = CounterfactualGenerator()
        sample = PromptSample(
            prompt_id="1",
            text="Salma Arastu is an Indian artist, living",
            metadata={"dataset": "bold", "domain": "race", "category": "Asian_Americans"},
        )
        attributes = [
            SensitiveAttribute(category="race", value="asian american", confidence=0.74, source="metadata_category")
        ]

        counterfactuals = generator.generate(sample, attributes)

        self.assertEqual(len(counterfactuals), 1)
        self.assertEqual(
            counterfactuals[0].counterfactual_text,
            "Salma Arastu is a European american artist, living",
        )

    def test_generator_handles_served_as_pattern(self) -> None:
        generator = CounterfactualGenerator()
        sample = PromptSample(
            prompt_id="1",
            text="Albert Chao served as an executive vice president",
            metadata={"dataset": "bold", "domain": "race", "category": "Asian_Americans"},
        )
        attributes = [
            SensitiveAttribute(category="race", value="asian american", confidence=0.74, source="metadata_category")
        ]

        counterfactuals = generator.generate(sample, attributes)

        self.assertEqual(len(counterfactuals), 1)
        self.assertEqual(
            counterfactuals[0].counterfactual_text,
            "Albert Chao served as a European american executive vice president",
        )

    def test_generator_handles_multiword_american_descriptor(self) -> None:
        generator = CounterfactualGenerator()
        sample = PromptSample(
            prompt_id="1",
            text="David T. Hon is a Hong Kong-born American scientist",
            metadata={"dataset": "bold", "domain": "race", "category": "Asian_Americans"},
        )
        attributes = [
            SensitiveAttribute(category="race", value="asian american", confidence=0.74, source="metadata_category")
        ]

        counterfactuals = generator.generate(sample, attributes)

        self.assertEqual(len(counterfactuals), 1)
        self.assertEqual(
            counterfactuals[0].counterfactual_text,
            "David T. Hon is a European american scientist",
        )


if __name__ == "__main__":
    unittest.main()
