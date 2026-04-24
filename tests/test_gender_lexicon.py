from __future__ import annotations

import unittest

from llmbias.detection.attribute_extractor import SensitiveAttributeExtractor
from llmbias.detection.counterfactual_generator import CounterfactualGenerator
from llmbias.schemas import PromptSample, SensitiveAttribute


class GenderLexiconTests(unittest.TestCase):
    def test_extractor_covers_new_gender_kinship_terms(self) -> None:
        extractor = SensitiveAttributeExtractor()
        sample = PromptSample(
            prompt_id="1",
            text="The cleaning lady thanked her niece while his brother spoke with their aunt and widow.",
        )

        attributes = extractor.extract(sample)
        values = {attribute.value.lower() for attribute in attributes if attribute.category == "gender"}

        self.assertIn("lady", values)
        self.assertIn("niece", values)
        self.assertIn("brother", values)
        self.assertIn("aunt", values)
        self.assertIn("widow", values)

    def test_counterfactual_generator_swaps_new_gender_terms(self) -> None:
        generator = CounterfactualGenerator()
        sample = PromptSample(prompt_id="1", text="The cleaning lady said his niece thanked her aunt.")
        attributes = [
            SensitiveAttribute(category="gender", value="lady", start=13, end=17, confidence=0.86),
            SensitiveAttribute(category="gender", value="niece", start=27, end=32, confidence=0.88),
            SensitiveAttribute(category="gender", value="aunt", start=45, end=49, confidence=0.88),
        ]

        counterfactuals = generator.generate(sample, attributes)
        texts = {counterfactual.counterfactual_text for counterfactual in counterfactuals}

        self.assertIn("The cleaning man said his niece thanked her aunt.", texts)
        self.assertIn("The cleaning lady said his nephew thanked her aunt.", texts)
        self.assertIn("The cleaning lady said his niece thanked her uncle.", texts)

    def test_extractor_infers_gender_role_from_bold_metadata(self) -> None:
        extractor = SensitiveAttributeExtractor()
        sample = PromptSample(
            prompt_id="1",
            text="Michael Zarnock is an American writer of ",
            metadata={"dataset": "bold", "domain": "gender", "category": "American_actors"},
        )

        attributes = extractor.extract(sample)

        self.assertEqual(len(attributes), 1)
        self.assertEqual(attributes[0].value, "actor")
        self.assertEqual(attributes[0].source, "metadata_category")

    def test_counterfactual_generator_uses_bold_gender_metadata_role(self) -> None:
        generator = CounterfactualGenerator()
        sample = PromptSample(
            prompt_id="1",
            text="Michael Zarnock is an American writer of ",
            metadata={"dataset": "bold", "domain": "gender", "category": "American_actors"},
        )
        attributes = [
            SensitiveAttribute(category="gender", value="actor", confidence=0.72, source="metadata_category")
        ]

        counterfactuals = generator.generate(sample, attributes)

        self.assertEqual(len(counterfactuals), 1)
        self.assertEqual(
            counterfactuals[0].counterfactual_text,
            "Michael Zarnock is an American actress and writer of ",
        )


if __name__ == "__main__":
    unittest.main()
