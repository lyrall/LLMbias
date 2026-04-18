from __future__ import annotations

from llmbias.schemas import CounterfactualSample, PromptSample, SensitiveAttribute


class CounterfactualGenerator:
    """Generate minimally edited prompt variants by swapping sensitive attributes."""

    _SWAPS = {
        "gender": {
            "她": "他",
            "他": "她",
            "女性": "男性",
            "男性": "女性",
            "woman": "man",
            "man": "woman",
            "female": "male",
            "male": "female",
            "actress": "actor",
            "actor": "actress",
            "comedienne": "comedian",
        },
        "religion": {
            "穆斯林": "基督徒",
            "基督徒": "穆斯林",
            "佛教徒": "基督徒",
            "muslim": "christian",
            "christian": "muslim",
            "buddhist": "christian",
        },
        "age": {
            "老年": "年轻",
            "年轻": "老年",
            "中年": "年轻",
            "老人": "青年",
            "青年": "老人",
            "elderly": "young",
            "old": "young",
            "older": "younger",
            "young": "elderly",
            "younger": "older",
            "middle-aged": "young",
            "grandfather": "grandson",
            "grandmother": "granddaughter",
            "grandpa": "grandson",
            "grandma": "granddaughter",
            "grandparent": "grandchild",
            "grandson": "grandfather",
            "granddaughter": "grandmother",
            "grandchild": "grandparent",
        },
        "race": {
            "黑人": "白人",
            "白人": "黑人",
            "亚裔": "白人",
            "black": "white",
            "white": "black",
            "asian": "white",
        },
    }

    def generate(
        self, sample: PromptSample, attributes: list[SensitiveAttribute]
    ) -> list[CounterfactualSample]:
        counterfactuals: list[CounterfactualSample] = []
        for attribute in attributes:
            swapped = self._SWAPS.get(attribute.category, {}).get(attribute.value)
            if not swapped:
                continue
            cf_text = sample.text.replace(attribute.value, swapped, 1)
            if cf_text == sample.text:
                cf_text = sample.text.replace(attribute.value.lower(), swapped, 1)
            if cf_text == sample.text:
                continue
            counterfactuals.append(
                CounterfactualSample(
                    source_prompt_id=sample.prompt_id,
                    original_text=sample.text,
                    counterfactual_text=cf_text,
                    swapped_attribute=SensitiveAttribute(
                        category=attribute.category,
                        value=swapped,
                        confidence=attribute.confidence,
                        source="counterfactual_swap",
                    ),
                    original_attribute=attribute,
                    generation_strategy="single_attribute_swap",
                )
            )
        return counterfactuals
