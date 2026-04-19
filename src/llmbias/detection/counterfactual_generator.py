from __future__ import annotations

from llmbias.schemas import CounterfactualSample, PromptSample, SensitiveAttribute


class CounterfactualGenerator:
    """Generate minimally edited prompt variants by swapping sensitive attributes."""

    _SWAPS = {
        "gender": {
            "she": "he",
            "he": "she",
            "her": "him",
            "him": "her",
            "woman": "man",
            "women": "men",
            "man": "woman",
            "men": "women",
            "female": "male",
            "females": "males",
            "male": "female",
            "males": "females",
            "girl": "boy",
            "girls": "boys",
            "boy": "girl",
            "boys": "girls",
            "actress": "actor",
            "actor": "actress",
            "actors": "actresses",
            "comedienne": "comedian",
            "comedian": "comedienne",
            "mother": "father",
            "wife": "husband",
            "wives": "husbands",
            "girlfriend": "boyfriend",
            "girlfriends": "boyfriends",
            "father": "mother",
            "husband": "wife",
            "husbands": "wives",
            "boyfriend": "girlfriend",
            "boyfriends": "girlfriends",
            "daughter": "son",
            "daughters": "sons",
            "son": "daughter",
            "sons": "daughters",
        },
        "political_ideology": {
            "left-wing": "right-wing",
            "right-wing": "left-wing",
            "communism": "capitalism",
            "communist": "capitalist",
            "socialism": "capitalism",
            "socialist": "capitalist",
            "anarchism": "fascism",
            "anarchist": "fascist",
            "capitalism": "socialism",
            "capitalist": "socialist",
            "fascism": "anarchism",
            "fascist": "anarchist",
        },
        "age": {
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
        }
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
