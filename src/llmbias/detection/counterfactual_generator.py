from __future__ import annotations

import re

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
            "actresses": "actors",
            "actor": "actress",
            "actors": "actresses",
            "comedienne": "comedian",
            "comedian": "comedienne",
            "sister": "brother",
            "sisters": "brothers",
            "brother": "sister",
            "brothers": "sisters",
            "mother": "father",
            "mothers": "fathers",
            "wife": "husband",
            "wives": "husbands",
            "girlfriend": "boyfriend",
            "girlfriends": "boyfriends",
            "father": "mother",
            "fathers": "mothers",
            "husband": "wife",
            "husbands": "wives",
            "boyfriend": "girlfriend",
            "boyfriends": "girlfriends",
            "mom": "dad",
            "moms": "dads",
            "dad": "mom",
            "dads": "moms",
            "daughter": "son",
            "daughters": "sons",
            "son": "daughter",
            "sons": "daughters",
            "widow": "widower",
            "widows": "widowers",
            "widower": "widow",
            "widowers": "widows",
            "bride": "groom",
            "brides": "grooms",
            "groom": "bride",
            "grooms": "brides",
        },
        "political_ideology": {
            "democracy": "authoritarianism",
            "social democracy": "authoritarian conservatism",
            "democratic socialism": "authoritarian capitalism",
            "liberalism": "conservatism",
            "liberal": "conservative",
            "social liberalism": "social conservatism",
            "conservatism": "liberalism",
            "conservative": "liberal",
            "far-left": "far-right",
            "far-right": "far-left",
            "centre-left": "centre-right",
            "center-left": "center-right",
            "centre-right": "centre-left",
            "center-right": "center-left",
            "jewish left": "jewish right",
            "jewish right": "jewish left",
            "european left": "european right",
            "european right": "european left",
            "left bloc": "right bloc",
            "right bloc": "left bloc",
            "historical left": "historical right",
            "historical right": "historical left",
            "left shachtmanism": "right shachtmanism",
            "right shachtmanism": "left shachtmanism",
            "spiritual left": "spiritual right",
            "spiritual right": "spiritual left",
            "spiritual leftism": "spiritual rightism",
            "spiritual rightism": "spiritual leftism",
            "left realism": "right realism",
            "right realism": "left realism",
            "left-wing": "right-wing",
            "right-wing": "left-wing",
            "leftist": "rightist",
            "rightist": "leftist",
            "new left": "new right",
            "new right": "new left",
            "old left": "old right",
            "old right": "old left",
            "christian left": "christian right",
            "christian right": "christian left",
            "religious right": "religious left",
            "religious left": "religious right",
            "evangelical left": "evangelical right",
            "evangelical right": "evangelical left",
            "anti-stalinist left": "anti-stalinist right",
            "anti-stalinist right": "anti-stalinist left",
            "left opposition": "right opposition",
            "right opposition": "left opposition",
            "regressive left": "regressive right",
            "regressive right": "regressive left",
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
        "religion": {
            "muslim": "christian",
            "islamic": "christian",
            "christian": "muslim",
            "christianity": "islam",
            "buddhist": "christian",
            "buddhism": "christianity",
            "ç©†æ–¯æž—": "åŸºç£å¾’",
            "åŸºç£å¾’": "ç©†æ–¯æž—",
            "ä½›æ•™å¾’": "åŸºç£å¾’",
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
        },
    }

    def generate(
        self, sample: PromptSample, attributes: list[SensitiveAttribute]
    ) -> list[CounterfactualSample]:
        counterfactuals: list[CounterfactualSample] = []
        for attribute in attributes:
            swapped = self._lookup_swap(attribute)
            if swapped is None:
                continue
            if attribute.source == "metadata_category":
                cf_text = self._replace_attribute_from_metadata(sample, attribute, swapped)
            else:
                cf_text = self._replace_attribute(sample.text, attribute, swapped)
            if cf_text is None or cf_text == sample.text:
                continue
            counterfactuals.append(
                CounterfactualSample(
                    source_prompt_id=sample.prompt_id,
                    original_text=sample.text,
                    counterfactual_text=cf_text,
                    swapped_attribute=SensitiveAttribute(
                        category=attribute.category,
                        value=self._match_surface_form(attribute.value, swapped),
                        confidence=attribute.confidence,
                        source="counterfactual_swap",
                    ),
                    original_attribute=attribute,
                    generation_strategy="single_attribute_swap",
                )
            )
        return counterfactuals

    def _lookup_swap(self, attribute: SensitiveAttribute) -> str | None:
        swaps = self._SWAPS.get(attribute.category, {})
        if not swaps:
            return None
        return swaps.get(attribute.value) or swaps.get(attribute.value.lower())

    def _replace_attribute(
        self,
        text: str,
        attribute: SensitiveAttribute,
        swapped: str,
    ) -> str | None:
        if attribute.start is not None and attribute.end is not None:
            original = text[attribute.start:attribute.end]
            if original and original.lower() == attribute.value.lower():
                replacement = self._match_surface_form(original, swapped)
                return text[:attribute.start] + replacement + text[attribute.end:]

        pattern = re.compile(re.escape(attribute.value), flags=re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            return None
        original = match.group(0)
        replacement = self._match_surface_form(original, swapped)
        return text[: match.start()] + replacement + text[match.end() :]

    def _match_surface_form(self, original: str, swapped: str) -> str:
        if original.isupper():
            return swapped.upper()
        if original[:1].isupper():
            return swapped[:1].upper() + swapped[1:]
        return swapped

    def _replace_attribute_from_metadata(
        self,
        sample: PromptSample,
        attribute: SensitiveAttribute,
        swapped: str,
    ) -> str | None:
        text = sample.text
        swapped_surface = self._match_surface_form(attribute.value, swapped)
        explicit = self._replace_explicit_role_term(text, attribute.value, swapped_surface)
        if explicit is not None and explicit != text:
            return explicit

        patterns: list[tuple[re.Pattern[str], str]] = [
            (
                re.compile(r"\b(is|was)\s+an\s+American\b", flags=re.IGNORECASE),
                r"\1 an American {role} and",
            ),
            (
                re.compile(r"\b(is|was)\s+an\b", flags=re.IGNORECASE),
                r"\1 an {role} and",
            ),
            (
                re.compile(r"\b(is|was)\s+a\b", flags=re.IGNORECASE),
                rf"\1 {self._indefinite_article(swapped_surface)} {swapped_surface} and",
            ),
            (
                re.compile(r"\b(is|was)\b", flags=re.IGNORECASE),
                rf"\1 {self._indefinite_article(swapped_surface)} {swapped_surface} and",
            ),
        ]

        for pattern, replacement in patterns:
            if pattern.search(text):
                return pattern.sub(replacement.format(role=swapped_surface), text, count=1)
        return None

    def _replace_explicit_role_term(self, text: str, original_value: str, swapped_value: str) -> str | None:
        pattern = re.compile(rf"\b{re.escape(original_value)}\b", flags=re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            return None
        replacement = self._match_surface_form(match.group(0), swapped_value)
        return text[: match.start()] + replacement + text[match.end() :]

    def _indefinite_article(self, value: str) -> str:
        return "an" if value[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
