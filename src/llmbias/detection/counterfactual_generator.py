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
            "lady": "man",
            "ladies": "men",
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
            "aunt": "uncle",
            "aunts": "uncles",
            "uncle": "aunt",
            "uncles": "aunts",
            "niece": "nephew",
            "nieces": "nephews",
            "nephew": "niece",
            "nephews": "nieces",
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
        "race": {
            "asian": "white",
            "black": "white",
            "white": "black",
            "asian american": "european american",
            "african american": "european american",
            "european american": "african american",
            "hispanic american": "european american",
            "latino american": "european american",
            "latina american": "european american",
            "latino": "white",
            "latina": "white",
            "hispanic": "white",
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
        if attribute.category == "race":
            return self._replace_race_from_metadata(sample.text, attribute, swapped)

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

    def _replace_race_from_metadata(
        self,
        text: str,
        attribute: SensitiveAttribute,
        swapped: str,
    ) -> str | None:
        swapped_surface = self._match_surface_form(attribute.value, swapped)
        swapped_surface = swapped_surface[:1].upper() + swapped_surface[1:]

        explicit_patterns: list[tuple[re.Pattern[str], str]] = [
            (
                re.compile(r"\b([A-Za-z][A-Za-z-]*)-American\b", flags=re.IGNORECASE),
                swapped_surface,
            ),
            (
                re.compile(
                    r"\b(an?)\s+([A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,2})\s+American\b",
                    flags=re.IGNORECASE,
                ),
                "{article} {race}",
            ),
            (
                re.compile(r"\b(Asian|African|European|Hispanic|Latino|Latina)-American\b", flags=re.IGNORECASE),
                swapped_surface,
            ),
            (
                re.compile(r"\b(Asian|African|European|Hispanic|Latino|Latina)\s+American\b", flags=re.IGNORECASE),
                swapped_surface,
            ),
        ]
        for pattern, replacement in explicit_patterns:
            match = pattern.search(text)
            if match:
                replacement_text = replacement
                if "{article}" in replacement:
                    replacement_text = replacement.format(
                        article=self._match_surface_form(match.group(1), self._indefinite_article(swapped_surface)),
                        race=swapped_surface,
                    )
                else:
                    replacement_text = self._match_surface_form(match.group(0), replacement)
                return text[: match.start()] + replacement_text + text[match.end() :]

        swapped_surface = swapped_surface.replace("-", " ")
        patterns: list[tuple[re.Pattern[str], str]] = [
            (
                re.compile(r"\b(is|was)\s+an\s+American\b", flags=re.IGNORECASE),
                r"\1 an {race}",
            ),
            (
                re.compile(r"\b(is|was)\s+a\s+American\b", flags=re.IGNORECASE),
                r"\1 a {race}",
            ),
            (
                re.compile(r"\b(is|was)\s+American\b", flags=re.IGNORECASE),
                r"\1 {race}",
            ),
        ]
        for pattern, replacement in patterns:
            if pattern.search(text):
                return pattern.sub(replacement.format(race=swapped_surface), text, count=1)

        profession_patterns = self._race_profession_patterns(swapped_surface)
        for pattern, replacement in profession_patterns:
            if pattern.search(text):
                return pattern.sub(replacement, text, count=1)

        return None

    def _indefinite_article(self, value: str) -> str:
        lowered = value.strip().lower()
        if lowered.startswith(("eu", "uni", "use", "u.s")):
            return "a"
        return "an" if lowered[:1] in {"a", "e", "i", "o", "u"} else "a"

    def _race_profession_patterns(self, swapped_surface: str) -> list[tuple[re.Pattern[str], str]]:
        article = self._indefinite_article(swapped_surface)
        role_keywords = (
            "actor|actress|artist|writer|poet|novelist|author|designer|engineer|"
            "entrepreneur|executive|businessman|businesswoman|businessperson|"
            "mathematician|scientist|professor|economist|photographer|director|"
            "producer|architect|journalist|politician|lawyer|physician|doctor|"
            "singer|songwriter|musician|composer|dancer|athlete|player|coach|"
            "researcher|educator|scholar|historian|philanthropist|activist"
        )
        return [
            (
                re.compile(
                    rf"\b(is|was)\s+(?:an|a)\s+[A-Za-z][A-Za-z-]+\s+({role_keywords})\b",
                    flags=re.IGNORECASE,
                ),
                rf"\1 {article} {swapped_surface} \2",
            ),
            (
                re.compile(
                    rf"\b(is|was)\s+the\s+({role_keywords})\b",
                    flags=re.IGNORECASE,
                ),
                rf"\1 the {swapped_surface} \2",
            ),
            (
                re.compile(
                    rf"\b(serves|served)\s+as\s+(?:an|a)\s+({role_keywords})\b",
                    flags=re.IGNORECASE,
                ),
                rf"\1 as {article} {swapped_surface} \2",
            ),
            (
                re.compile(
                    rf"\b(serves|served)\s+as\s+the\s+({role_keywords})\b",
                    flags=re.IGNORECASE,
                ),
                rf"\1 as the {swapped_surface} \2",
            ),
        ]
