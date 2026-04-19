from __future__ import annotations

import re

from llmbias.schemas import PromptSample, SensitiveAttribute


class SensitiveAttributeExtractor:
    """Rule-first attribute extractor for counterfactual prompt construction."""

    _LEXICON = {
        "gender": {
            "she": 0.98,
            "he": 0.98,
            "her": 0.9,
            "him": 0.9,
            "她": 0.98,
            "他": 0.98,
            "女性": 0.95,
            "男性": 0.95,
            "woman": 0.95,
            "women": 0.95,
            "man": 0.95,
            "men": 0.95,
            "female": 0.9,
            "females": 0.9,
            "male": 0.9,
            "males": 0.9,
            "girl": 0.9,
            "girls": 0.9,
            "boy": 0.9,
            "boys": 0.9,
            "actress": 0.92,
            "actor": 0.92,
            "actors": 0.9,
            "comedienne": 0.9,
            "comedian": 0.9,
            "mother": 0.88,
            "wife": 0.88,
            "wives": 0.88,
            "girlfriend": 0.88,
            "girlfriends": 0.88,
            "father": 0.88,
            "husband": 0.88,
            "husbands": 0.88,
            "boyfriend": 0.88,
            "boyfriends": 0.88,
            "daughter": 0.88,
            "daughters": 0.88,
            "son": 0.88,
            "sons": 0.88,
        },
        "political_ideology": {
            "far-left": 0.97,
            "far-right": 0.97,
            "centre-left": 0.96,
            "center-left": 0.96,
            "centre-right": 0.96,
            "center-right": 0.96,
            "left-wing": 0.95,
            "right-wing": 0.95,
            "leftist": 0.94,
            "rightist": 0.94,
            "new left": 0.94,
            "new right": 0.94,
            "old left": 0.92,
            "old right": 0.92,
            "christian left": 0.95,
            "christian right": 0.95,
            "religious right": 0.93,
            "evangelical left": 0.94,
            "evangelical right": 0.94,
            "anti-stalinist left": 0.95,
            "anti-stalinist right": 0.95,
            "left opposition": 0.95,
            "right opposition": 0.95,
            "regressive left": 0.93,
            "regressive right": 0.93,
            "communism": 0.95,
            "communist": 0.95,
            "socialism": 0.95,
            "socialist": 0.95,
            "anarchism": 0.92,
            "anarchist": 0.92,
            "capitalism": 0.95,
            "capitalist": 0.95,
            "fascism": 0.98,
            "fascist": 0.98,
        },
        "religion": {
            "穆斯林": 0.95,
            "基督徒": 0.95,
            "佛教徒": 0.92,
            "muslim": 0.95,
            "christian": 0.95,
            "buddhist": 0.92,
        },
        "age": {
            "老年": 0.92,
            "年轻": 0.9,
            "中年": 0.85,
            "老人": 0.92,
            "青年": 0.88,
            "elderly": 0.92,
            "old": 0.9,
            "older": 0.9,
            "young": 0.9,
            "younger": 0.9,
            "middle-aged": 0.85,
            "grandfather": 0.95,
            "grandmother": 0.95,
            "grandpa": 0.95,
            "grandma": 0.95,
            "grandparent": 0.93,
            "grandson": 0.93,
            "granddaughter": 0.93,
        },
        "race": {
            "黑人": 0.94,
            "白人": 0.94,
            "亚裔": 0.94,
            "black": 0.9,
            "white": 0.9,
            "asian": 0.9,
        },
    }

    def extract(self, sample: PromptSample) -> list[SensitiveAttribute]:
        matches: list[SensitiveAttribute] = []
        seen: set[tuple[str, int, str]] = set()
        for category, values in self._LEXICON.items():
            for value, confidence in values.items():
                for start, end in self._find_mentions(sample.text, value):
                    key = (category, start, value.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    matches.append(
                        SensitiveAttribute(
                            category=category,
                            value=sample.text[start:end],
                            start=start,
                            end=end,
                            confidence=confidence,
                            source="rule_lexicon",
                        )
                    )
        return sorted(matches, key=lambda item: (item.start is None, item.start or -1))

    def _find_mentions(self, text: str, value: str) -> list[tuple[int, int]]:
        if value.isascii() and any(char.isalpha() for char in value):
            pattern = re.compile(rf"\b{re.escape(value)}\b", flags=re.IGNORECASE)
            return [(match.start(), match.end()) for match in pattern.finditer(text)]

        matches: list[tuple[int, int]] = []
        start = text.find(value)
        while start >= 0:
            matches.append((start, start + len(value)))
            start = text.find(value, start + len(value))
        return matches
