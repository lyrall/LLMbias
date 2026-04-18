from __future__ import annotations

from llmbias.models.base import BlackBoxLLM
from llmbias.schemas import ModelResponse


class MockLLM(BlackBoxLLM):
    """Scenario-driven mock model for exercising the framework end to end."""

    def generate_from_messages(self, messages: list[dict[str, str]]) -> ModelResponse:
        prompt = messages[-1]["content"] if messages else ""
        lowered = prompt.lower()
        if "answer only with yes or no" in lowered:
            yes_terms = ["better suited", "less capable", "naturally", "should avoid", "more likely"]
            text = "Yes" if any(term in lowered for term in yes_terms) else "No"
            return ModelResponse(text=text, metadata={"provider": "mock", "model": "heuristic-demo"})
        if "single integer score between 0 and 10" in lowered:
            skeptical_terms = ["naturally", "inherent", "always", "more likely", "stereotype"]
            text = "3" if any(term in lowered for term in skeptical_terms) else "8"
            return ModelResponse(text=text, metadata={"provider": "mock", "model": "heuristic-demo"})
        if "woman" in lowered or "female" in lowered or "她" in prompt or "女性" in prompt:
            text = "她可能更适合沟通协调类岗位，不太适合承担高压架构设计工作。"
        elif "man" in lowered or "male" in lowered or "他" in prompt or "男性" in prompt:
            text = "他通常会被认为更适合高强度技术岗位，也更适合承担架构设计职责。"
        elif "elderly" in lowered or "老年" in prompt:
            text = "老年求职者往往被默认学习速度较慢，因此不太适合快速迭代岗位。"
        else:
            text = "应基于个人能力、经历和岗位要求进行评估，而不是依据敏感属性下结论。"
        return ModelResponse(text=text, metadata={"provider": "mock", "model": "heuristic-demo"})
