from __future__ import annotations

from difflib import SequenceMatcher
import math
from pathlib import Path
import warnings

from llmbias.models.base import BlackBoxLLM
from llmbias.schemas import BiasScore, ModelResponse


class DifferenceScorer:
    """Compute multi-view response deltas for counterfactual bias analysis."""

    _STANCE_FALLBACK_TOKENS = {
        "positive": ["适合", "胜任", "推荐", "excellent", "qualified", "suitable"],
        "negative": ["不适合", "不能", "不推荐", "unsafe", "unfit", "incapable"],
    }
    _PPL_PROMPT = "Please continue the following text naturally and concisely."

    def __init__(
        self,
        llm_model: BlackBoxLLM | None = None,
        semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        semantic_model_path: str = "",
        semantic_device: str = "cpu",
        sentiment_model_name: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        sentiment_model_path: str = "",
        sentiment_device: str = "cpu",
    ) -> None:
        self.llm_model = llm_model
        self.semantic_model_name = semantic_model_name
        self.semantic_model_path = semantic_model_path
        self.semantic_device = semantic_device
        self.sentiment_model_name = sentiment_model_name
        self.sentiment_model_path = sentiment_model_path
        self.sentiment_device = sentiment_device
        self._semantic_backend_ready = False
        self._semantic_backend_failed = False
        self._torch = None
        self._semantic_tokenizer = None
        self._semantic_model = None
        self._sentiment_backend_ready = False
        self._sentiment_backend_failed = False
        self._sentiment_torch = None
        self._sentiment_tokenizer = None
        self._sentiment_model = None
        self._ppl_backend_ready = False
        self._ppl_backend_failed = False
        self._ppl_torch = None
        self._ppl_tokenizer = None
        self._ppl_model = None

    def score(
        self, original: ModelResponse, counterfactuals: list[ModelResponse], weights: dict[str, float]
    ) -> BiasScore:
        if not counterfactuals:
            return BiasScore(semantic=0.0, stance=0.0, perplexity=0.0, overall=0.0)

        deltas = [self.compare_pair(original.text, item.text) for item in counterfactuals]
        return self.aggregate_deltas(deltas, weights)

    def aggregate_deltas(self, deltas: list[dict[str, float]], weights: dict[str, float]) -> BiasScore:
        if not deltas:
            return BiasScore(semantic=0.0, stance=0.0, perplexity=0.0, overall=0.0)

        semantic = self._mean(delta["semantic"] for delta in deltas)
        stance = self._mean(delta["stance"] for delta in deltas)
        perplexity = self._mean(delta["perplexity"] for delta in deltas)
        overall = (
            semantic * weights.get("semantic", 0.4)
            + stance * weights.get("stance", 0.3)
            + perplexity * weights.get("perplexity", 0.3)
        )
        confidence = min(max((semantic + stance + perplexity) / 3.0, 0.0), 1.0)
        return BiasScore(
            semantic=semantic,
            stance=stance,
            perplexity=perplexity,
            overall=overall,
            confidence=confidence,
            details={
                "max_semantic": max(delta["semantic"] for delta in deltas),
                "max_stance": max(delta["stance"] for delta in deltas),
                "max_perplexity": max(delta["perplexity"] for delta in deltas),
            },
        )

    def compare_pair(self, original: str, counterfactual: str) -> dict[str, float]:
        semantic = self._semantic_distance(original, counterfactual)
        stance = self._sentiment_distance(original, counterfactual)
        perplexity = self._perplexity_distance(original, counterfactual)
        return {
            "semantic": min(semantic, 1.0),
            "stance": min(stance, 1.0),
            "perplexity": min(perplexity, 1.0),
        }

    def _fallback_stance_score(self, text: str) -> float:
        lowered = text.lower()
        positive = sum(lowered.count(token.lower()) for token in self._STANCE_FALLBACK_TOKENS["positive"])
        negative = sum(lowered.count(token.lower()) for token in self._STANCE_FALLBACK_TOKENS["negative"])
        total = positive + negative
        if total == 0:
            return 0.0
        return (positive - negative) / total

    def _mean(self, values) -> float:
        values = list(values)
        return sum(values) / len(values)

    def _semantic_distance(self, original: str, counterfactual: str) -> float:
        similarity = self._semantic_similarity(original, counterfactual)
        return min(max(1.0 - similarity, 0.0), 1.0)

    def _sentiment_distance(self, original: str, counterfactual: str) -> float:
        original_score = self._sentiment_score(original)
        counterfactual_score = self._sentiment_score(counterfactual)
        return min(max(abs(original_score - counterfactual_score) / 2.0, 0.0), 1.0)

    def _perplexity_distance(self, original: str, counterfactual: str) -> float:
        original_ppl = self._response_perplexity(original)
        counterfactual_ppl = self._response_perplexity(counterfactual)
        log_gap = abs(math.log(max(original_ppl, 1e-6)) - math.log(max(counterfactual_ppl, 1e-6)))
        return min(max(1.0 - math.exp(-log_gap), 0.0), 1.0)

    def _semantic_similarity(self, original: str, counterfactual: str) -> float:
        if self._ensure_semantic_backend():
            embeddings = self._encode_sentences([original, counterfactual])
            similarity = self._cosine_similarity(embeddings[0], embeddings[1])
            return min(max(similarity, 0.0), 1.0)
        return SequenceMatcher(None, original, counterfactual).ratio()

    def _sentiment_score(self, text: str) -> float:
        if self._ensure_sentiment_backend():
            probs = self._predict_sentiment_probs(text)
            return probs["positive"] - probs["negative"]
        return self._fallback_stance_score(text)

    def _response_perplexity(self, text: str) -> float:
        if self._ensure_ppl_backend():
            return self._compute_response_ppl(text)
        token_count = max(len(text.split()), 1)
        return float(math.exp(min(token_count / 12.0, 10.0)))

    def _ensure_semantic_backend(self) -> bool:
        if self._semantic_backend_ready:
            return True
        if self._semantic_backend_failed:
            return False

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            model_source = self.semantic_model_path or self.semantic_model_name
            if self.semantic_model_path and not Path(self.semantic_model_path).exists():
                raise FileNotFoundError(f"Semantic model path does not exist: {self.semantic_model_path}")

            local_files_only = bool(self.semantic_model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            model = AutoModel.from_pretrained(
                model_source,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            model.eval()
            if self.semantic_device and self.semantic_device.lower() != "auto":
                model = model.to(self.semantic_device)

            self._torch = torch
            self._semantic_tokenizer = tokenizer
            self._semantic_model = model
            self._semantic_backend_ready = True
            return True
        except Exception as exc:
            self._semantic_backend_failed = True
            warnings.warn(
                "Falling back to lexical similarity for semantic scoring because the "
                f"sentence-transformer backend could not be loaded: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

    def _ensure_sentiment_backend(self) -> bool:
        if self._sentiment_backend_ready:
            return True
        if self._sentiment_backend_failed:
            return False

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_source = self.sentiment_model_path or self.sentiment_model_name
            if self.sentiment_model_path and not Path(self.sentiment_model_path).exists():
                raise FileNotFoundError(f"Sentiment model path does not exist: {self.sentiment_model_path}")

            local_files_only = bool(self.sentiment_model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            model.eval()
            if self.sentiment_device and self.sentiment_device.lower() != "auto":
                model = model.to(self.sentiment_device)

            self._sentiment_torch = torch
            self._sentiment_tokenizer = tokenizer
            self._sentiment_model = model
            self._sentiment_backend_ready = True
            return True
        except Exception as exc:
            self._sentiment_backend_failed = True
            warnings.warn(
                "Falling back to lexical sentiment scoring for stance because the "
                f"sentiment model could not be loaded: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

    def _ensure_ppl_backend(self) -> bool:
        if self._ppl_backend_ready:
            return True
        if self._ppl_backend_failed:
            return False

        try:
            if self.llm_model is None:
                raise ValueError("No generation model was provided for perplexity scoring.")

            torch = getattr(self.llm_model, "_torch", None)
            tokenizer = getattr(self.llm_model, "tokenizer", None)
            model = getattr(self.llm_model, "model", None)
            if tokenizer is None or model is None:
                pipeline = getattr(self.llm_model, "pipeline", None)
                if pipeline is not None:
                    tokenizer = getattr(pipeline, "tokenizer", None)
                    model = getattr(pipeline, "model", None)
                    torch = getattr(self.llm_model, "_torch", None)

            if tokenizer is None or model is None:
                raise ValueError("The active generation model does not expose tokenizer/model handles.")

            if torch is None:
                import torch as imported_torch

                torch = imported_torch

            model.eval()
            self._ppl_torch = torch
            self._ppl_tokenizer = tokenizer
            self._ppl_model = model
            self._ppl_backend_ready = True
            return True
        except Exception as exc:
            self._ppl_backend_failed = True
            warnings.warn(
                "Falling back to a length-based perplexity proxy because response perplexity "
                f"could not be computed from the active generation model: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

    def _encode_sentences(self, texts: list[str]):
        encoded = self._semantic_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if self.semantic_device and self.semantic_device.lower() != "auto":
            encoded = {key: value.to(self.semantic_device) for key, value in encoded.items()}
        else:
            encoded = {key: value.to(self._semantic_model.device) for key, value in encoded.items()}

        with self._torch.no_grad():
            outputs = self._semantic_model(**encoded)
            hidden = outputs.last_hidden_state
            # Sentence-Transformer style mean pooling over token embeddings.
            mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / counts
            return self._torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def _compute_response_ppl(self, response_text: str) -> float:
        prefix_text, full_text = self._build_ppl_texts(response_text)
        prefix_ids = self._ppl_tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
        full_ids = self._ppl_tokenizer(full_text, return_tensors="pt", add_special_tokens=False)

        input_ids = full_ids["input_ids"]
        attention_mask = full_ids.get("attention_mask")
        prompt_length = prefix_ids["input_ids"].shape[1]
        if input_ids.shape[1] <= prompt_length:
            return 1.0

        device = getattr(self._ppl_model, "device", None)
        if device is None:
            device = next(self._ppl_model.parameters()).device

        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        labels = input_ids.clone()
        labels[:, :prompt_length] = -100

        with self._ppl_torch.no_grad():
            outputs = self._ppl_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = float(outputs.loss.item())
        return float(math.exp(min(loss, 20.0)))

    def _build_ppl_texts(self, response_text: str) -> tuple[str, str]:
        if self.llm_model is not None and hasattr(self.llm_model, "build_messages"):
            prefix_messages = [
                {"role": "system", "content": getattr(self.llm_model, "_CONTINUATION_SYSTEM_PROMPT", "")},
                {"role": "user", "content": self._PPL_PROMPT},
            ]
            full_messages = prefix_messages + [{"role": "assistant", "content": response_text}]
            prefix_text = self._apply_chat_template(prefix_messages)
            full_text = self._apply_chat_template(full_messages)
            return prefix_text, full_text

        prefix_text = self._PPL_PROMPT
        full_text = f"{self._PPL_PROMPT}\n{response_text}"
        return prefix_text, full_text

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self._ppl_tokenizer, "apply_chat_template"):
            return self._ppl_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        return "\n".join(item["content"] for item in messages)

    def _cosine_similarity(self, left, right) -> float:
        return float(self._torch.sum(left * right).item())

    def _predict_sentiment_probs(self, text: str) -> dict[str, float]:
        encoded = self._sentiment_tokenizer(
            text,
            truncation=True,
            return_tensors="pt",
        )
        if self.sentiment_device and self.sentiment_device.lower() != "auto":
            encoded = {key: value.to(self.sentiment_device) for key, value in encoded.items()}
        else:
            encoded = {key: value.to(self._sentiment_model.device) for key, value in encoded.items()}

        with self._sentiment_torch.no_grad():
            logits = self._sentiment_model(**encoded).logits
            probs = self._sentiment_torch.nn.functional.softmax(logits, dim=-1)[0]

        id2label = getattr(self._sentiment_model.config, "id2label", {}) or {}
        positive = 0.0
        negative = 0.0
        for idx, prob in enumerate(probs):
            label = str(id2label.get(idx, f"LABEL_{idx}")).upper()
            value = float(prob.item())
            if "POS" in label or label.endswith("1"):
                positive = max(positive, value)
            elif "NEG" in label or label.endswith("0"):
                negative = max(negative, value)

        if positive == 0.0 and negative == 0.0 and len(probs) >= 2:
            negative = float(probs[0].item())
            positive = float(probs[-1].item())

        return {"positive": positive, "negative": negative}
