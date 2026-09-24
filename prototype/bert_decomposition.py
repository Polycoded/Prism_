"""Optional DistilBERT intent-boundary inference and guarded integration."""
from __future__ import annotations

import os
import re
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path

from .decomposition import Decomposer, Intent, digest, terms


LABELS = ("O", "B-INTENT", "I-INTENT")


@dataclass(frozen=True)
class BoundarySpan:
    start_char: int
    end_char: int
    confidence: float
    text: str


@dataclass(frozen=True)
class BoundaryPrediction:
    spans: tuple[BoundarySpan, ...]
    labels: tuple[str, ...]
    word_confidences: tuple[float, ...]
    latency_ms: float
    valid: bool
    failure_reason: str | None = None

    def public(self):
        return asdict(self)


class DistilBertBoundaryDetector:
    """Lazy, local-only token classifier with whitespace-to-character alignment."""

    def __init__(self, model_path: str | Path, max_length: int = 256):
        self.model_path = Path(model_path).resolve()
        self.max_length = max_length
        self._tokenizer = self._model = self._torch = None

    def _load(self):
        if self._model is not None:
            return
        if not self.model_path.joinpath("config.json").exists():
            raise FileNotFoundError(f"BERT boundary checkpoint not found: {self.model_path}")
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, use_fast=True
        )
        self._model = AutoModelForTokenClassification.from_pretrained(
            self.model_path, local_files_only=True
        )
        self._model.eval()

    def predict(self, text: str) -> BoundaryPrediction:
        started = time.perf_counter()
        matches = list(re.finditer(r"\S+", text))
        if not matches:
            return BoundaryPrediction((), (), (), 0.0, False, "empty_input")
        try:
            self._load()
            words = [m.group() for m in matches]
            encoded = self._tokenizer(
                words, is_split_into_words=True, return_tensors="pt",
                truncation=True, max_length=self.max_length,
            )
            word_ids = encoded.word_ids(batch_index=0)
            model_inputs = {k: v for k, v in encoded.items() if k != "offset_mapping"}
            with self._torch.inference_mode():
                logits = self._model(**model_inputs).logits[0]
                probs = self._torch.softmax(logits, dim=-1)
            labels, confidences, seen = [], [], set()
            for word_id, vector in zip(word_ids, probs):
                if word_id is None or word_id in seen:
                    continue
                seen.add(word_id)
                index = int(self._torch.argmax(vector).item())
                labels.append(LABELS[index])
                confidences.append(float(vector[index].item()))
            if len(labels) != len(words):
                return BoundaryPrediction((), tuple(labels), tuple(confidences),
                    round((time.perf_counter()-started)*1000, 2), False, "input_truncated")
            spans, active = [], None
            for index, label in enumerate(labels + ["O"]):
                if label == "B-INTENT":
                    if active is not None:
                        spans.append((active, index - 1))
                    active = index
                elif label == "I-INTENT":
                    if active is None:
                        return BoundaryPrediction((), tuple(labels), tuple(confidences),
                            round((time.perf_counter()-started)*1000, 2), False,
                            f"invalid_I_transition_at_word_{index}")
                elif active is not None:
                    spans.append((active, index - 1))
                    active = None
            result = tuple(BoundarySpan(
                matches[a].start(), matches[b].end(),
                min(confidences[a:b+1]), text[matches[a].start():matches[b].end()]
            ) for a, b in spans)
            return BoundaryPrediction(result, tuple(labels), tuple(confidences),
                round((time.perf_counter()-started)*1000, 2), bool(result),
                None if result else "no_intent_span")
        except Exception as exc:
            return BoundaryPrediction((), (), (),
                round((time.perf_counter()-started)*1000, 2), False,
                f"model_error:{type(exc).__name__}:{str(exc)[:160]}")


class BertIntentAdapter:
    """Converts source-aligned BERT spans into the production Intent contract."""

    def __init__(self, rules: Decomposer, detector, threshold: float = 0.90,
                 max_spans: int = 8):
        self.rules = rules
        self.detector = detector
        self.threshold = threshold
        self.max_spans = max_spans
        self.aliases = rules.aliases
        self.method = "bert_intent_boundary"
        self._prediction = ContextVar(f"bert_adapter_prediction_{id(self)}", default=None)

    def entities(self, text):
        return self.rules.entities(text)

    def topic(self, text, entities=()):
        return self.rules.topic(text, entities)

    def split(self, text):
        prediction = self.detector.predict(text)
        self._prediction.set(prediction)
        if not prediction.valid:
            raise ValueError(prediction.failure_reason or "invalid_prediction")
        if len(prediction.spans) > self.max_spans:
            raise ValueError("excessive_span_count")
        if min(span.confidence for span in prediction.spans) < self.threshold:
            raise ValueError("confidence_below_threshold")
        prior_end = -1
        inherited = self.entities(text)
        if len(inherited) != 1:
            inherited = ()
        intents, seen = [], set()
        covered = [False] * len(text)
        for span_index, span in enumerate(prediction.spans):
            if span.start_char < prior_end or not (0 <= span.start_char < span.end_char <= len(text)):
                raise ValueError("overlapping_or_invalid_span")
            prior_end = span.end_char
            for position in range(span.start_char, span.end_char):
                covered[position] = True
            raw = text[span.start_char:span.end_char]
            left_trim = len(raw) - len(raw.lstrip(" ,;?.!"))
            cleaned = raw.strip(" ,;?.!")
            if not cleaned or not terms(cleaned):
                raise ValueError("empty_or_glue_only_span")
            # The ATIS/SNIPS construction occasionally teaches the model to
            # emit a fronted shared scope as its own span. Scope is metadata,
            # not a retrievable request, so omit it and inherit its entity in
            # the following request spans.
            if (span_index == 0 and len(prediction.spans) > 1 and
                    re.match(r"^(?:for|regarding|about)\b", cleaned, re.I) and
                    not re.search(r"\b(?:what|when|where|how|can|could|does|do|is|are|which|who)\b", cleaned, re.I)):
                continue
            start = span.start_char + left_trim
            connective = re.match(r"^(?:and(?:\s+(?:also|then))?|also|then)\s+", cleaned, re.I)
            if connective:
                start += connective.end()
                cleaned = cleaned[connective.end():]
            # Speech transcripts often attach a filler to the following
            # request even though it carries no retrieval meaning.
            filler = re.match(r"^(?:um+|uh+|erm+|okay|ok|well)\b[\s,]*", cleaned, re.I)
            if filler:
                start += filler.end()
                cleaned = cleaned[filler.end():]
            # Remove a fronted shared venue scope when the model includes it
            # in the first request span. Entity extraction still runs over
            # the complete utterance, so the scope is inherited below.
            if span_index == 0:
                request = re.search(
                    r"\b(?:what|when|where|how|can|could|does|do|is|are|which|who)\b",
                    cleaned,
                    re.I,
                )
                prefix = cleaned[:request.start()] if request else ""
                prefix_has_scope = bool(
                    prefix
                    and (
                        re.match(r"^(?:for|regarding|about)\b", prefix, re.I)
                        or any(self.aliases[entity].lower() in prefix.lower() for entity in inherited)
                    )
                )
                if request and prefix_has_scope:
                    start += request.start()
                    cleaned = cleaned[request.start():]
            end = start + len(cleaned)
            local_entities = self.entities(cleaned)
            entity_ids = local_entities or inherited
            subject = " / ".join(self.aliases[e] for e in entity_ids)
            query = cleaned
            if subject and not local_entities:
                query = subject + " " + re.sub(
                    r"\b(?:it|its|their|they|them)\b", "", cleaned, flags=re.I
                )
            topic = tuple(sorted(self.topic(query, entity_ids)))
            constraints = tuple(re.findall(
                r"\b(?:\d+(?:\.\d+)?(?:\s*(?:people|month[s]?|year[s]?|day[s]?|ghz|percent))?|not|without|before|after|only)\b",
                cleaned.lower(),
            ))
            fingerprint = digest(repr((entity_ids, topic, constraints)))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            intents.append(Intent(fingerprint, query.strip(), subject, entity_ids,
                                  topic, constraints, ((start, end),), self.method))
        if not intents:
            raise ValueError("no_usable_intents")
        uncovered = "".join(character if not covered[index] else " " for index, character in enumerate(text))
        allowed_gap_words = r"and|also|then|plus|but|or|for|regarding|about|um+|uh+|erm+|okay|ok|well"
        uncovered = re.sub(rf"\b(?:{allowed_gap_words})\b", " ", uncovered, flags=re.I)
        for entity in inherited:
            uncovered = re.sub(re.escape(self.aliases[entity]), " ", uncovered, flags=re.I)
        if terms(uncovered):
            raise ValueError("meaningful_uncovered_content")
        return tuple(intents), prediction

    def last_prediction(self):
        return self._prediction.get()


class HybridDecomposer:
    """Rules-compatible decomposer with BERT shadowing and safe fallback."""

    def __init__(self, chunks, mode: str, fallback_parser: str = "spacy",
                 model_path: str | Path | None = None, threshold: float = 0.90,
                 detector=None):
        if mode not in {"bert", "shadow", "auto"}:
            raise ValueError("BERT decomposer mode must be bert, shadow, or auto")
        self.mode = mode
        self.rules = Decomposer(chunks, fallback_parser)
        default_model = Path(__file__).resolve().parents[1] / "bert_intent_tagger" / "model" / "checkpoints" / "best"
        self.detector = detector or DistilBertBoundaryDetector(model_path or default_model)
        self.bert = BertIntentAdapter(self.rules, self.detector, threshold)
        self.disagree_threshold = float(os.environ.get("CITEFRONTIER_BERT_DISAGREE_THRESHOLD", ".97"))
        self.aliases = self.rules.aliases
        self.method = f"{mode}:bert_intent_boundary+{self.rules.method}"
        self._diagnostic = ContextVar(f"bert_decomposer_diagnostic_{id(self)}", default=None)

    def entities(self, text):
        return self.rules.entities(text)

    def topic(self, text, entities=()):
        return self.rules.topic(text, entities)

    @staticmethod
    def _signature(intents, text):
        result = []
        for intent in intents:
            if not intent.source_spans:
                continue
            start, end = intent.source_spans[-1]
            while end > start and text[end-1] in " ,;?.!":
                end -= 1
            result.append((start, end))
        return tuple(result)

    def split(self, text):
        rule_intents = self.rules.split(text)
        try:
            bert_intents, prediction = self.bert.split(text)
            accepted, failure = True, None
            agrees = self._signature(rule_intents, text) == self._signature(bert_intents, text)
            minimum = min((span.confidence for span in prediction.spans), default=0.0)
            if not agrees and minimum < self.disagree_threshold:
                raise ValueError("disagreement_confidence_below_threshold")
        except Exception as exc:
            bert_intents, prediction = (), self.bert.last_prediction()
            accepted, failure, agrees = False, str(exc), False
        served = rule_intents if self.mode == "shadow" or not accepted else bert_intents
        self._diagnostic.set({
            "mode": self.mode,
            "served": "rules" if served is rule_intents else "bert",
            "accepted": accepted,
            "fallback_reason": failure,
            "agreement": accepted and agrees,
            "rule_spans": [list(span) for span in self._signature(rule_intents, text)],
            "bert_spans": [list(span) for span in self._signature(bert_intents, text)],
            "bert_latency_ms": prediction.latency_ms if prediction else None,
            "min_confidence": min((span.confidence for span in prediction.spans), default=None) if prediction else None,
        })
        return served

    def consume_diagnostic(self):
        value = self._diagnostic.get()
        self._diagnostic.set(None)
        return value
