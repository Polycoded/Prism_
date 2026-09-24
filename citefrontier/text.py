"""Small deterministic text helpers.

No fixture phrases are embedded here. The helpers operate on clauses and
token overlap so the same code can run on the supplied corpus later.
"""

from __future__ import annotations

import re

from .models import Intent

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "for", "from", "i",
        "in", "is", "it", "me", "my", "need", "of", "on", "or", "please", "the",
        "to", "we", "what", "with", "would", "you",
    }
)


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(text.lower()))


def content_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in tokenize(text) if token not in _STOPWORDS)


def normalized(text: str) -> str:
    return " ".join(tokenize(text))


def is_presentation_request(text: str) -> bool:
    lower = normalized(text)
    patterns = (
        "repeat my last answer",
        "last answer in",
        "make that answer",
        "put that in",
        "shorten the answer",
        "summarize the answer",
        "two bullets",
        "three bullets",
    )
    # Mixed requests must not hide a new factual question behind a format cue.
    if re.search(r"\b(?:and|also|but)\b", lower):
        return False
    return any(pattern in lower for pattern in patterns)


def is_non_retrieval_chitchat(text: str) -> bool:
    return normalized(text) in {"hi", "hello", "hey", "thanks", "thank you"}


def has_explicit_correction(text: str) -> bool:
    lower = normalized(text)
    return " instead" in f" {lower}" or " actually" in f" {lower}" or " rather than" in f" {lower}"


def split_intents(text: str) -> tuple[Intent, ...]:
    """Split a compound request into minimal searchable clauses.

    This is intentionally modest: it avoids a benchmark-specific intent
    ontology and lets a structured model replace it after the baseline is
    measured.
    """

    # Fronted scope belongs to every coordinated question, not a separate intent.
    scope = ""
    fronted = re.match(r"^\s*(?:for|regarding|about)\s+([^,;]+)[,;]\s*(.+)$", text, re.I)
    if fronted:
        scope, text = fronted.groups()
    clauses = re.split(r"(?:[;]|,\s*(?=(?:what|when|where|how|can|does|do|is)\b)|\b(?:and|also)\b)+", text, flags=re.IGNORECASE)
    intents: list[Intent] = []
    seen: set[str] = set()
    for clause in clauses:
        tokens = content_tokens(f"{scope} {clause}")
        if len(tokens) < 2:
            continue
        key = "-".join(tokens[:8])
        if key not in seen:
            intents.append(Intent(key=key, query=" ".join(tokens)))
            seen.add(key)

    if not intents:
        tokens = content_tokens(text)
        if tokens:
            intents.append(Intent(key="-".join(tokens[:8]), query=" ".join(tokens)))
    return tuple(intents)
