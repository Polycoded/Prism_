"""Pretrained dependency analysis, conservative splitting, and scoped rewriting."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from threading import RLock

WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
FILLER = set("a an the and or for to of on in at is are was were be been being it its they their them this that these those i we you my our your please tell show give summarize repeat me about what when where which who why how do does did can could would should must will has have had need needed required want know much many long get with from by as s not only also actually instead than rather now thanks thank hello hi open opens opening applies apply kept done standard used use often".split())
SYNONYMS = {"guarantee":"warranty", "guarantees":"warranty", "protection":"warranty", "duration":"period", "length":"period", "last":"period", "lasts":"period", "covered":"coverage", "cover":"coverage", "covers":"coverage", "exclusion":"exclude", "exclusions":"exclude", "excluded":"exclude", "excluding":"exclude", "proof":"receipt", "receipts":"receipt", "documents":"receipt", "documentation":"receipt", "water":"liquid", "dropped":"drop", "drops":"drop", "accidental":"damage", "scratches":"scratch", "purchase":"purchase", "purchased":"purchase", "cancel":"cancellation", "cancelling":"cancellation", "canceled":"cancellation", "replaced":"replace", "replacement":"replace", "changing":"change", "installation":"install", "installed":"install", "setting":"setup", "set":"setup", "processed":"process", "processing":"process", "locally":"local", "complimentary":"charge", "free":"charge", "trial":"plus", "retain":"history", "retained":"history", "retention":"history", "footage":"video", "months":"month", "years":"year", "days":"day", "seconds":"second", "requirements":"require", "requirement":"require", "repairs":"repair", "batteries":"battery", "hours":"hour", "profiles":"profile", "settings":"setting", "features":"feature", "offers":"offer", "includes":"include"}


def words(text):
    return WORD.findall(text.lower())


def terms(text):
    result = set(SYNONYMS.get(w, w) for w in words(text) if w not in FILLER)
    if re.search(r"\b\d+[ -](?:month|year|day|hour)s?\b", text, re.I):
        result.add("period")
    if re.search(r"\bhow long\b", text, re.I):
        result.add("period")
    return result


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Intent:
    key: str
    query: str
    subject: str
    entity_ids: tuple[str, ...]
    topic: tuple[str, ...]
    constraints: tuple[str, ...]
    source_spans: tuple[tuple[int, int], ...]
    method: str

    def public(self):
        return asdict(self)


class Decomposer:
    def __init__(self, chunks, parser="spacy"):
        self.aliases = {c.doc_id: c.metadata.get("entity", c.doc_id.replace("_", " ")) for c in chunks}
        # Section names supply domain vocabulary without query/answer fixtures.
        self.properties = set().union(*(terms(c.section.split('.')[0]) for c in chunks))
        self.lock = RLock()
        self.nlp = None
        if parser == "spacy":
            import spacy
            self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        elif parser != "rules":
            raise ValueError("Parser must be spacy or rules")
        self.method = "spacy_dependency_rules" if self.nlp else "structural_rules"

    def entities(self, text):
        return tuple(doc for doc, alias in self.aliases.items()
                     if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.I))

    def topic(self, text, entities=()):
        for entity in entities:
            text = re.sub(re.escape(self.aliases[entity]), " ", text, flags=re.I)
        return terms(text)

    def split(self, text):
        original = text
        scope = ""
        base = 0
        front = re.match(r"\s*(?:for|regarding|about)\s+([^,;]+)[,;]\s*", text, re.I)
        if front and self.entities(front[1]) and not self.topic(front[1], self.entities(front[1])):
            scope = front[1]
            base = front.end()
            text = text[base:]
        cuts = set()
        protected = [(m.start(), m.end()) for m in re.finditer(r"\b(?:between\b[^,;?]+?\band\s+\d+|terms and conditions|research and development)\b", text, re.I)]
        for alias in self.aliases.values():
            protected.extend((m.start(), m.end()) for m in re.finditer(re.escape(alias), text, re.I))
        with self.lock:
            doc = self.nlp(text) if self.nlp else None
        for m in re.finditer(r";\s*|[?!.]\s+(?=(?:what|when|where|how|can|do|does|is|are|which|who)\b)|,\s*(?=(?:what|when|where|how|can|do|does|is)\b)|\b(?:and\s+also|and|also)\s+", text, re.I):
            if any(a <= m.start() < b for a, b in protected):
                continue
            right = text[m.end():]
            left = text[max((b for _, b in cuts), default=0):m.start()]
            explicit = bool(re.match(r"(?:what|when|where|how|can|does|do|is|are|which|who)\b", right, re.I))
            has_request = bool(re.search(r"\b(?:what|when|where|how|tell|show|give|check|need|want)\b", left, re.I))
            # Noun coordination may request distinct properties. Do not split
            # a number range, list of constraints, or unrelated verb object.
            noun_conj = False
            if doc:
                token = next((t for t in doc if t.idx >= m.end()), None)
                noun_conj = bool(token and (token.dep_ == "conj" or token.head.dep_ == "conj") and
                                 token.pos_ in {"NOUN", "ADJ", "DET"} and has_request)
            # A dependency conjunction alone also matches "receipt and serial
            # number". Require distinct corpus properties for implicit requests.
            left_properties = self.topic(left, self.entities(left)) & self.properties
            right_properties = self.topic(right, self.entities(right)) & self.properties
            distinct_properties = bool(left_properties and right_properties and left_properties.isdisjoint(right_properties))
            if explicit or m[0].startswith(";") or (distinct_properties and (noun_conj or scope)):
                cuts.add((m.start(), m.end()))
        segments = []
        start = 0
        for a, b in sorted(cuts):
            segments.append((start, a))
            start = b
        segments.append((start, len(text)))
        inherited = self.entities(scope) or self.entities(original)
        if len(inherited) != 1:
            inherited = ()
        intents = []
        seen = set()
        recent_entities = ()
        for a, b in segments:
            raw = text[a:b].strip(" ,;?")
            if not terms(raw):
                continue
            local_entities = self.entities(raw)
            pronoun = bool(re.search(r"\b(?:it|its|they|their|them)\b", raw, re.I))
            entity_ids = local_entities or (recent_entities if pronoun and len(recent_entities) == 1 else inherited)
            if local_entities:
                recent_entities = local_entities
            subject = " / ".join(self.aliases[e] for e in entity_ids)
            query = raw
            if subject and not local_entities:
                query = subject + " " + re.sub(r"\b(?:it|its|their|they|them)\b", "", raw, flags=re.I)
            topic = tuple(sorted(self.topic(query, entity_ids)))
            constraints = tuple(re.findall(r"\b(?:\d+(?:\.\d+)?(?:\s*(?:people|month[s]?|year[s]?|day[s]?|ghz|percent))?|not|without|before|after|only)\b", raw.lower()))
            fingerprint = digest(repr((entity_ids, topic, constraints)))
            # Exact canonical equivalence is a complete-link compatibility
            # relation. No transitive embedding-only merging is permitted.
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            spans = ((base + a, base + b),)
            if scope:
                spans = ((0, base),) + spans
            intents.append(Intent(fingerprint, query.strip(), subject, entity_ids, topic, constraints, spans, self.method))
        return tuple(intents)


def presentation(text):
    """Full-match grammar; mixed factual requests cannot be suppressed."""
    if translation_request(text):
        return True
    return bool(re.fullmatch(r"\s*(?:please\s+)?(?:put (?:that|it|the answer) in (?:\w+ )?bullets|(?:show|repeat) (?:that|it|the answer|my last answer|your last answer) (?:as|in) (?:\w+ )?bullets|(?:shorten|summarize|repeat) (?:that|it|the answer|my last answer|your last answer)|make (?:that|it|the answer) shorter)[.!?\s]*", text, re.I))


def social(text):
    return " ".join(words(text)) in {"hi", "hello", "hey", "thanks", "thank you"}


def translation_request(text):
    return bool(re.fullmatch(r"\s*(?:please\s+)?translate\s+(?:that|it|the answer|your last answer|my last answer)\s+(?:into|to)\s+[a-z]+[.!?\s]*", text, re.I))


def presentation_prefix(text):
    """Wait on potential formatting commands without suppressing mixed facts."""
    if re.search(r"\b(?:and|also|but|what|where|when|why|how|does|can)\b", text, re.I):
        return False
    return bool(re.match(r"^\s*(?:please\s+)?(?:repeat|show|put|shorten|summarize|make|translate)\s+(?:that|it|the answer|your(?:\s+last)?(?:\s+answer)?|my(?:\s+last)?(?:\s+answer)?)\b", text, re.I))
