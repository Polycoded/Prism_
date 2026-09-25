import re


_FILLER_RE = re.compile(r"(?<!\w)(?:um+|uh+|erm+|er+|ah+)(?!\w)[,\s]*", re.IGNORECASE)
_REPETITION_RE = re.compile(r"\b([a-z][a-z0-9'-]{1,30})(?:\s+\1\b)+", re.IGNORECASE)
_CORRECTION_VALUE = r"[A-Za-z0-9][A-Za-z0-9:.%'-]*"
_EXPLICIT_VALUE_RE = re.compile(
    rf"(?P<old>{_CORRECTION_VALUE})\s*(?:—|–|-|,)\s*(?:sorry(?:,\s*I\s+meant)?|no|I\s+mean|actually|make\s+that),?\s*(?P<new>{_CORRECTION_VALUE})",
    re.IGNORECASE,
)
_ACCEPTED_VALUE_RE = re.compile(
    rf"(?P<new>{_CORRECTION_VALUE})\s*,?\s*not\s+(?P<old>{_CORRECTION_VALUE})",
    re.IGNORECASE,
)
_SUBJECT_CORRECTION_RE = re.compile(
    r"[,;.!?\s]+(?:sorry[,\s]+(?:I\s+meant\s+)?|I\s+meant\s+|actually[\s,]*(?:I\s+meant\s+)?)([^,;.!?]+)[.!?\s]*$",
    re.IGNORECASE,
)


def _collapse_spacing(text):
    text = re.sub(r"\s+([,?.!])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,")


def _looks_like_model_id(value):
    return bool(re.fullmatch(r"[A-Za-z]+\d+", value or ""))


def _contains_alias(text, value):
    pattern = r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def normalize_disfluencies(text):
    cleaned = text or ""
    changes = []

    while True:
        match = _FILLER_RE.search(cleaned)
        if not match:
            break
        filler = match.group(0).strip(" ,")
        if filler:
            changes.append({"type": "filler", "text": filler})
        cleaned = cleaned[: match.start()] + " " + cleaned[match.end() :]

    while True:
        match = _REPETITION_RE.search(cleaned)
        if not match:
            break
        word = match.group(1)
        changes.append({"type": "repetition", "text": word})
        cleaned = cleaned[: match.start()] + word + cleaned[match.end() :]

    cleaned = _collapse_spacing(cleaned)
    return cleaned, changes


def _normalize_explicit_value_correction(text):
    for pattern, kind in ((_EXPLICIT_VALUE_RE, "explicit_value"), (_ACCEPTED_VALUE_RE, "accepted_value")):
        match = pattern.search(text)
        if not match:
            continue

        old = match.group("old")
        new = match.group("new")
        if _looks_like_model_id(old) or _looks_like_model_id(new):
            continue

        replacement = text[: match.start()] + new + text[match.end() :]
        replacement = re.sub(r"\s+([,.?!])", r"\1", replacement)
        replacement = re.sub(r"\s{2,}", " ", replacement)
        replacement = replacement.strip(" ,")
        return replacement, {"type": kind, "from": old, "to": new}

    return text, None


def _normalize_subject_correction(text, aliases):
    if not aliases:
        return text, None, False

    match = _SUBJECT_CORRECTION_RE.search(text)
    if not match:
        return text, None, False

    prefix = text[: match.start()]
    replacement = match.group(1).strip()
    if not replacement:
        return text, None, False

    alias_values = list(dict.fromkeys(str(value) for value in aliases.values()))
    subjects = [value for value in alias_values if _contains_alias(prefix, value)]
    if not subjects:
        return text, None, False

    candidates = [value for value in alias_values if value.lower() == replacement.lower()]
    if len(subjects) == 1 and not candidates:
        subject = subjects[0]
        family = " ".join(subject.split()[:-1])
        replacement_last = replacement.split()[-1] if replacement.split() else replacement
        for value in alias_values:
            value_tokens = value.split()
            if len(value_tokens) > 1 and " ".join(value_tokens[:-1]).lower() == family.lower() and value_tokens[-1].lower() == replacement_last.lower():
                candidates.append(value)

    explicit = bool(re.search(r"sorry|I\s+meant", match.group(0), re.IGNORECASE))
    if not candidates:
        if not explicit and not re.fullmatch(r"[A-Za-z]+\s*\d+", replacement):
            return text, None, False
        return text, None, True

    if len(subjects) != 1 or len(candidates) != 1:
        return text, None, True

    subject = subjects[0]
    candidate = candidates[0]
    if subject.lower() == candidate.lower():
        return text, None, False

    new_prefix = prefix.replace(subject, candidate, 1)
    new_text = new_prefix + text[match.end() :]
    return new_text, {"from": subject, "to": candidate}, False


def normalize_correction(text, aliases):
    text = text or ""
    aliases = aliases or {}

    corrected, explicit_change = _normalize_explicit_value_correction(text)
    if explicit_change is not None:
        normalized, subject_change, ambiguous = _normalize_subject_correction(corrected, aliases)
        if subject_change is not None or ambiguous:
            return normalized, subject_change, ambiguous
        return corrected, explicit_change, False

    normalized, subject_change, ambiguous = _normalize_subject_correction(text, aliases)
    if subject_change is not None or ambiguous:
        return normalized, subject_change, ambiguous
    return text, None, False
