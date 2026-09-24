"""Curated synthetic venue-domain boundary gate for the optional adapter.

This produces a review set, not human-annotated real-user ground truth. A
human must approve or correct the JSONL before it can serve as a promotion
gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from citefrontier.models import CorpusChunk
from prototype.bert_decomposition import BertIntentAdapter, DistilBertBoundaryDetector
from prototype.decomposition import Decomposer

ROOT = Path(__file__).resolve().parents[1]


def case(case_id, pieces, joiners, prefix=""):
    text = prefix
    spans = []
    for index, piece in enumerate(pieces):
        if index:
            text += joiners[index - 1]
        start = len(text)
        text += piece
        spans.append([start, len(text)])
    return {"id": case_id, "text": text, "gold_spans": spans,
            "annotation_source": "curated_synthetic_pending_human_review"}


def build_cases():
    entities = ["Venue A", "Venue B", "Riverside Hall", "Orchid Center"]
    properties = [
        "what is the capacity", "what is the cancellation policy",
        "what catering options are available", "is parking available",
        "what does the venue cost", "is outside food permitted",
        "what equipment is included", "is the venue wheelchair accessible",
        "when does the venue close", "is a deposit required",
    ]
    cases = []
    # Thirty single-intent controls, including coordination that must remain
    # inside one request.
    singles = [
        "Can Venue A fit between 100 and 200 people?",
        "Explain the terms and conditions for Venue A.",
        "Book Venue A and Venue B for one workshop.",
        "Does Riverside Hall include tables and chairs?",
        "Is food and beverage service available at Orchid Center?",
        "What audio and video equipment is included at Venue A?",
    ]
    for i in range(30):
        text = singles[i % len(singles)] if i < len(singles) else f"For {entities[i % 4]}, {properties[i % 10]}?"
        if text.startswith("For "):
            start = text.index(",") + 2
        else:
            start = 0
        end = len(text.rstrip(" .?!"))
        cases.append({"id": f"venue-single-{i+1:03d}", "text": text,
                      "gold_spans": [[start, end]],
                      "annotation_source": "curated_synthetic_pending_human_review"})
    # Thirty explicit two-intent requests with shared scope.
    for i in range(30):
        p1, p2 = properties[i % 10], properties[(i + 3) % 10]
        prefix = f"For {entities[i % 4]}, "
        cases.append(case(f"venue-explicit-{i+1:03d}", [p1, p2], [" and "], prefix))
    # Thirty elliptical/list-like requests.
    fragments = [
        "capacity for 300 people", "the cancellation policy", "catering options",
        "parking availability", "the required deposit", "closing time",
        "wheelchair access", "included projectors", "outside food rules", "the rental price",
    ]
    for i in range(30):
        pieces = [f"I need {entities[i % 4]} {fragments[i % 10]}", fragments[(i + 2) % 10]]
        cases.append(case(f"venue-elliptical-{i+1:03d}", pieces, [", and "]))
    # Thirty three-intent requests, including disfluent connectors.
    for i in range(30):
        pieces = [properties[i % 10], properties[(i + 2) % 10], properties[(i + 5) % 10]]
        joiners = ["; ", "; "] if i % 2 == 0 else [", and also ", ", and then "]
        cases.append(case(f"venue-three-{i+1:03d}", pieces, joiners, f"Regarding {entities[i % 4]}, "))
    # Forty punctuation-free ASR-like compounds with fillers and restarts.
    for i in range(40):
        entity = entities[i % 4]
        p1, p2 = properties[i % 10], properties[(i + 4) % 10]
        prefix = f"um for {entity} " if i % 2 == 0 else f"okay {entity} "
        joiner = " and uh " if i % 3 == 0 else " also "
        cases.append(case(f"venue-asr-{i+1:03d}", [p1, p2], [joiner], prefix))
    # Forty single-intent hard negatives where coordination is internal.
    negatives = [
        "Can Venue A hold between 100 and 200 attendees",
        "Explain Venue B terms and conditions",
        "Does Riverside Hall provide tables and chairs",
        "Is food and beverage service available at Orchid Center",
        "Book Venue A and Venue B for the same event",
        "Do Alice and Bob count toward the guest limit",
        "What audio and video equipment comes with Venue A",
        "Is research and development space available at Venue B",
    ]
    for i in range(40):
        text = negatives[i % len(negatives)]
        cases.append({"id": f"venue-negative-{i+1:03d}", "text": text,
                      "gold_spans": [[0, len(text)]],
                      "annotation_source": "curated_synthetic_pending_human_review"})
    return cases


def normalized_rule_spans(decomposer, text):
    result = []
    for intent in decomposer.split(text):
        start, end = intent.source_spans[-1]
        while end > start and text[end-1] in " ,;?.!":
            end -= 1
        result.append([start, end])
    return result


def main():
    chunks = tuple(CorpusChunk(
        f"V{i}:1", f"V{i}", "Venue.1", f"{name} venue information.", {"entity": name}
    ) for i, name in enumerate(["Venue A", "Venue B", "Riverside Hall", "Orchid Center"], 1))
    rules = Decomposer(chunks, parser="rules")
    detector = DistilBertBoundaryDetector(ROOT / "model" / "checkpoints" / "best")
    bert = BertIntentAdapter(rules, detector, threshold=0.90)
    cases = build_cases()
    bert_correct = rule_correct = auto_correct = accepted = fallback = 0
    latencies = []
    errors = []
    for row in cases:
        try:
            intents, prediction = bert.split(row["text"])
            predicted = [list(intent.source_spans[-1]) for intent in intents]
            accepted += 1
        except Exception as exc:
            predicted, prediction = [], bert.last_prediction()
            fallback += 1
            row["fallback_reason"] = str(exc)
        if prediction:
            latencies.append(prediction.latency_ms)
        rule_spans = normalized_rule_spans(rules, row["text"])
        bert_ok = predicted == row["gold_spans"]
        rule_ok = rule_spans == row["gold_spans"]
        minimum = min((s.confidence for s in prediction.spans), default=0.0) if prediction else 0.0
        auto_spans = predicted if predicted and (predicted == rule_spans or minimum >= .97) else rule_spans
        auto_ok = auto_spans == row["gold_spans"]
        bert_correct += bert_ok
        rule_correct += rule_ok
        auto_correct += auto_ok
        if not bert_ok and len(errors) < 30:
            errors.append({"id": row["id"], "text": row["text"], "gold": row["gold_spans"],
                           "bert": predicted, "rules": rule_spans})
        row["bert_spans"] = predicted
        row["rule_spans"] = rule_spans
        row["bert_correct"] = bert_ok
        row["rules_correct"] = rule_ok
        row["auto_spans"] = auto_spans
        row["auto_correct"] = auto_ok
        row["bert_min_confidence"] = minimum if prediction else None
    results = {
        "dataset": "venue_domain_curated_synthetic_v1",
        "examples": len(cases),
        "annotation_status": "pending_human_review",
        "promotion_eligible": False,
        "bert": {"exact": bert_correct, "rate": bert_correct/len(cases),
                 "accepted": accepted, "fallback": fallback,
                 "accepted_precision": bert_correct/accepted if accepted else None,
                 "threshold": 0.90,
                 "warm_latency_ms": {
                     "p50": median(latencies[1:]) if len(latencies) > 1 else None,
                     "p95": sorted(latencies[1:])[int((len(latencies[1:])-1)*.95)] if len(latencies) > 1 else None,
                     "cold_first": latencies[0] if latencies else None
                 }},
        "rules": {"exact": rule_correct, "rate": rule_correct/len(cases)},
        "auto": {"exact": auto_correct, "rate": auto_correct/len(cases),
                 "improvement_over_rules_percentage_points": 100*(auto_correct-rule_correct)/len(cases)},
        "errors": errors,
        "decision": "Shadow integration only until the review set is independently human-approved and corrected."
    }
    (ROOT / "results" / "venue_domain_review.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases), encoding="utf-8"
    )
    (ROOT / "results" / "venue_domain_evaluation.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
