import math
from functools import lru_cache
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path


MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
MODEL_CACHE_DIR = Path("models/nli_cache")


SALES_LABELS = [
    "buying_intent",
    "urgency",
    "budget_discussion",
    "objection",
    "negative_intent",
    "no_current_need",
    "off_topic",
]

HYPOTHESES: Dict[str, List[str]] = {

    "buying_intent": [
        "This email asks to proceed with purchasing the product.",
        "This email explicitly requests to buy or acquire the product.",
        "This email indicates readiness to move forward with a purchase decision.",
        "This email asks for pricing in order to evaluate buying the product.",
        "This email requests a demo or trial to evaluate the product.",
        "This email asks for onboarding, contract setup, or activation.",
        "This email asks what the next step is to start using the product.",
        "This email expresses clear interest in purchasing the product.",
        "This email indicates the sender is ready to become a customer.",
        "This email signals intent to adopt the product in their organization.",
        "This email suggests procurement planning or internal buying discussion.",
        "This email compares this product with alternatives before deciding to buy.",
        "This email requests a proposal or contract document to proceed.",
        "This email asks for documentation needed to get internal approval.",
        "This email asks what is needed to get started or go live.",
        "This email is seeking formal paperwork to finalize a purchase.",
        "this email expresses interest in any form or wish"
    ],

    "urgency": [
        "This email asks for a response today or within a specific deadline.",
        "This email states a deadline that blocks next steps if missed.",
        "This email says the matter is time-sensitive and requires immediate action.",
        "This email requires urgent attention to avoid delays in decision-making.",
        "This email indicates a critical issue that needs immediate resolution.",
        "This email asks to prioritize this request over others.",
        "This email pressures for a fast decision or quick turnaround.",
        "This email indicates an expiring opportunity or time-limited offer.",
        "This email emphasizes that delays will negatively impact progress.",
        "This email mentions an end-of-quarter or fiscal deadline.",
        "This email says they need the solution live or deployed by a specific date.",
        "This email asks what needs to happen to meet a timeline.",
        "This email frames the request around an internal deadline or milestone.",
        "This email indicates a launch or go-live date is approaching.",
    ],

    "budget_discussion": [
        "This email asks for pricing or cost information.",
        "This email requests a quote or invoice.",
        "This email asks about discounts or pricing options.",
        "This email discusses payment terms or billing conditions.",
        "This email asks about refund, cancellation, or contract pricing terms.",
        "This email compares pricing between different solutions.",
        "This email requests clarification about total cost or licensing fees.",
        "This email discusses budget constraints or financial approval.",
        "This email asks whether the solution fits within a budget range.",
        "This email negotiates pricing or commercial conditions.",
        "This email explores cost breakdown or pricing structure details."
    ],

    "objection": [
        "This email says the product is not a good fit.",
        "This email gives reasons that prevent moving forward.",
        "This email states explicit hesitation or uncertainty about continuing.",
        "This email raises concerns about product suitability.",
        "This email challenges the value proposition of the product.",
        "This email expresses explicit doubt about implementation feasibility.",
        "This email questions whether the solution meets their needs.",
        "This email highlights risks or downsides of adopting the product.",
        "This email indicates resistance to proceeding further."
    ],

    "negative_intent": [
        "This email explicitly says they are not interested.",
        "This email rejects the product or proposal.",
        "This email asks to stop further contact.",
        "This email states they will not proceed with the offer.",
        "This email clearly declines any further engagement.",
        "This email indicates a firm decision against adoption.",
        "This email refuses to continue discussions about the product."
    ],

    "no_current_need": [
        "This email says they already use another solution.",
        "This email says they are not looking for a new solution.",
        "This email says the current system is sufficient.",
        "This email says they are not evaluating new tools.",
        "This email indicates no current requirement for the product.",
        "This email states that switching solutions is not necessary.",
        "This email mentions they are satisfied with existing tools.",
        "This email indicates the problem is already solved internally.",
        "This email says there is no business need at the moment.",
        "This email says they recently signed or renewed a contract with another vendor.",
        "This email indicates they are locked into an existing agreement.",
        "This email says they just committed to a competing solution.",
    ],

    "off_topic": [
        "This email is not about buying, evaluating, or adopting any product or service.",
        "This email does not discuss pricing, budget, or other commercial terms.",
        "This email does not express business interest, objection, or intent about a product or service.",
        "This email does not involve any decision-making, evaluation, or purchasing process.",
        "This email is not related to business, sales, or professional services discussions.",
        "This email is not about product evaluation or procurement decisions.",
        "This email is primarily administrative or operational communication unrelated to sales.",
        "This email is a personal or non-business message.",
        "This email is an automatic or system-generated notification.",
        "This email does not show any commercial intent or product evaluation.",
        "This email is not related to buying, budgeting, or procurement decisions."
    ]
}

POSITIVE_WEIGHTS = {
    "buying_intent": 1.5 / 2.9,
    "budget_discussion": 1 / 2.9,
    "urgency": 0.4 / 2.9,
}

NEGATIVE_WEIGHTS = {
    "objection": 0.1 / 7.1,
    "negative_intent": 1 / 7.1,
    "no_current_need": 1 / 7.1,
    "off_topic": 5 / 7.1,
}


@lru_cache(maxsize=1)
def get_mnli_model():
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
    )
    model.eval()
    return tokenizer, model


def _normalize_label(label: str) -> str:
    return label.lower().replace(" ", "_")


def _build_hypotheses(label: str) -> List[str]:
    return HYPOTHESES.get(label, [f"This email expresses {label}."])


def _mnli_logits(tokenizer, model, premise: str, hypothesis: str) -> torch.Tensor:
    """Return MNLI head logits over [contradiction, neutral, entailment]."""
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    return model(**inputs).logits[0]  # (3,)


def _score_entailment(tokenizer, model, premise: str, hypotheses: List[str]) -> float:
    """
    For each hypothesis, compute delta = e - max(n, c).
    This measures confidence that entailment dominates both other classes.
    Normalize with ReLU(tanh(delta)) to squash into [0, 1].
    OR-aggregate across hypotheses: any strong match lifts the label.
    """
    best_score = float("-inf")

    for hypothesis in hypotheses:
        logits = _mnli_logits(tokenizer, model, premise, hypothesis)
        c, e, n = logits[0], logits[1], logits[2]
        delta = e - max(n, c)
        score = delta.item()
        if score > best_score:
            best_score = score

    if best_score == float("-inf"):
        return 0.0

    return float(max(0.0, math.tanh(best_score)))


def _score_contradiction(tokenizer, model, premise: str, hypotheses: List[str]) -> float:
    """
    For each hypothesis, compute delta = c - max(n, e).
    This measures confidence that contradiction dominates both other classes.
    Normalize with ReLU(tanh(delta)) to squash into [0, 1].
    OR-aggregate across hypotheses.
    """
    best_score = float("-inf")

    for hypothesis in hypotheses:
        logits = _mnli_logits(tokenizer, model, premise, hypothesis)
        c, e, n = logits[0], logits[1], logits[2]
        delta = c - max(n, e)
        score = delta.item()
        if score > best_score:
            best_score = score

    if best_score == float("-inf"):
        return 0.0

    return float(max(0.0, math.tanh(best_score)))


def _score_label(tokenizer, model, premise: str, label: str) -> float:
    hypotheses = _build_hypotheses(label)

    raw_scores: List[float] = []
    for h in hypotheses:
        logits = _mnli_logits(tokenizer, model, premise, h)
        c, e, n = logits[0], logits[1], logits[2]
        delta = (e - max(n, c)).item()
        raw_scores.append(math.tanh(delta))  # signed, no ReLU yet

    if not raw_scores:
        return 0.0

    if label == "off_topic":
        # High when no hypothesis is confidently entailed
        # Use raw max so negative deltas (strong contradiction) also suppress off_topic
        max_e = max(raw_scores)
        return float(max(0.0, math.tanh(1.0 - max_e)))

    # ReLU only here — after OR-aggregation
    return float(max(0.0, max(raw_scores)))


def analyze_email_mnli(email_body: str, labels: List[str] | None = None) -> Dict:
    if not email_body or not email_body.strip():
        return {
            "labels": {},
            "engagement_score": 0,
            "bucket": "cold",
            "signal_type": "empty email",
            "top_label": None,
            "id2_label": None,

        }

    labels = labels or SALES_LABELS
    tokenizer, model = get_mnli_model()

    premise = email_body.strip()
    label_scores: Dict[str, float] = {}

    with torch.no_grad():
        for label in labels:
            label_scores[_normalize_label(label)] = round(
                _score_label(tokenizer, model, premise, label), 4
            )

    engagement_score = compute_email_engagement_score(label_scores)
    signal_type = get_email_signal_type(label_scores)
    top_label = max(label_scores, key=label_scores.get)

    return {
        "labels": label_scores,
        "engagement_score": engagement_score,
        "signal_type": signal_type,
        "top_label": top_label,
    }


def compute_email_engagement_score(label_scores: Dict[str, float]) -> float:
    positive_score = sum(
        label_scores.get(label, 0.0) * weight
        for label, weight in POSITIVE_WEIGHTS.items()
    )

    negative_score = sum(
        label_scores.get(label, 0.0) * weight
        for label, weight in NEGATIVE_WEIGHTS.items()
    )

    engagement_score = (positive_score - negative_score)

    return round(max(0, engagement_score), 2)


INTENT_GROUPS = {
    "active_buying": {"buying_intent", "budget_discussion", "urgency"},
    "risk_or_friction": {"objection"},
    "negative": {"negative_intent", "no_current_need"},
    "irrelevant": {"off_topic"},
}


def _group_scores(label_scores: Dict[str, float]) -> Dict[str, float]:
    return {
        group: max(label_scores.get(lbl, 0.0) for lbl in labels)
        for group, labels in INTENT_GROUPS.items()
    }


def get_email_signal_type(label_scores: Dict[str, float]) -> str:
    group_scores = _group_scores(label_scores)

    active = group_scores["active_buying"]
    friction = group_scores["risk_or_friction"]
    negative = group_scores["negative"]
    irrelevant = group_scores["irrelevant"]

    if irrelevant > 0.60:
        return "off topic"

    if negative > 0.65:
        return "explicit rejection"

    if active >= 0.35 and friction >= 0.35:
        return "purchase consideration with concerns"

    if active >= 0.45:
        return "active purchase consideration"

    if friction >= 0.45:
        return "concerns or objections raised"

    return "weak or unclear signal"

