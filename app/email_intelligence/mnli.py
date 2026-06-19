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
]

HYPOTHESES: Dict[str, List[str]] = {

     "buying_intent": [
        "The sender explicitly states they want to buy the product.",
        "The sender explicitly states they are ready to proceed with a purchase.",
        "The sender explicitly asks to start the purchasing process.",
        "The sender explicitly asks for a contract or agreement to begin.",
        "The sender explicitly states an intention to become a customer.",
        "The sender explicitly states that a decision to purchase has been made.",
    ],

    "objection": [
        "The sender explicitly states they have concerns about the product.",
        "The sender explicitly states the product may not meet their needs.",
        "The sender explicitly raises doubts about whether to proceed.",
        "The sender explicitly mentions a risk or problem with the product.",
        "The sender explicitly says something is blocking them from proceeding.",
    ],

    "urgency": [
        "The sender explicitly states a deadline.",
        "The sender asks for an immediate response.",
        "The sender says a delay will block progress.",
        "The sender indicates time pressure on the decision.",
        "The sender states the request is urgent.",
        "The sender says action is required within a short time frame.",
    ],

    "budget_discussion": [
        "The sender asks about the price of the product.",
        "The sender requests a quote or pricing information.",
        "The sender discusses budget constraints.",
        "The sender asks whether the cost fits their budget.",
        "The sender negotiates pricing or payment terms.",
        "The sender asks about billing or invoicing.",
    ],

    "negative_intent": [
        "The sender explicitly states they are not interested.",
        "The sender states they will not proceed.",
        "The sender asks to stop further communication.",
        "The sender rejects the product or proposal.",
        "The sender declines any further engagement.",
    ],

    "no_current_need": [
        "The sender explicitely states they do not currently need the product.",
        "The sender explicitely says they are not looking for a solution right now.",
        "The sender explicitely states their current solution is sufficient.",
        "The sender explicitely says they are not evaluating alternatives.",
        "The sender explicitely indicates no active project related to this product.",
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

def compute_off_topic(label_scores: Dict[str, float]) -> float:
    if not label_scores:
        return 1.0
    max_signal = max(label_scores.values())
    return round(max(0.0, 1.0 - max_signal), 4)

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

    if label == "unclear_or_off_topic":
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
    label_scores["unclear_or_off_topic"] = compute_off_topic(label_scores)
    
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

def run_mnli_benchmark() -> Dict:
    """
      Runs a fixed benchmark suite against the MNLI email analyzer.
      No parameters. Safe for API exposure.
    """

    BENCHMARK_EMAILS = [
      # --- Strong buying intent ---
      {
          "id": "buy_1",
          "email": "We want to buy your product.",
          "expect": "buying_intent",
      },
      {
          "id": "buy_2",
          "email": "We are ready to move forward with the purchase.",
          "expect": "buying_intent",
      },
      {
          "id": "buy_3",
          "email": "We would like to go ahead and purchase your software.",
          "expect": "buying_intent",
      },
      {
          "id": "buy_4",
          "email": "Please count us in, we want to become a customer.",
          "expect": "buying_intent",
      },
      {
          "id": "buy_5",
          "email": "We have decided to purchase the enterprise plan.",
          "expect": "buying_intent",
      },

     # --- Urgent buying ---
      {
          "id": "urgent_1",
          "email": "We need to finalize this purchase by tomorrow.",
          "expect": "urgency",
      },
      {
          "id": "urgent_2",
          "email": "This needs to be sorted out today, it's time sensitive.",
          "expect": "urgency",
      },
      {
          "id": "urgent_3",
          "email": "We need a response from you as soon as possible.",
          "expect": "urgency",
      },

     # --- Budget discussion ---
      {
          "id": "budget_1",
          "email": "Can you send us pricing and payment terms?",
          "expect": "budget_discussion",
      },
      {
          "id": "budget_2",
          "email": "What does your pricing look like for a team of fifty?",
          "expect": "budget_discussion",
      },
      {
          "id": "budget_3",
          "email": "Could you send over a formal quote for this project?",
          "expect": "budget_discussion",
      },
      {
          "id": "budget_4",
          "email": "We'd like to discuss the billing cycle and invoice schedule.",
          "expect": "budget_discussion",
      },

     # --- Mixed intent + objection ---
      {
          "id": "mixed_1",
          "email": "We want to proceed, but we have concerns about integration.",
          "expect": "unclear_or_off_topic",
      },

     # --- Objection only ---
      {
          "id": "obj_1",
          "email": "We are not sure this solution fits our needs.",
          "expect": "objection",
      },
      {
          "id": "obj_2",
          "email": "We have concerns about how this will integrate with our existing systems.",
          "expect": "objection",
      },
      {
          "id": "obj_3",
          "email": "This solution seems too complex for our team to adopt.",
          "expect": "objection",
      },
      {
          "id": "obj_4",
          "email": "We are doubtful this will scale to our needs.",
          "expect": "objection",
      },

     # --- Explicit rejection ---
      {
          "id": "neg_1",
          "email": "We are not interested and will not proceed.",
          "expect": "negative_intent",
      },
      {
          "id": "neg_2",
          "email": "Please remove us from your mailing list, we are not interested.",
          "expect": "negative_intent",
      },
      {
          "id": "neg_3",
          "email": "We have decided to go with a different vendor.",
          "expect": "negative_intent",
      },

     # --- No current need ---
      {
          "id": "cold_1",
          "email": "We are not evaluating new tools at the moment.",
          "expect": "no_current_need",
      },
      {
          "id": "cold_2",
          "email": "Our current provider is meeting all of our needs right now.",
          "expect": "no_current_need",
      },
      {
          "id": "cold_3",
          "email": "We are not in the market for this kind of tool at this time.",
          "expect": "no_current_need",
      },

     # --- Off topic / unclear ---
      {
          "id": "off_1",
          "email": "The meeting has been rescheduled to Tuesday.",
          "expect": "unclear_or_off_topic",
      },
      {
          "id": "off_2",
          "email": "Happy holidays to you and your team.",
          "expect": "unclear_or_off_topic",
      },
      {
          "id": "off_3",
          "email": "Please find the attached invoice for last month.",
          "expect": "unclear_or_off_topic",
      },
      {
          "id": "edge_1",
          "email": "Sounds good.",
          "expect": "unclear_or_off_topic",
      },
      {
          "id": "edge_2",
          "email": "This looks interesting.",
          "expect": "unclear_or_off_topic",
      },
      {
          "id": "edge_3",
          "email": "Thanks for the update.",
          "expect": "unclear_or_off_topic",
      },
    ]    
    
    results = []
    failures = []

    for case in BENCHMARK_EMAILS:
        output = analyze_email_mnli(case["email"])

        passed = output["top_label"] == case["expect"]

        entry = {
            "id": case["id"],
            "email": case["email"],
            "expected_signal": case["expect"],
            "actual_signal": output["signal_type"],
            "top_label": output["top_label"],
            "engagement_score": output["engagement_score"],
            "labels": output["labels"],
            "pass": passed,
        }

        results.append(entry)

        if not passed:
            failures.append(case["id"])

    return {
        "total_cases": len(BENCHMARK_EMAILS),
        "passed": len(BENCHMARK_EMAILS) - len(failures),
        "failed": len(failures),
        "failure_ids": failures,
        "results": results,
    }