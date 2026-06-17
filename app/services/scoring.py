"""
Lead scoring service — loads the trained sklearn Pipeline once at import time
and exposes score_lead() / score_leads_df() for use by route handlers.
"""
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "lead_scorer.pkl"

_CATEGORICAL_FEATURES = ["company_size", "seniority", "lead_source", "industry"]
_NUMERIC_FEATURES = [
    "annual_revenue",
    "num_employees",
    "engagement_score_raw",
    "email_opens",
    "website_visits",
    "form_submissions",
    "days_since_created",
]
_ALL_FEATURES = _NUMERIC_FEATURES + _CATEGORICAL_FEATURES


def _load_pipeline():
    if not _MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {_MODEL_PATH}. Run `python models/train.py` first."
        )
    return joblib.load(_MODEL_PATH)


# Loaded once at first call, then cached in this module-level variable
_pipeline: Optional[object] = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = _load_pipeline()
    return _pipeline


def score_lead(features: dict) -> float:
    """Return conversion probability [0, 1] for a single lead."""
    pipeline = get_pipeline()
    df = pd.DataFrame([features])[_ALL_FEATURES]
    prob = pipeline.predict_proba(df)[0][1]
    return round(float(prob), 4)


def score_leads_df(df: pd.DataFrame) -> pd.Series:
    """Score all rows in a DataFrame that has the required feature columns.

    Returns a Series of probabilities indexed like df.
    """
    pipeline = get_pipeline()
    X = df[_ALL_FEATURES]
    probs = pipeline.predict_proba(X)[:, 1]
    return pd.Series(probs, index=df.index).round(4)


def score_to_bucket(score: float) -> str:
    if score >= 0.75:
        return "hot"
    if score >= 0.45:
        return "warm"
    if score >= 0.20:
        return "low"
    return "cold"


def get_lead_by_id(lead_id: str) -> dict | None:
    """Return the full row dict for a lead, or None if not found."""
    data_path = Path(__file__).parent.parent.parent / "data" / "leads.csv"
    if not data_path.exists():
        return None
    df = pd.read_csv(data_path)
    row = df[df["lead_id"] == lead_id]
    return row.iloc[0].to_dict() if not row.empty else None


def update_lead_score(lead_id: str, new_score: float) -> bool:
    """Write a blended score into the 'score' column of leads.csv."""
    data_path = Path(__file__).parent.parent.parent / "data" / "leads.csv"
    if not data_path.exists():
        return False
    df = pd.read_csv(data_path)
    if "score" not in df.columns:
        df["score"] = pd.NA
    mask = df["lead_id"] == lead_id
    if not mask.any():
        return False
    df.loc[mask, "score"] = round(new_score, 4)
    df.to_csv(data_path, index=False)
    return True
