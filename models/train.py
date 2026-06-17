"""
Train a GradientBoosting classifier on the simulated leads dataset.

Saves the fitted sklearn Pipeline (preprocessor + model) to
models/lead_scorer.pkl so the API can load it once at startup.

Run:
    python models/train.py
"""
import sys
from pathlib import Path

# Allow running from project root or from this directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent.parent / "data" / "leads.csv"
MODEL_PATH = Path(__file__).parent / "lead_scorer.pkl"

CATEGORICAL_FEATURES = ["company_size", "seniority", "lead_source", "industry"]
NUMERIC_FEATURES = [
    "annual_revenue",
    "num_employees",
    "engagement_score_raw",
    "email_opens",
    "website_visits",
    "form_submissions",
    "days_since_created",
]
TARGET = "converted"
DROP_COLS = ["lead_id", "name", "company_id", TARGET]


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}. Run `python data/simulate.py` first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=DROP_COLS)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
