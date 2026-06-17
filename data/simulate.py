"""
Simulate a synthetic lead dataset using a latent quality variable.

Each lead has a hidden quality score q ~ N(0,1). All features are noisy
functions of q so that inter-feature correlations emerge naturally rather
than being injected post-hoc. The conversion label is a Bernoulli draw
from sigmoid(1.8 * q).
"""
import uuid
import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_LEADS = 1_000
N_COMPANIES = 200

RNG = np.random.default_rng(RANDOM_SEED)

COMPANY_SIZE_BINS = ["Small", "Medium", "Large", "Enterprise"]
SENIORITY_BINS = ["Individual Contributor", "Manager", "Director", "VP", "C-Suite"]
LEAD_SOURCES = ["Web", "Email", "Event", "Referral", "Inbound"]
INDUSTRIES = ["Technology", "Finance", "Healthcare", "Manufacturing", "Retail"]

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Barbara", "David", "Elizabeth", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Lisa", "Daniel", "Nancy",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Dorothy", "Paul", "Kimberly", "Andrew", "Emily", "Kenneth", "Donna",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Adams",
]

COMPANY_POOL = [f"COMP-{i:04d}" for i in range(1, N_COMPANIES + 1)]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate(n: int = N_LEADS) -> pd.DataFrame:
    q = RNG.standard_normal(n)

    # Ordinal features correlated with q
    company_raw = 0.6 * q + RNG.normal(0, 0.8, n)
    company_size = pd.cut(
        company_raw,
        bins=[-np.inf, -0.8, 0.2, 1.0, np.inf],
        labels=COMPANY_SIZE_BINS,
    )

    seniority_raw = 0.4 * q + RNG.normal(0, 0.9, n)
    seniority = pd.cut(
        seniority_raw,
        bins=[-np.inf, -1.0, -0.2, 0.5, 1.2, np.inf],
        labels=SENIORITY_BINS,
    )

    # Continuous features correlated with q
    annual_revenue = np.exp(0.3 * q + 2.0 + RNG.normal(0, 0.7, n)).round(2)
    num_employees = np.exp(0.25 * q + 4.5 + RNG.normal(0, 0.6, n)).astype(int)

    engagement_score_raw = (_sigmoid(0.7 * q + RNG.normal(0, 0.5, n)) * 100).round(2)

    # Count features: Poisson rate driven by q
    email_rate = np.maximum(0.1, 5 * _sigmoid(0.5 * q + RNG.normal(0, 0.5, n)))
    email_opens = RNG.poisson(email_rate).astype(int)

    visit_rate = np.maximum(0.1, 15 * _sigmoid(0.5 * q + RNG.normal(0, 0.5, n)))
    website_visits = RNG.poisson(visit_rate).astype(int)

    form_rate = np.maximum(0.1, 3 * _sigmoid(0.4 * q + RNG.normal(0, 0.6, n)))
    form_submissions = RNG.poisson(form_rate).astype(int)

    # Independent categorical features
    lead_source = RNG.choice(LEAD_SOURCES, size=n)
    industry = RNG.choice(INDUSTRIES, size=n)
    days_since_created = RNG.integers(1, 366, size=n)

    # Identity fields
    first = RNG.choice(FIRST_NAMES, size=n)
    last = RNG.choice(LAST_NAMES, size=n)
    names = [f"{f} {l}" for f, l in zip(first, last)]
    company_ids = RNG.choice(COMPANY_POOL, size=n)

    # Conversion label
    p_convert = _sigmoid(1.8 * q)
    converted = RNG.binomial(1, p_convert).astype(int)

    df = pd.DataFrame(
        {
            "lead_id": [str(uuid.uuid4()) for _ in range(n)],
            "name": names,
            "company_id": company_ids,
            "company_size": company_size,
            "seniority": seniority,
            "annual_revenue": annual_revenue,
            "num_employees": num_employees,
            "engagement_score_raw": engagement_score_raw,
            "email_opens": email_opens,
            "website_visits": website_visits,
            "form_submissions": form_submissions,
            "lead_source": lead_source,
            "industry": industry,
            "days_since_created": days_since_created,
            "converted": converted,
        }
    )

    return df


if __name__ == "__main__":
    out_path = Path(__file__).parent / "leads.csv"
    df = simulate()
    df.to_csv(out_path, index=False)
    total = len(df)
    converted = df["converted"].sum()
    print(f"Saved {total} leads → {out_path}")
    print(f"Conversion rate: {converted}/{total} ({converted/total:.1%})")
