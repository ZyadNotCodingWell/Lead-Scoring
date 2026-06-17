"""
Maps Salesforce Lead fields to the pipeline's internal feature schema.
All mapping logic lives here so Apex callouts stay thin.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional


# ── Company size ──────────────────────────────────────────────────────────────

_COMPANY_SIZE = [(50, "Small"), (200, "Medium"), (1_000, "Large")]


def map_company_size(n_employees: Optional[int]) -> str:
    if n_employees is None:
        return "Small"
    for threshold, label in _COMPANY_SIZE:
        if n_employees < threshold:
            return label
    return "Enterprise"


# ── Seniority ─────────────────────────────────────────────────────────────────

_SENIORITY_RULES = [
    (["ceo", "cto", "cfo", "coo", "chief", "president", "founder", "owner"], "C-Suite"),
    (["vp", "vice president", "vice-president"],                              "VP"),
    (["director"],                                                            "Director"),
    (["manager", "lead", "head", "supervisor"],                               "Manager"),
]


def map_seniority(title: Optional[str]) -> str:
    if not title:
        return "Individual Contributor"
    t = title.lower()
    for keywords, label in _SENIORITY_RULES:
        if any(k in t for k in keywords):
            return label
    return "Individual Contributor"


# ── Industry ──────────────────────────────────────────────────────────────────

_INDUSTRY_MAP = {
    "technology": "Technology",     "software": "Technology",
    "internet": "Technology",       "telecommunications": "Technology",
    "electronics": "Technology",    "semiconductor": "Technology",
    "financial services": "Finance","finance": "Finance",
    "banking": "Finance",           "insurance": "Finance",
    "investment banking": "Finance","capital markets": "Finance",
    "healthcare": "Healthcare",     "pharmaceuticals": "Healthcare",
    "biotechnology": "Healthcare",  "medical devices": "Healthcare",
    "hospital": "Healthcare",       "health": "Healthcare",
    "manufacturing": "Manufacturing","automotive": "Manufacturing",
    "chemicals": "Manufacturing",   "construction": "Manufacturing",
    "aerospace": "Manufacturing",   "industrial": "Manufacturing",
}


def map_industry(sf_industry: Optional[str]) -> str:
    if not sf_industry:
        return "Retail"
    return _INDUSTRY_MAP.get(sf_industry.lower().strip(), "Retail")


# ── Lead source ───────────────────────────────────────────────────────────────

_LEAD_SOURCE_MAP = {
    "web": "Web",               "website": "Web",           "web site": "Web",
    "email": "Email",           "email campaign": "Email",
    "event": "Event",           "trade show": "Event",      "conference": "Event",
    "seminar": "Event",         "webinar": "Event",
    "referral": "Referral",     "partner": "Referral",      "word of mouth": "Referral",
    "employee referral": "Referral",
}


def map_lead_source(sf_source: Optional[str]) -> str:
    if not sf_source:
        return "Inbound"
    return _LEAD_SOURCE_MAP.get(sf_source.lower().strip(), "Inbound")


# ── Date ──────────────────────────────────────────────────────────────────────

def days_since(created_date_str: Optional[str]) -> int:
    if not created_date_str:
        return 30
    try:
        created = datetime.fromisoformat(created_date_str.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - created
        return max(1, delta.days)
    except Exception:
        return 30


# ── Main mapper ───────────────────────────────────────────────────────────────

def map_sf_lead(sf: dict) -> dict:
    """
    Convert a SalesforceLeadPayload dict to the internal feature schema used
    by the sklearn pipeline.  Dollar amounts are converted to $M.
    """
    revenue_usd = sf.get("annual_revenue") or 0.0
    n_emp = sf.get("number_of_employees")

    return {
        "lead_id":               sf.get("salesforce_id") or str(uuid.uuid4()),
        "name":                  sf.get("name"),
        "company_id":            sf.get("company_id"),
        "company_size":          map_company_size(n_emp),
        "seniority":             map_seniority(sf.get("title")),
        "annual_revenue":        round(revenue_usd / 1_000_000, 4),
        "num_employees":         n_emp or 100,
        "engagement_score_raw":  float(sf.get("engagement_score", 50.0)),
        "email_opens":           int(sf.get("email_opens", 0)),
        "website_visits":        int(sf.get("website_visits", 0)),
        "form_submissions":      int(sf.get("form_submissions", 0)),
        "lead_source":           map_lead_source(sf.get("lead_source")),
        "industry":              map_industry(sf.get("industry")),
        "days_since_created":    days_since(sf.get("created_date")),
        "converted":             int(bool(sf.get("is_converted", False))),
    }
