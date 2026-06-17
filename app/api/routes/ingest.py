"""
Salesforce ingestion routes.

POST /ingest/score        — score a single Salesforce lead (real-time, called from Apex)
POST /ingest/score/batch  — score a batch of Salesforce leads in one callout
POST /ingest/leads        — append / replace training data in leads.csv
"""
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas.salesforce import (
    SalesforceBatchScoreRequest,
    SalesforceBatchScoreResponse,
    SalesforceLeadBatchPayload,
    SalesforceLeadPayload,
    SalesforceScoreResult,
)
from app.services import scoring as scoring_svc
from app.services.salesforce_mapper import map_sf_lead

router = APIRouter()
DATA_PATH = Path("data/leads.csv")


@router.post("/score", response_model=SalesforceScoreResult)
def score_salesforce_lead(lead: SalesforceLeadPayload):
    """
    Score a single Salesforce lead.
    Called from the LeadScoreTrigger Apex class on Lead insert/update.
    """
    features = map_sf_lead(lead.model_dump())
    try:
        score = scoring_svc.score_lead(features)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return SalesforceScoreResult(
        salesforce_id=lead.salesforce_id,
        score=score,
        bucket=scoring_svc.score_to_bucket(score),
    )


@router.post("/score/batch", response_model=SalesforceBatchScoreResponse)
def score_salesforce_leads_batch(request: SalesforceBatchScoreRequest):
    """
    Score a batch of Salesforce leads in a single callout.
    Use this from an @future Apex method to stay within governor limits.
    """
    try:
        results = []
        for lead in request.leads:
            features = map_sf_lead(lead.model_dump())
            score = scoring_svc.score_lead(features)
            results.append(
                SalesforceScoreResult(
                    salesforce_id=lead.salesforce_id,
                    score=score,
                    bucket=scoring_svc.score_to_bucket(score),
                )
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return SalesforceBatchScoreResponse(results=results)


@router.post("/leads")
def ingest_leads(payload: SalesforceLeadBatchPayload):
    """
    Append or replace the training dataset with leads exported from Salesforce.

    Steps:
      1. Call this endpoint with historical leads (IsConverted = true/false).
      2. Run `python models/train.py` to retrain the classifier.
      3. Future real-time scores will reflect the updated model.

    mode="append"  — deduplicates on salesforce_id, keeps latest version
    mode="replace" — overwrites leads.csv entirely
    """
    rows = [map_sf_lead(lead.model_dump()) for lead in payload.leads]
    incoming_df = pd.DataFrame(rows)

    if payload.mode == "replace" or not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        incoming_df.to_csv(DATA_PATH, index=False)
        action = "replaced"
        total = len(rows)
    else:
        existing_df = pd.read_csv(DATA_PATH)
        combined = (
            pd.concat([existing_df, incoming_df])
            .drop_duplicates(subset="lead_id", keep="last")
            .reset_index(drop=True)
        )
        combined.to_csv(DATA_PATH, index=False)
        action = "appended"
        total = len(combined)

    return {
        "status": "ok",
        "action": action,
        "ingested": len(rows),
        "total_in_dataset": total,
        "next_step": "Run `python models/train.py` to retrain the classifier.",
    }
