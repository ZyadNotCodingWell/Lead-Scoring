"""
Lead scoring routes.

POST /score/lead     — score a single lead
POST /score/batch    — score a list of leads
POST /score/optimize — assign leads to salespeople via OR-Tools
"""
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas.lead import (
    BatchScoreRequest,
    BatchScoreResponse,
    LeadFeatures,
    LeadScoreResponse,
    OptimizeRequest,
    OptimizeResponse,
)
from app.services import optimizer as optimizer_svc
from app.services import scoring as scoring_svc

router = APIRouter()


@router.post("/lead", response_model=LeadScoreResponse)
def score_single_lead(lead: LeadFeatures):
    try:
        score = scoring_svc.score_lead(lead.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return LeadScoreResponse(score=score, bucket=scoring_svc.score_to_bucket(score))


@router.post("/batch", response_model=BatchScoreResponse)
def score_batch(request: BatchScoreRequest):
    try:
        results = [
            LeadScoreResponse(
                score=(s := scoring_svc.score_lead(lead.model_dump())),
                bucket=scoring_svc.score_to_bucket(s),
            )
            for lead in request.leads
        ]
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return BatchScoreResponse(scores=results)


@router.post("/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest):
    """
    Loads leads.csv, scores every lead, then runs the OR-Tools assignment
    optimizer to distribute leads across the provided sales team.

    Routing rules applied automatically:
      - regular salesmen → Individual Contributor / Manager leads
      - senior salesmen  → Director / VP / C-Suite leads
    """
    data_path = Path("data/leads.csv")
    if not data_path.exists():
        raise HTTPException(
            status_code=404,
            detail="leads.csv not found. Run `python data/simulate.py` first.",
        )

    df = pd.read_csv(data_path)

    try:
        df["score"] = scoring_svc.score_leads_df(df)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    salespeople = [sp.model_dump() for sp in request.salespeople]
    result = optimizer_svc.optimize_leads(
        df=df,
        salespeople=salespeople,
        industry_quotas=request.industry_quotas,
    )
    return OptimizeResponse(**result)
