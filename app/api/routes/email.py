"""
Email intelligence route.

POST /email/analyze
  - Runs MNLI intent analysis on a prospect reply.
  - If lead_id is supplied: fetches the lead's current model score, blends it
    with the MNLI score (60 % MNLI + 40 % model), and persists the update to
    leads.csv so downstream Salesforce reads see the refreshed score.
"""
from fastapi import APIRouter, HTTPException
from app.schemas.lead import EmailReplyRequest, EmailReplyResponse
from app.email_intelligence.mnli import analyze_email_mnli, run_mnli_benchmark
from app.services import scoring as scoring_svc

MNLI_WEIGHT = 0.6  # 60 % email engagement, 40 % model score

router = APIRouter()


def _bucket_from_score(score: float) -> str:
    if score >= 0.75:
        return "hot"
    if score >= 0.45:
        return "warm"
    if score >= 0.20:
        return "low"
    return "cold"


@router.post("/analyze", response_model=EmailReplyResponse)
def analyze_email(request: EmailReplyRequest):
    mnli_result = analyze_email_mnli(request.email_body)

    previous_score: float | None = None
    blended: float | None = None
    score_updated = False

    if request.lead_id:
        lead_data = scoring_svc.get_lead_by_id(request.lead_id)
        if lead_data is None:
            raise HTTPException(
                status_code=404, detail=f"Lead '{request.lead_id}' not found in leads.csv"
            )
        previous_score = scoring_svc.score_lead(lead_data)
        blended = round(MNLI_WEIGHT * mnli_result["engagement_score"] + (1 - MNLI_WEIGHT) * previous_score, 4)
        score_updated = scoring_svc.update_lead_score(request.lead_id, blended)

    final_score = blended if blended is not None else mnli_result["engagement_score"]

    return EmailReplyResponse(
        mnli_analysis=mnli_result,
        previous_score=previous_score,
        blended_score=blended,
        score_updated=score_updated,
        final_bucket=_bucket_from_score(final_score),
    )


@router.get("/benchmark")
def benchmark_email_analysis():
    results = run_mnli_benchmark()
    return {"benchmark_results": results}