from fastapi import FastAPI
from app.api.routes import health, email, score, ingest, export

app = FastAPI(
    title="Salesforce Lead Scoring Service",
    description=(
        "End-to-end lead scoring pipeline: structured ML model scoring, "
        "MNLI email intent analysis, score blending, and prospecting optimization."
    ),
    version="0.2.0",
)

app.include_router(health.router,  prefix="/health",  tags=["Health"])
app.include_router(score.router,   prefix="/score",   tags=["Lead Scoring"])
app.include_router(email.router,   prefix="/email",   tags=["Email Intelligence"])
app.include_router(ingest.router,  prefix="/ingest",  tags=["Salesforce Ingestion"])
app.include_router(export.router,  prefix="/export",  tags=["Export"])
