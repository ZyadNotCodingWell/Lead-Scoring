from fastapi import FastAPI

from app.api.routes import health, email

app = FastAPI(
    title="Salesforce Lead Scoring Service",
    description="A FastAPI microservice for Salesforce-based lead scoring.",
    version="0.1.0",
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(email.router, prefix="/email", tags=["Email Intelligence"])
