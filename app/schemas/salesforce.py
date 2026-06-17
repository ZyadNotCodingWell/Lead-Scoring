from typing import List, Optional
from pydantic import BaseModel, Field


class SalesforceLeadPayload(BaseModel):
    """Salesforce Lead fields as sent from Apex callouts."""
    salesforce_id: str = Field(..., description="18-char Salesforce Lead ID")
    name: Optional[str] = None
    company_id: Optional[str] = None
    number_of_employees: Optional[int] = None
    annual_revenue: Optional[float] = None  # Salesforce stores in full dollars
    title: Optional[str] = None
    industry: Optional[str] = None
    lead_source: Optional[str] = None
    created_date: Optional[str] = None  # ISO 8601 string from Salesforce
    is_converted: bool = False
    # Engagement fields — populated from Pardot / Marketing Cloud custom fields
    email_opens: int = 0
    website_visits: int = 0
    form_submissions: int = 0
    engagement_score: float = 50.0


class SalesforceBatchScoreRequest(BaseModel):
    leads: List[SalesforceLeadPayload]


class SalesforceScoreResult(BaseModel):
    salesforce_id: str
    score: float
    bucket: str


class SalesforceBatchScoreResponse(BaseModel):
    results: List[SalesforceScoreResult]


class SalesforceLeadBatchPayload(BaseModel):
    leads: List[SalesforceLeadPayload]
    mode: str = Field("append", description="append | replace")
