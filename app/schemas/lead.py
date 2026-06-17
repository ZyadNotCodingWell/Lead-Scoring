from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class LeadFeatures(BaseModel):
    name: Optional[str] = Field(None, description="Full name of the lead contact")
    company_id: Optional[str] = Field(None, description="Company identifier, e.g. COMP-0042")
    company_size: str = Field(..., description="Small | Medium | Large | Enterprise")
    seniority: str = Field(
        ...,
        description="Individual Contributor | Manager | Director | VP | C-Suite",
    )
    annual_revenue: float = Field(..., gt=0, description="Annual revenue in $M")
    num_employees: int = Field(..., gt=0)
    engagement_score_raw: float = Field(..., ge=0, le=100)
    email_opens: int = Field(..., ge=0)
    website_visits: int = Field(..., ge=0)
    form_submissions: int = Field(..., ge=0)
    lead_source: str = Field(..., description="Web | Email | Event | Referral | Inbound")
    industry: str = Field(
        ..., description="Technology | Finance | Healthcare | Manufacturing | Retail"
    )
    days_since_created: int = Field(..., ge=1)


class LeadScoreResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Conversion probability")
    bucket: str = Field(..., description="hot | warm | low | cold")


class BatchScoreRequest(BaseModel):
    leads: List[LeadFeatures]


class BatchScoreResponse(BaseModel):
    scores: List[LeadScoreResponse]


class EmailReplyRequest(BaseModel):
    email_body: str = Field(..., min_length=1)
    lead_id: Optional[str] = Field(
        None,
        description="Lead ID from leads.csv. When provided the route fetches the "
        "lead's model score and blends it with the MNLI score "
        "(60 % MNLI + 40 % model) then persists the update.",
    )


class EmailReplyResponse(BaseModel):
    mnli_analysis: Dict
    previous_score: Optional[float] = Field(None, description="Model score before the reply")
    blended_score: Optional[float] = Field(None, description="60% MNLI + 40% model score")
    score_updated: bool = False
    final_bucket: str


# ── Optimizer schemas ─────────────────────────────────────────────────────────

class Salesperson(BaseModel):
    name: str = Field(..., min_length=1)
    role: str = Field(..., description="salesman | senior")
    capacity: int = Field(..., ge=1, description="Max leads this person can handle")


class AssignedLead(BaseModel):
    lead_id: str
    name: Optional[str] = None
    company_id: Optional[str] = None
    company_size: str
    seniority: str
    industry: str
    score: float


class SalespersonAssignment(BaseModel):
    name: str
    role: str
    capacity: int
    assigned_count: int
    leads: List[AssignedLead]


class OptimizeRequest(BaseModel):
    salespeople: List[Salesperson] = Field(
        ..., description="Team roster with roles and individual capacities"
    )
    industry_quotas: Dict[str, int] = Field(
        default_factory=dict,
        description="Minimum leads per industry, e.g. {'Technology': 5, 'Finance': 3}",
    )


class OptimizeResponse(BaseModel):
    assignments: List[SalespersonAssignment]
    total_leads_assigned: int
    total_expected_value: float
    status: str
