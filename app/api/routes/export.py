"""
Export routes.

POST /export/trigger      — triggered by Salesforce Scheduled Apex every 8 h;
                            runs the optimizer and saves assignments_{ts}.csv
GET  /export/assignments  — download the latest assignment as a CSV file
"""
import io
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.services import optimizer as optimizer_svc
from app.services import scoring as scoring_svc

router = APIRouter()

CONFIG_PATH = Path("config/optimizer.json")
DATA_PATH = Path("data/leads.csv")
OUTPUT_DIR = Path("output")


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Optimizer config not found at {CONFIG_PATH}. "
                   "Create config/optimizer.json with salespeople and industry_quotas.",
        )
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _build_assignments_df(df: pd.DataFrame, config: dict) -> tuple[dict, pd.DataFrame]:
    df = df.copy()
    df["score"] = scoring_svc.score_leads_df(df)

    result = optimizer_svc.optimize_leads(
        df=df,
        salespeople=config["salespeople"],
        industry_quotas=config.get("industry_quotas", {}),
    )

    rows = []
    for a in result["assignments"]:
        for lead in a["leads"]:
            rows.append({"salesperson": a["name"], "role": a["role"], **lead})

    cols = ["salesperson", "role", "lead_id", "company_size", "seniority", "industry", "score"]
    assignments_df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    return result, assignments_df


@router.post("/trigger")
def trigger_assignment_export():
    """
    Called by the Salesforce Scheduled Apex job every 8 hours.
    Runs the full optimizer pipeline and saves a timestamped CSV to output/.
    """
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="leads.csv not found.")

    df = pd.read_csv(DATA_PATH)
    config = _load_config()

    try:
        result, assignments_df = _build_assignments_df(df, config)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"assignments_{ts}.csv"
    assignments_df.to_csv(csv_path, index=False)

    return {
        "status": result["status"],
        "total_leads_assigned": result["total_leads_assigned"],
        "solver_status": result["status"],
        "saved_to": str(csv_path),
        "timestamp": ts,
    }


@router.get("/assignments")
def download_assignments():
    """Download the current optimal assignment as a CSV file."""
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="leads.csv not found.")

    df = pd.read_csv(DATA_PATH)
    config = _load_config()

    try:
        _, assignments_df = _build_assignments_df(df, config)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    stream = io.StringIO()
    assignments_df.to_csv(stream, index=False)
    stream.seek(0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=assignments_{ts}.csv"},
    )
