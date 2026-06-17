#!/usr/bin/env python
"""
Scheduled assignment job — runs the optimizer and saves a timestamped CSV.

Schedule with cron (Linux/macOS):
    0 0,8,16 * * * cd /path/to/Lead_Scoring-main && python jobs/scheduled_assignment.py

Schedule with Windows Task Scheduler:
    Program:   python
    Arguments: jobs/scheduled_assignment.py
    Start in:  C:\\path\\to\\Lead_Scoring-main
    Trigger:   Daily, repeat every 8 hours starting 00:00

The script reads config/optimizer.json for the team roster and quotas,
scores all leads in data/leads.csv, runs the OR-Tools optimizer, and writes
output/assignments_YYYYMMDD_HHMMSS.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime

import pandas as pd

from app.services import optimizer as optimizer_svc
from app.services import scoring as scoring_svc

CONFIG_PATH = Path("config/optimizer.json")
DATA_PATH   = Path("data/leads.csv")
OUTPUT_DIR  = Path("output")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"[ERROR] Config not found at {CONFIG_PATH}")
        print("        Edit config/optimizer.json with your team roster and quotas.")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return json.load(f)


def main():
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Starting assignment run...")

    if not DATA_PATH.exists():
        print(f"[ERROR] No leads data at {DATA_PATH}. Run data ingestion first.")
        sys.exit(1)

    config = load_config()
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} leads.")

    print("  Scoring leads...")
    df["score"] = scoring_svc.score_leads_df(df)

    print("  Running optimizer...")
    result = optimizer_svc.optimize_leads(
        df=df,
        salespeople=config["salespeople"],
        industry_quotas=config.get("industry_quotas", {}),
    )

    rows = []
    for a in result["assignments"]:
        for lead in a["leads"]:
            rows.append({"salesperson": a["name"], "role": a["role"], **lead})

    assignments_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"assignments_{ts}.csv"
    assignments_df.to_csv(out_path, index=False)

    print(f"  Solver status:    {result['status']}")
    print(f"  Leads assigned:   {result['total_leads_assigned']}")
    print(f"  Output:           {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
