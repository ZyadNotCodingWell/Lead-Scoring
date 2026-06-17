"""
Prospecting optimizer using OR-Tools pywraplp.

Assigns leads to salespeople respecting:
  - Per-salesperson capacity limits
  - Seniority routing: regular salesmen → IC/Manager leads only;
    senior salesmen → Director/VP/C-Suite leads only
  - Per-industry minimum quotas (independently configurable)

Objective: maximise total conversion score across all assignments.
"""
from typing import Dict, List, Optional

import pandas as pd
from ortools.linear_solver import pywraplp

SENIOR_SENIORITIES = {"Director", "VP", "C-Suite"}
JUNIOR_SENIORITIES = {"Individual Contributor", "Manager"}


def optimize_leads(
    df: pd.DataFrame,
    salespeople: List[Dict],
    industry_quotas: Optional[Dict[str, int]] = None,
) -> Dict:
    """
    Parameters
    ----------
    df           : DataFrame with columns [lead_id, score, seniority, industry, ...]
    salespeople  : [{"name": str, "role": "salesman"|"senior", "capacity": int}, ...]
    industry_quotas : {industry_name: min_leads} — default 0 for omitted industries
    """
    industry_quotas = industry_quotas or {}

    if df.empty or not salespeople:
        return _empty_result(salespeople, "no input")

    n = len(df)
    n_sp = len(salespeople)

    senior_idx = [j for j, sp in enumerate(salespeople) if sp["role"] == "senior"]
    junior_idx = [j for j, sp in enumerate(salespeople) if sp["role"] == "salesman"]

    seniorities = df["seniority"].tolist()
    industries = df["industry"].tolist()
    scores = df["score"].tolist()

    def _eligible(seniority: str) -> List[int]:
        return senior_idx if seniority in SENIOR_SENIORITIES else junior_idx

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        return _empty_result(salespeople, "solver unavailable")

    # Sparse binary variables: x[(i,j)] only for feasible (lead, salesperson) pairs
    x: Dict = {}
    for i in range(n):
        for j in _eligible(seniorities[i]):
            x[(i, j)] = solver.BoolVar(f"x_{i}_{j}")

    # Objective: maximise sum of scores
    objective = solver.Objective()
    for (i, j), var in x.items():
        objective.SetCoefficient(var, float(scores[i]))
    objective.SetMaximization()

    # Each lead assigned to at most one salesperson
    for i in range(n):
        lead_vars = [x[(i, j)] for j in _eligible(seniorities[i]) if (i, j) in x]
        if lead_vars:
            solver.Add(solver.Sum(lead_vars) <= 1)

    # Per-salesperson capacity
    for j, sp in enumerate(salespeople):
        sp_vars = [x[(i, j)] for i in range(n) if (i, j) in x]
        if sp_vars:
            solver.Add(solver.Sum(sp_vars) <= int(sp["capacity"]))

    # Per-industry minimum quotas
    for industry, min_q in industry_quotas.items():
        if min_q <= 0:
            continue
        ind_vars = [
            x[(i, j)]
            for i in range(n)
            for j in _eligible(seniorities[i])
            if (i, j) in x and industries[i] == industry
        ]
        if ind_vars:
            solver.Add(solver.Sum(ind_vars) >= min_q)

    status = solver.Solve()

    status_str = {
        pywraplp.Solver.OPTIMAL: "optimal",
        pywraplp.Solver.FEASIBLE: "feasible",
    }.get(status, "infeasible")

    assignments = []
    total_ev = 0.0

    for j, sp in enumerate(salespeople):
        leads = []
        for i in range(n):
            if (i, j) in x and x[(i, j)].solution_value() > 0.5:
                row = df.iloc[i]
                leads.append({
                    "lead_id": str(row["lead_id"]),
                    "name": str(row["name"]) if "name" in row and pd.notna(row["name"]) else None,
                    "company_id": str(row["company_id"]) if "company_id" in row and pd.notna(row["company_id"]) else None,
                    "company_size": str(row["company_size"]),
                    "seniority": str(row["seniority"]),
                    "industry": str(row["industry"]),
                    "score": round(float(row["score"]), 4),
                })
                total_ev += float(row["score"])

        assignments.append({
            "name": sp["name"],
            "role": sp["role"],
            "capacity": int(sp["capacity"]),
            "assigned_count": len(leads),
            "leads": leads,
        })

    return {
        "assignments": assignments,
        "total_leads_assigned": sum(a["assigned_count"] for a in assignments),
        "total_expected_value": round(total_ev, 4),
        "status": status_str,
    }


def _empty_result(salespeople: List[Dict], status: str) -> Dict:
    return {
        "assignments": [
            {"name": sp["name"], "role": sp["role"], "capacity": sp["capacity"],
             "assigned_count": 0, "leads": []}
            for sp in salespeople
        ],
        "total_leads_assigned": 0,
        "total_expected_value": 0.0,
        "status": status,
    }
