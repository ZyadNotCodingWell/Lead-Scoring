"""
Lead Scoring Pipeline — Streamlit Dashboard

Tab 1 — Score Distribution
Tab 2 — Team Assignment (optimizer with per-salesperson routing + per-industry quotas)
Tab 3 — Email Reply Analyzer (MNLI + 60/40 score blending)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Lead Scoring Pipeline",
    page_icon="Ls",
    layout="wide",
)

DATA_PATH = Path("data/leads.csv")
MODEL_PATH = Path("models/lead_scorer.pkl")

MNLI_WEIGHT = 0.6
INDUSTRIES = ["Technology", "Finance", "Healthcare", "Manufacturing", "Retail"]

DEFAULT_TEAM = pd.DataFrame([
    {"Name": "Alice",          "Role": "salesman", "Capacity": 20},
    {"Name": "Bob",            "Role": "salesman", "Capacity": 20},
    {"Name": "Carol (Senior)", "Role": "senior",   "Capacity": 10},
])

DEFAULT_QUOTAS = pd.DataFrame([
    {"Industry": ind, "Min Leads": 0} for ind in INDUSTRIES
])

# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_scoring_service():
    if not MODEL_PATH.exists():
        return None
    from app.services.scoring import (
        score_leads_df, score_to_bucket, score_lead,
        get_lead_by_id, update_lead_score,
    )
    return score_leads_df, score_to_bucket, score_lead, get_lead_by_id, update_lead_score


def check_dependencies():
    missing = []
    if not DATA_PATH.exists():
        missing.append("`python data/simulate.py`")
    if not MODEL_PATH.exists():
        missing.append("`python models/train.py`")
    return missing


def _bucket(s: float) -> str:
    return "hot" if s >= 0.65 else "warm" if s >= 0.40 else "low" if s >= 0.15 else "cold"


# ── sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Lead Scoring Pipeline")
st.sidebar.markdown("---")
st.sidebar.caption("Email blend rule: **60 % MNLI + 40 % model score**")
st.sidebar.caption(
    "Routing rule: regular salesmen → IC / Manager leads; "
    "senior salesmen → Director / VP / C-Suite leads."
)

# ── preflight check ───────────────────────────────────────────────────────────

missing = check_dependencies()
if missing:
    st.title("Lead Scoring Pipeline")
    st.error("Missing prerequisites. Please run the following commands first:")
    for cmd in missing:
        st.code(cmd, language="bash")
    st.stop()

# ── load data + score ─────────────────────────────────────────────────────────

df_raw = load_data()
svc = load_scoring_service()

if df_raw is None or svc is None:
    st.error("Could not load data or model.")
    st.stop()

score_leads_df, score_to_bucket, score_lead, get_lead_by_id, update_lead_score = svc

with st.spinner("Scoring leads..."):
    df = df_raw.copy()
    df["converted"] = df["converted"].astype(int)
    df["model_score"] = score_leads_df(df)
    if "score" in df.columns:
        df["score"] = df["score"].where(df["score"].notna(), df["model_score"])
    else:
        df["score"] = df["model_score"]
    for lid, s in st.session_state.get("score_overrides", {}).items():
        df.loc[df["lead_id"] == lid, "score"] = s
    df["bucket"] = df["score"].apply(score_to_bucket)

# ── tabs ──────────────────────────────────────────────────────────────────────

st.title("Lead Scoring Pipeline")
tab1, tab2, tab3 = st.tabs(["Score Distribution", "Team Assignment", "Email Reply Analyzer"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Score Distribution
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Leads", len(df))
    col2.metric("Hot", int((df["bucket"] == "hot").sum()))
    col3.metric("Warm", int((df["bucket"] == "warm").sum()))
    col4.metric("Converted (actual)", int(df["converted"].sum()))

    st.markdown("### Score Distribution")
    fig_hist = px.histogram(
        df,
        x="score",
        color="bucket",
        nbins=40,
        title="Conversion Probability Distribution",
        labels={"score": "Model Score", "bucket": "Bucket"},
        color_discrete_map={"hot": "#e53935", "warm": "#fb8c00", "low": "#fdd835", "cold": "#1e88e5"},
        category_orders={"bucket": ["hot", "warm", "low", "cold"]},
        template="plotly_dark",
    )
    fig_hist.update_layout(bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Score by Industry")
        ind_stats = (
            df.groupby("industry")["score"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_score", "count": "n_leads"})
            .sort_values("avg_score", ascending=False)
        )
        fig_bar = px.bar(
            ind_stats, x="industry", y="avg_score",
            color="avg_score", color_continuous_scale="Blues",
            title="Average Score by Industry",
            labels={"avg_score": "Avg Score", "industry": "Industry"},
            template="plotly_dark",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.markdown("### Score vs. Actual Conversion")
        fig_box = px.box(
            df, x="converted", y="score", color="converted",
            title="Model Score by Actual Conversion Label",
            labels={"converted": "Converted (0=No, 1=Yes)", "score": "Model Score"},
            color_discrete_map={0: "#1e88e5", 1: "#e53935"},
            template="plotly_dark",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### Top 20 Leads")
    _top20_cols = [c for c in ["lead_id", "name", "company_id", "company_size", "seniority", "industry", "score", "bucket", "converted"] if c in df.columns]
    top20 = (
        df[_top20_cols]
        .sort_values("score", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    st.dataframe(top20, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Team Assignment
# ──────────────────────────────────────────────────────────────────────────────
with tab2:

    # ── Team configuration ────────────────────────────────────────────────────
    with st.expander("Team Configuration", expanded=True):
        st.caption(
            "Add or remove salespeople. **Role** determines which leads they receive: "
            "`salesman` → Individual Contributor / Manager; "
            "`senior` → Director / VP / C-Suite."
        )
        team_df = st.data_editor(
            DEFAULT_TEAM,
            column_config={
                "Name": st.column_config.TextColumn("Name", required=True),
                "Role": st.column_config.SelectboxColumn(
                    "Role",
                    options=["salesman", "senior"],
                    required=True,
                ),
                "Capacity": st.column_config.NumberColumn(
                    "Capacity (max leads)",
                    min_value=1,
                    max_value=200,
                    step=1,
                    required=True,
                ),
            },
            num_rows="dynamic",
            use_container_width=True,
            key="team_editor",
        )

    # ── Industry quotas ───────────────────────────────────────────────────────
    with st.expander("Industry Quotas (minimum leads per industry)", expanded=False):
        st.caption("Set the minimum number of leads that must be assigned per industry. 0 = no requirement.")
        quotas_df = st.data_editor(
            DEFAULT_QUOTAS,
            column_config={
                "Industry": st.column_config.TextColumn("Industry", disabled=True),
                "Min Leads": st.column_config.NumberColumn(
                    "Min Leads",
                    min_value=0,
                    max_value=100,
                    step=1,
                ),
            },
            use_container_width=True,
            hide_index=True,
            key="quotas_editor",
        )

    run_btn = st.button("Run Assignment Optimizer", use_container_width=True, type="primary")

    # ── Validate team ─────────────────────────────────────────────────────────
    valid_team = team_df.dropna(subset=["Name", "Role", "Capacity"])
    valid_team = valid_team[valid_team["Role"].isin(["salesman", "senior"])]
    valid_team = valid_team[valid_team["Name"].str.strip() != ""]

    if run_btn:
        if valid_team.empty:
            st.error("Add at least one salesperson before running the optimizer.")
        else:
            salespeople = [
                {"name": row["Name"], "role": row["Role"], "capacity": int(row["Capacity"])}
                for _, row in valid_team.iterrows()
            ]
            industry_quotas = {
                row["Industry"]: int(row["Min Leads"])
                for _, row in quotas_df.iterrows()
                if int(row["Min Leads"]) > 0
            }

            with st.spinner("Running OR-Tools optimizer..."):
                from app.services.optimizer import optimize_leads

                result = optimize_leads(
                    df=df,
                    salespeople=salespeople,
                    industry_quotas=industry_quotas,
                )
                st.session_state["opt_result"] = result

    # ── Results ───────────────────────────────────────────────────────────────
    result = st.session_state.get("opt_result")

    if result is None:
        st.info("Configure the team above and click **Run Assignment Optimizer**.")
    elif result["status"] == "infeasible":
        st.error(
            "Optimizer returned infeasible. Industry quotas may exceed the combined "
            "capacity of eligible salespeople. Try lowering quotas or adding more capacity."
        )
    else:
        assignments = result["assignments"]

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Leads Assigned", result["total_leads_assigned"])
        c2.metric("Total Score (EV)", f"{result['total_expected_value']:.2f}")
        c3.metric("Solver Status", result["status"].title())

        # Summary table
        st.markdown("### Team Summary")
        summary_rows = []
        for a in assignments:
            role_label = "Senior" if a["role"] == "senior" else "Regular"
            util = f"{a['assigned_count'] / a['capacity'] * 100:.0f}%" if a["capacity"] else "—"
            summary_rows.append({
                "Salesperson": a["name"],
                "Role": role_label,
                "Capacity": a["capacity"],
                "Assigned": a["assigned_count"],
                "Utilization": util,
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # Bar chart of utilization
        summary_df = pd.DataFrame(summary_rows)
        summary_df["Assigned_num"] = [a["assigned_count"] for a in assignments]
        summary_df["Remaining"] = [a["capacity"] - a["assigned_count"] for a in assignments]

        fig_assign = px.bar(
            summary_df,
            x="Salesperson",
            y=["Assigned_num", "Remaining"],
            title="Capacity Utilization per Salesperson",
            labels={"value": "Leads", "variable": ""},
            color_discrete_map={"Assigned_num": "#1e88e5", "Remaining": "#37474f"},
            barmode="stack",
            template="plotly_dark",
        )
        fig_assign.for_each_trace(
            lambda t: t.update(name="Assigned" if t.name == "Assigned_num" else "Remaining capacity")
        )
        st.plotly_chart(fig_assign, use_container_width=True)

        # Per-salesperson lead tables
        st.markdown("### Assigned Leads by Salesperson")
        for a in assignments:
            role_label = "Senior" if a["role"] == "senior" else "Regular"
            header = f"{a['name']} ({role_label}) — {a['assigned_count']} / {a['capacity']} leads"
            with st.expander(header, expanded=False):
                if a["leads"]:
                    leads_df = pd.DataFrame(a["leads"])
                    leads_df = leads_df.sort_values("score", ascending=False).reset_index(drop=True)
                    st.dataframe(leads_df, use_container_width=True)
                else:
                    st.caption("No leads assigned.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Email Reply Analyzer
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Email Reply Analyzer")
    st.caption(
        "Paste a prospect reply and supply a Lead ID to blend the MNLI engagement "
        "score with the lead's model score (**60 % MNLI + 40 % model**) and persist "
        "the update. Omit the Lead ID for a standalone MNLI analysis."
    )

    col_id, col_email = st.columns([1, 3])
    with col_id:
        lead_id_input = st.text_input("Lead ID (optional)", placeholder="Paste lead_id here…")
        if lead_id_input:
            preview = df[df["lead_id"] == lead_id_input]
            if not preview.empty:
                row = preview.iloc[0]
                name_str = f"{row['name']} · " if "name" in row and pd.notna(row.get("name")) else ""
                cid_str = f" ({row['company_id']})" if "company_id" in row and pd.notna(row.get("company_id")) else ""
                st.success(f"{name_str}{row['company_size']} / {row['seniority']}{cid_str}")
                st.metric("Current Score", f"{row['score']:.3f}")
            else:
                st.warning("Lead ID not found in dataset.")
    with col_email:
        email_body = st.text_area("Prospect Reply", height=160, placeholder="Paste email reply here…")

    analyze_btn = st.button("Analyze Reply", use_container_width=True)

    if analyze_btn and email_body.strip():
        with st.spinner("Running MNLI analysis…"):
            try:
                from app.email_intelligence.mnli import analyze_email_mnli, _normalize_label, SALES_LABELS

                mnli = analyze_email_mnli(email_body)

                previous_score: float | None = None
                blended: float | None = None
                score_updated = False

                if lead_id_input:
                    lead_data = get_lead_by_id(lead_id_input)
                    if lead_data:
                        previous_score = score_lead(lead_data)
                        blended = round(MNLI_WEIGHT * mnli["engagement_score"] + (1 - MNLI_WEIGHT) * previous_score, 4)
                        score_updated = update_lead_score(lead_id_input, blended)
                        st.session_state.setdefault("score_overrides", {})[lead_id_input] = blended
                        load_data.clear()

                final_score = blended if blended is not None else mnli["engagement_score"]

                st.markdown("#### Results")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("MNLI Engagement", f"{min(100, max(0,100*mnli['engagement_score'])):+.2f}", help="-0 … +100")
                m1.caption(f"Signal: **{mnli['signal_type']}**")

                if blended is not None:
                    delta = f"{blended - previous_score:+.3f}" if previous_score is not None else None
                    m3.metric("Previous Score", f"{previous_score:.3f}")
                    m4.metric("Blended Score", f"{blended:.3f}", delta=delta, help="60 % MNLI + 40 % model")
                    if score_updated:
                        st.success(f"Score updated and saved for lead `{lead_id_input}`.")
                    else:
                        st.warning("Score computed but could not be saved.")
                else:
                    m3.metric("Final Bucket", _bucket(final_score).upper())

                st.markdown("#### Intent Label Scores")
                labels_df = pd.DataFrame(
                    {
                        "label": SALES_LABELS,
                        "score": [mnli["labels"].get(_normalize_label(l), 0.0) for l in SALES_LABELS],
                    }
                ).sort_values("score", ascending=False)
                fig_labels = px.bar(
                    labels_df, x="score", y="label", orientation="h",
                    color="score", color_continuous_scale="RdYlGn", range_color=[0, 1],
                    title="MNLI Label Scores",
                    template="plotly_dark",
                )
                st.plotly_chart(fig_labels, use_container_width=True)

            except Exception as exc:
                st.error(f"MNLI analysis failed: {exc}")

    elif analyze_btn:
        st.warning("Please enter an email body.")
