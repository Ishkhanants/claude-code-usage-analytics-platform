"""
Claude Code Analytics Platform — Streamlit Dashboard

Pages:
  🏠 Overview          — KPIs, daily trend, activity heatmap
  📈 Usage Trends      — Time-series, model share, forecasting
  💰 Cost & Tokens     — Cost breakdown, cache efficiency, distributions
  🔧 Tool Analysis     — Tool usage, success rates, practice breakdown
  👥 Teams & Users     — Practice/level/location insights, top engineers
  🤖 ML Insights       — Anomaly detection, clustering, forecasting
  📊 Statistical Deep Dive — ANOVA, correlations, distributions
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import modules
from etl import run_etl
import analytics as an
import ml
import cohort as ch
import validation as val


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Claude Code Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid #3d3d5c;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] { color: #a0a0c0 !important; font-size: 0.82rem; }
    [data-testid="stMetricValue"] { color: #e0e0ff !important; font-size: 1.7rem; font-weight: 700; }
    [data-testid="stMetricDelta"] { font-size: 0.78rem; }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #c0c0e0;
        border-bottom: 2px solid #5555aa; padding-bottom: 6px;
        margin: 20px 0 12px 0;
    }
    .anomaly-badge {
        background: #ff4444; color: white; border-radius: 6px;
        padding: 2px 8px; font-size: 0.75rem; font-weight: 600;
    }
    .insight-box {
        background: linear-gradient(135deg, #1a2a1a, #0a1a0a);
        border: 1px solid #2d5a2d; border-radius: 8px;
        padding: 12px 16px; margin: 8px 0;
        font-size: 0.9rem; color: #90d090;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Loading data…")
def load_kpis():          return an.kpi_summary()

@st.cache_data(ttl=300)
def load_daily():         return an.daily_activity()

@st.cache_data(ttl=300)
def load_heatmap():       return an.hourly_heatmap()

@st.cache_data(ttl=300)
def load_weekly():        return an.weekly_trend()

@st.cache_data(ttl=300)
def load_cost_model():    return an.cost_by_model()

@st.cache_data(ttl=300)
def load_cost_practice(): return an.cost_by_practice()

@st.cache_data(ttl=300)
def load_cost_level():    return an.cost_by_level()

@st.cache_data(ttl=300)
def load_token_dist():    return an.token_distribution()

@st.cache_data(ttl=300)
def load_cache_eff():     return an.cache_efficiency()

@st.cache_data(ttl=300)
def load_tools():         return an.tool_usage_summary()

@st.cache_data(ttl=300)
def load_tools_prac():    return an.tool_usage_by_practice()

@st.cache_data(ttl=300)
def load_top_users():     return an.top_users(30)

@st.cache_data(ttl=300)
def load_practice():      return an.practice_summary()

@st.cache_data(ttl=300)
def load_location():      return an.location_summary()

@st.cache_data(ttl=300)
def load_sessions_dist(): return an.session_depth_distribution()

@st.cache_data(ttl=300)
def load_errors():        return an.error_summary()

@st.cache_data(ttl=300)
def load_error_trend():   return an.error_trend()

@st.cache_data(ttl=300)
def load_engagement():    return an.user_engagement_scores()

@st.cache_data(ttl=600)
def load_forecast_cost(): return ml.forecast_daily_cost(14)

@st.cache_data(ttl=600)
def load_forecast_sess(): return ml.forecast_daily_sessions(14)

@st.cache_data(ttl=600)
def load_anomalies():     return ml.detect_user_anomalies()

@st.cache_data(ttl=600)
def load_daily_anomalies(): return ml.detect_daily_cost_anomalies()

@st.cache_data(ttl=600)
def load_clusters():      return ml.cluster_users()

@st.cache_data(ttl=600)
def load_model_trend():   return ml.model_usage_trend()

@st.cache_data(ttl=600)
def load_anova():         return ml.cost_anova_by_practice()

@st.cache_data(ttl=600)
def load_level_corr():    return ml.level_cost_correlation()

@st.cache_data(ttl=600)
def load_session_stats(): return ml.session_duration_stats()

@st.cache_data(ttl=600)
def load_decompose():        return ml.decompose_daily_cost()

@st.cache_data(ttl=600)
def load_efficiency():       return ml.model_efficiency_index()

@st.cache_data(ttl=600)
def load_activity_profiles(): return ml.user_activity_profiles()

@st.cache_data(ttl=300)
def load_all_emails():       return an.all_user_emails()

@st.cache_data(ttl=600)
def load_retention():        return ch.weekly_retention_cohort()

@st.cache_data(ttl=600)
def load_funnel():           return ch.session_funnel()

@st.cache_data(ttl=600)
def load_gaps():             return ch.inter_session_gaps()

@st.cache_data(ttl=600)
def load_rolling_clusters(): return ch.rolling_sessions_by_cluster()

@st.cache_data(ttl=600)
def load_tool_adoption():    return ch.tool_adoption_over_time()

@st.cache_data(ttl=600)
def load_new_returning():    return ch.new_vs_returning_daily()

@st.cache_data(ttl=600, show_spinner="Running data quality checks…")
def load_validation():       return val.run_all_checks()



# ---------------------------------------------------------------------------
# Plotly theme helper
# ---------------------------------------------------------------------------
THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c0c0e0"),
    margin=dict(l=30, r=20, t=40, b=30),
)
PALETTE = px.colors.qualitative.Vivid


def fig_theme(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(height=height, **THEME)
    fig.update_xaxes(showgrid=True, gridcolor="#2a2a4a", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#2a2a4a", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Ensure DB is loaded
# ---------------------------------------------------------------------------

def ensure_db():
    try:
        from db import get_connection
        conn = get_connection()
        n = conn.execute("SELECT COUNT(*) FROM api_requests").fetchone()[0]
        conn.close()
        if n == 0:
            raise ValueError("empty")
    except Exception:
        with st.spinner("Running ETL pipeline… (~30s)"):
            run_etl(force=False)


ensure_db()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = {
    "🏠 Overview":               "overview",
    "📈 Usage Trends":           "trends",
    "💰 Cost & Tokens":          "cost",
    "🔧 Tool Analysis":          "tools",
    "👥 Teams & Users":          "teams",
    "🤖 ML Insights":            "ml",
    "📊 Statistical Deep Dive":  "stats",
    "🔍 Engineer Drill-Down":    "drilldown",
    "🔄 Cohort & Retention":     "cohort",
    "🛡️ Data Quality":           "quality",
}

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Anthropic_logo.svg/320px-Anthropic_logo.svg.png", width=140)
    st.markdown("## Claude Code Analytics")
    st.markdown("*Internal developer telemetry*")
    st.divider()
    page_label = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    page = PAGES[page_label]

    st.divider()
    kpis = load_kpis()
    st.markdown(f"**📅 Dataset**")
    st.caption(f"{int(kpis['total_sessions']):,} sessions · {int(kpis['total_users']):,} engineers")
    st.caption(f"{int(kpis['total_api_calls']):,} API calls · ${kpis['total_cost_usd']:,.2f} cost")


# ===========================================================================
# PAGE: Overview
# ===========================================================================
if page == "overview":
    st.title("⚡ Claude Code — Usage Overview")
    st.caption("Aggregated telemetry across all engineering practices · last 60 days")

    # --- KPI Row ---
    k = kpis
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Engineers", f"{int(k['total_users'])}")
    c2.metric("Sessions", f"{int(k['total_sessions']):,}")
    c3.metric("API Calls", f"{int(k['total_api_calls']):,}")
    c4.metric("Total Cost", f"${k['total_cost_usd']:,.0f}")
    c5.metric("Avg Session", f"{k['avg_session_min']} min")
    c6.metric("Error Rate", f"{k['error_rate_pct']}%")

    st.markdown("---")

    # --- Daily activity trend ---
    daily = load_daily()
    daily["date"] = pd.to_datetime(daily["date"])

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<p class="section-header">📅 Daily Sessions & Cost</p>', unsafe_allow_html=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=daily["date"], y=daily["sessions"],
            name="Sessions", marker_color="#7c6fff", opacity=0.8,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["cost_usd"],
            name="Cost ($)", line=dict(color="#ff9f43", width=2.5),
            mode="lines+markers", marker=dict(size=4),
        ), secondary_y=True)
        fig.update_yaxes(title_text="Sessions", secondary_y=False)
        fig.update_yaxes(title_text="Cost (USD)", secondary_y=True)
        st_fig = fig_theme(fig, 320)
        st.plotly_chart(st_fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">📊 Practice Cost Split</p>', unsafe_allow_html=True)
        cp = load_cost_practice()
        fig2 = px.pie(cp, values="total_cost", names="practice",
                      color_discrete_sequence=PALETTE,
                      hole=0.45)
        fig2.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(fig_theme(fig2, 320), use_container_width=True)

    # --- Heatmap ---
    st.markdown('<p class="section-header">🗓️ Activity Heatmap — Sessions by Day & Hour</p>', unsafe_allow_html=True)
    hm = load_heatmap()
    pivot = hm.pivot_table(index="dow", columns="hour", values="sessions", fill_value=0)
    pivot.index = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    fig3 = px.imshow(
        pivot, labels=dict(x="Hour of Day", y="", color="Sessions"),
        color_continuous_scale="Viridis", aspect="auto",
        text_auto=True,
    )
    fig3.update_xaxes(tickvals=list(range(0, 24, 2)), ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)])
    st.plotly_chart(fig_theme(fig3, 280), use_container_width=True)

    # --- Active users & tokens row ---
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<p class="section-header">👥 Daily Active Engineers</p>', unsafe_allow_html=True)
        fig4 = px.area(daily, x="date", y="active_users",
                       color_discrete_sequence=["#26de81"])
        fig4.update_traces(fill="tozeroy", fillcolor="rgba(38,222,129,0.15)")
        st.plotly_chart(fig_theme(fig4, 260), use_container_width=True)

    with col4:
        st.markdown('<p class="section-header">🔢 Daily Token Consumption</p>', unsafe_allow_html=True)
        fig5 = px.bar(daily, x="date", y="tokens",
                      color_discrete_sequence=["#fd9644"])
        st.plotly_chart(fig_theme(fig5, 260), use_container_width=True)


# ===========================================================================
# PAGE: Usage Trends
# ===========================================================================
elif page == "trends":
    st.title("📈 Usage Trends & Model Adoption")

    # Weekly trend
    st.markdown('<p class="section-header">📆 Weekly Sessions & Active Engineers</p>', unsafe_allow_html=True)
    weekly = load_weekly()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=weekly["week"], y=weekly["sessions"],
                         name="Sessions", marker_color="#7c6fff", opacity=0.75), secondary_y=False)
    fig.add_trace(go.Scatter(x=weekly["week"], y=weekly["active_users"],
                             name="Active Engineers", line=dict(color="#26de81", width=2.5),
                             mode="lines+markers"), secondary_y=True)
    fig.update_yaxes(title_text="Sessions", secondary_y=False)
    fig.update_yaxes(title_text="Engineers", secondary_y=True)
    st.plotly_chart(fig_theme(fig, 340), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">🤖 Model Call Distribution</p>', unsafe_allow_html=True)
        cm = load_cost_model()
        fig2 = px.bar(cm, x="model", y="calls", color="model",
                      color_discrete_sequence=PALETTE, text="calls")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False, xaxis_tickangle=-25)
        st.plotly_chart(fig_theme(fig2, 320), use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">📊 Weekly Model Share (%)</p>', unsafe_allow_html=True)
        mt = load_model_trend()
        fig3 = px.area(mt, x="week", y="share_pct", color="model",
                       color_discrete_sequence=PALETTE, groupnorm="percent")
        fig3.update_layout(yaxis_title="Share %", xaxis_tickangle=-35)
        st.plotly_chart(fig_theme(fig3, 320), use_container_width=True)

    # Avg cost per session trend
    st.markdown('<p class="section-header">💵 Weekly Average Cost Per Session</p>', unsafe_allow_html=True)
    fig4 = px.line(weekly, x="week", y="avg_cost_per_session",
                   markers=True, color_discrete_sequence=["#ff9f43"], line_shape="spline")
    fig4.add_hrule(y=weekly["avg_cost_per_session"].mean(), line_dash="dash",
                   line_color="#aaaaaa", annotation_text="Period avg")
    st.plotly_chart(fig_theme(fig4, 280), use_container_width=True)

    # Forecasting section (preview)
    st.markdown('<p class="section-header">🔮 14-day Session Forecast (preview)</p>', unsafe_allow_html=True)
    fc = load_forecast_sess()
    fig5 = go.Figure()
    hist = fc[~fc["is_forecast"]]
    fut = fc[fc["is_forecast"]]
    fig5.add_trace(go.Scatter(x=hist["date"], y=hist["sessions"],
                              name="Actual", line=dict(color="#7c6fff", width=2)))
    fig5.add_trace(go.Scatter(x=hist["date"], y=hist["predicted"],
                              name="Model fit", line=dict(color="#aaaaff", dash="dash")))
    fig5.add_trace(go.Scatter(x=fut["date"], y=fut["predicted"],
                              name="Forecast", line=dict(color="#ff9f43", width=2.5)))
    fig5.add_trace(go.Scatter(
        x=pd.concat([fut["date"], fut["date"].iloc[::-1]]),
        y=pd.concat([fut["upper_95"], fut["lower_95"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(255,159,67,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI",
    ))
    st.caption("Full ML forecasting with anomaly overlay is in 🤖 ML Insights")
    st.plotly_chart(fig_theme(fig5, 320), use_container_width=True)


# ===========================================================================
# PAGE: Cost & Tokens
# ===========================================================================
elif page == "cost":
    st.title("💰 Cost & Token Analytics")

    # Model cost breakdown
    st.markdown('<p class="section-header">💸 Cost Breakdown by Model</p>', unsafe_allow_html=True)
    cm = load_cost_model()
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(cm, x="model", y="total_cost", color="model",
                     color_discrete_sequence=PALETTE, text_auto=".2f",
                     labels={"total_cost": "Total Cost ($)"})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig_theme(fig, 320), use_container_width=True)
    with col2:
        fig2 = px.bar(cm, x="model", y=["input_tokens", "output_tokens", "cache_read", "cache_creation"],
                      barmode="stack", color_discrete_sequence=PALETTE,
                      labels={"value": "Tokens", "variable": "Type"})
        fig2.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig_theme(fig2, 320), use_container_width=True)

    # Cost by practice & level
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<p class="section-header">🏢 Cost by Practice</p>', unsafe_allow_html=True)
        cp = load_cost_practice()
        fig3 = px.bar(cp, x="total_cost", y="practice", orientation="h",
                      color="total_cost", color_continuous_scale="Plasma", text_auto=".0f")
        fig3.update_traces(textposition="outside")
        st.plotly_chart(fig_theme(fig3, 320), use_container_width=True)

    with col4:
        st.markdown('<p class="section-header">🎓 Cost Per Engineer by Level</p>', unsafe_allow_html=True)
        cl = load_cost_level().sort_values("level_num")
        fig4 = px.bar(cl, x="level", y="cost_per_engineer",
                      color="cost_per_engineer", color_continuous_scale="Viridis",
                      text_auto=".2f", labels={"cost_per_engineer": "Cost / Engineer ($)"})
        st.plotly_chart(fig_theme(fig4, 320), use_container_width=True)

    # Token percentiles
    st.markdown('<p class="section-header">📐 Token Distribution Percentiles</p>', unsafe_allow_html=True)
    td = load_token_dist()
    st.dataframe(
        td.style.format("{:.2f}").background_gradient(cmap="YlOrRd", axis=1),
        use_container_width=True,
    )

    # Cache efficiency
    st.markdown('<p class="section-header">⚡ Cache Efficiency by Model</p>', unsafe_allow_html=True)
    ce = load_cache_eff()
    col5, col6 = st.columns([2, 1])
    with col5:
        fig5 = px.bar(ce, x="model", y=["cache_hits", "fresh_tokens", "cache_writes"],
                      barmode="group", color_discrete_sequence=PALETTE,
                      labels={"value": "Tokens", "variable": "Token Type"})
        fig5.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig_theme(fig5, 300), use_container_width=True)
    with col6:
        fig6 = px.bar(ce, x="model", y="cache_hit_rate_pct", text_auto=".1f",
                      color="cache_hit_rate_pct", color_continuous_scale="Greens",
                      labels={"cache_hit_rate_pct": "Hit Rate %"})
        fig6.update_traces(textposition="outside")
        fig6.update_layout(yaxis_range=[0, 105])
        st.plotly_chart(fig_theme(fig6, 300), use_container_width=True)

    # Model duration vs cost scatter
    st.markdown('<p class="section-header">⏱️ Avg Duration vs Avg Cost by Model</p>', unsafe_allow_html=True)
    fig7 = px.scatter(cm, x="avg_duration_s", y="avg_cost", size="calls",
                      color="model", text="model", size_max=50,
                      color_discrete_sequence=PALETTE,
                      labels={"avg_duration_s": "Avg Duration (s)", "avg_cost": "Avg Cost ($)"})
    fig7.update_traces(textposition="top center")
    st.plotly_chart(fig_theme(fig7, 380), use_container_width=True)


# ===========================================================================
# PAGE: Tool Analysis
# ===========================================================================
elif page == "tools":
    st.title("🔧 Tool Usage Analysis")

    tools = load_tools()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">🛠️ Tool Call Volume</p>', unsafe_allow_html=True)
        fig = px.bar(tools.sort_values("decisions", ascending=True),
                     x="decisions", y="tool_name", orientation="h",
                     color="decisions", color_continuous_scale="Turbo",
                     text_auto=",d")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig_theme(fig, 420), use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">✅ Tool Success Rate</p>', unsafe_allow_html=True)
        fig2 = px.bar(tools.sort_values("success_rate_pct", ascending=True),
                      x="success_rate_pct", y="tool_name", orientation="h",
                      color="success_rate_pct", color_continuous_scale="RdYlGn",
                      range_color=[88, 100], text_auto=".1f")
        fig2.update_traces(textposition="outside")
        fig2.update_xaxes(range=[85, 101])
        st.plotly_chart(fig_theme(fig2, 420), use_container_width=True)

    # Avg duration
    st.markdown('<p class="section-header">⏱️ Average Tool Execution Duration</p>', unsafe_allow_html=True)
    dur = tools[tools["avg_duration_s"].notna()].sort_values("avg_duration_s", ascending=False)
    fig3 = px.bar(dur, x="tool_name", y="avg_duration_s",
                  color="avg_duration_s", color_continuous_scale="Reds",
                  text_auto=".2f", labels={"avg_duration_s": "Avg Duration (s)"})
    fig3.update_traces(textposition="outside")
    st.plotly_chart(fig_theme(fig3, 300), use_container_width=True)

    # Heatmap: tools by practice
    st.markdown('<p class="section-header">🏢 Tool Usage by Practice (heatmap)</p>', unsafe_allow_html=True)
    tp = load_tools_prac()
    pivot = tp.pivot_table(index="practice", columns="tool_name", values="uses", fill_value=0)
    fig4 = px.imshow(pivot, color_continuous_scale="Blues", text_auto=True, aspect="auto")
    st.plotly_chart(fig_theme(fig4, 360), use_container_width=True)

    # Decision source donut
    col3, col4 = st.columns([1, 2])
    with col3:
        st.markdown('<p class="section-header">🔐 Decision Sources</p>', unsafe_allow_html=True)
        ds = an.tool_decision_sources()
        fig5 = px.pie(ds, values="count", names="source",
                      color_discrete_sequence=PALETTE, hole=0.5)
        fig5.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_theme(fig5, 300), use_container_width=True)

    with col4:
        st.markdown('<p class="section-header">📋 Tool Detail Table</p>', unsafe_allow_html=True)
        st.dataframe(
            tools[["tool_name", "decisions", "accepted", "rejected",
                   "success_rate_pct", "avg_duration_s"]]
            .sort_values("decisions", ascending=False)
            .style.format({
                "success_rate_pct": "{:.1f}%",
                "avg_duration_s": "{:.2f}s",
            })
            .background_gradient(subset=["decisions"], cmap="Blues")
            .background_gradient(subset=["success_rate_pct"], cmap="RdYlGn"),
            use_container_width=True, height=260,
        )


# ===========================================================================
# PAGE: Teams & Users
# ===========================================================================
elif page == "teams":
    st.title("👥 Teams & Engineer Insights")

    # Practice overview table
    st.markdown('<p class="section-header">🏢 Practice Summary</p>', unsafe_allow_html=True)
    ps = load_practice()
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(ps, x="practice", y="total_cost",
                     color="practice", text_auto=".0f",
                     color_discrete_sequence=PALETTE,
                     labels={"total_cost": "Total Cost ($)"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig_theme(fig, 300), use_container_width=True)
    with col2:
        fig2 = px.scatter(ps, x="avg_session_min", y="avg_turns_per_session",
                          size="total_cost", color="practice",
                          size_max=60, text="practice",
                          color_discrete_sequence=PALETTE,
                          labels={"avg_session_min": "Avg Session (min)",
                                  "avg_turns_per_session": "Avg Turns / Session"})
        fig2.update_traces(textposition="top center")
        st.plotly_chart(fig_theme(fig2, 300), use_container_width=True)

    st.dataframe(
        ps.style.format({
            "total_cost": "${:,.2f}", "cost_per_engineer": "${:,.2f}",
            "avg_session_min": "{:.1f} min", "avg_turns_per_session": "{:.1f}",
        }).background_gradient(subset=["total_cost"], cmap="Reds"),
        use_container_width=True,
    )

    # Level breakdown
    st.markdown('<p class="section-header">🎓 Cost & Volume by Seniority Level</p>', unsafe_allow_html=True)
    cl = load_cost_level().sort_values("level_num")
    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.bar(cl, x="level", y="api_calls", color="practice",
                      color_discrete_sequence=PALETTE,
                      labels={"api_calls": "API Calls"})
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig_theme(fig3, 280), use_container_width=True)
    with col4:
        fig4 = px.bar(cl, x="level", y="cost_per_engineer", text_auto=".2f",
                      color="cost_per_engineer", color_continuous_scale="Plasma")
        st.plotly_chart(fig_theme(fig4, 280), use_container_width=True)

    # Location
    st.markdown('<p class="section-header">🌍 Usage by Location</p>', unsafe_allow_html=True)
    loc = load_location()
    col5, col6 = st.columns([3, 2])
    with col5:
        fig5 = px.bar(loc, x="location", y="total_cost",
                      color="location", text_auto=".0f",
                      color_discrete_sequence=PALETTE)
        fig5.update_layout(showlegend=False)
        st.plotly_chart(fig_theme(fig5, 280), use_container_width=True)
    with col6:
        fig6 = px.pie(loc, values="engineers", names="location",
                      color_discrete_sequence=PALETTE, hole=0.4)
        st.plotly_chart(fig_theme(fig6, 280), use_container_width=True)

    # Top users table
    st.markdown('<p class="section-header">🏆 Top Engineers by Cost</p>', unsafe_allow_html=True)
    tu = load_top_users()
    top_n = st.slider("Show top N engineers", 5, 30, 15)
    st.dataframe(
        tu.head(top_n)[["full_name", "practice", "level", "location",
                         "sessions", "api_calls", "total_cost", "avg_turns", "avg_session_min"]]
        .style.format({
            "total_cost": "${:,.2f}",
            "avg_turns": "{:.1f}",
            "avg_session_min": "{:.1f}",
        })
        .background_gradient(subset=["total_cost"], cmap="YlOrRd"),
        use_container_width=True,
    )

    # Session depth distribution
    st.markdown('<p class="section-header">📐 Session Complexity Distribution</p>', unsafe_allow_html=True)
    sd = load_sessions_dist()
    col7, col8 = st.columns(2)
    with col7:
        fig7 = px.histogram(sd, x="num_turns", nbins=40,
                            color_discrete_sequence=["#7c6fff"],
                            labels={"num_turns": "Turns per Session"},
                            marginal="box")
        st.plotly_chart(fig_theme(fig7, 280), use_container_width=True)
    with col8:
        fig8 = px.histogram(sd, x="duration_min", nbins=40,
                            color_discrete_sequence=["#26de81"],
                            labels={"duration_min": "Duration (min)"},
                            marginal="box")
        st.plotly_chart(fig_theme(fig8, 280), use_container_width=True)


# ===========================================================================
# PAGE: ML Insights
# ===========================================================================
elif page == "ml":
    st.title("🤖 Predictive Analytics & ML Insights")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔮 Forecasting", "🚨 Anomaly Detection", "🧩 User Clustering", "📉 Error Patterns"]
    )

    # ---- Forecasting ----
    with tab1:
        st.markdown('<p class="section-header">📈 14-Day Cost Forecast (Polynomial Regression)</p>', unsafe_allow_html=True)
        fc = load_forecast_cost()
        da = load_daily_anomalies()

        fig = go.Figure()
        hist = fc[~fc["is_forecast"]]
        fut  = fc[fc["is_forecast"]]

        # Anomaly highlights
        anom = da[da["is_anomaly"]]
        fig.add_trace(go.Scatter(
            x=anom["date"], y=anom["cost_usd"],
            mode="markers", marker=dict(color="red", size=12, symbol="x"),
            name="Cost Anomalies",
        ))
        fig.add_trace(go.Bar(
            x=hist["date"], y=hist["cost_usd"],
            name="Actual Daily Cost", marker_color="#7c6fff", opacity=0.6,
        ))
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["predicted"],
            name="Polynomial Fit", line=dict(color="#aaaaff", dash="dash", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=fut["date"], y=fut["predicted"],
            name="14-day Forecast", line=dict(color="#ff9f43", width=3),
            mode="lines+markers",
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([fut["date"], fut["date"].iloc[::-1]]),
            y=pd.concat([fut["upper_95"], fut["lower_95"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(255,159,67,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI",
        ))
        st.plotly_chart(fig_theme(fig, 400), use_container_width=True)

        st.markdown('<p class="section-header">🤝 14-Day Session Volume Forecast</p>', unsafe_allow_html=True)
        fcs = load_forecast_sess()
        fig2 = go.Figure()
        h2 = fcs[~fcs["is_forecast"]]
        f2 = fcs[fcs["is_forecast"]]
        fig2.add_trace(go.Bar(x=h2["date"], y=h2["sessions"],
                              name="Actual Sessions", marker_color="#26de81", opacity=0.6))
        fig2.add_trace(go.Scatter(x=h2["date"], y=h2["predicted"],
                                  name="Fit", line=dict(color="#90ff90", dash="dash")))
        fig2.add_trace(go.Scatter(x=f2["date"], y=f2["predicted"],
                                  name="Forecast", line=dict(color="#ff9f43", width=3),
                                  mode="lines+markers"))
        fig2.add_trace(go.Scatter(
            x=pd.concat([f2["date"], f2["date"].iloc[::-1]]),
            y=pd.concat([f2["upper_95"], f2["lower_95"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(255,159,67,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI",
        ))
        st.plotly_chart(fig_theme(fig2, 360), use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        💡 <b>Methodology:</b> Degree-2 polynomial regression fitted on the 60-day historical window.
        Confidence intervals are ±1.96σ of residuals. Anomalies are flagged via 7-day rolling Z-score > 2.5.
        </div>
        """, unsafe_allow_html=True)

    # ---- Anomaly Detection ----
    with tab2:
        st.markdown('<p class="section-header">🚨 Per-User Anomaly Detection (Isolation Forest)</p>', unsafe_allow_html=True)
        anom_df = load_anomalies()

        n_anom = anom_df["is_anomaly"].sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Engineers", len(anom_df))
        col2.metric("Flagged Anomalies", int(n_anom), delta=f"{n_anom/len(anom_df)*100:.1f}%")
        col3.metric("Contamination Rate", "10% (configured)")

        # Scatter: sessions vs cost, colored by anomaly
        fig3 = px.scatter(
            anom_df, x="sessions", y="total_cost",
            color="is_anomaly", symbol="practice",
            size="api_calls", size_max=40,
            color_discrete_map={True: "#ff4444", False: "#7c6fff"},
            hover_data=["full_name", "practice", "level", "anomaly_score"],
            labels={"is_anomaly": "Anomalous", "total_cost": "Total Cost ($)"},
        )
        st.plotly_chart(fig_theme(fig3, 420), use_container_width=True)

        # Daily cost anomalies
        st.markdown('<p class="section-header">📅 Daily Cost Anomaly Detection (Z-Score)</p>', unsafe_allow_html=True)
        da2 = load_daily_anomalies()
        da2["date"] = pd.to_datetime(da2["date"])
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=da2["date"], y=da2["cost_usd"],
                              marker_color=np.where(da2["is_anomaly"], "#ff4444", "#7c6fff"),
                              name="Daily Cost"))
        fig4.add_trace(go.Scatter(x=da2["date"], y=da2["rolling_mean"],
                                  name="7-day Rolling Mean", line=dict(color="#ffff66", dash="dot")))
        n_day_anom = da2["is_anomaly"].sum()
        st.plotly_chart(fig_theme(fig4, 320), use_container_width=True)
        st.caption(f"🔴 {n_day_anom} anomalous days detected (|Z| > 2.5 from 7-day rolling mean)")

        # Anomalous users table
        st.markdown('<p class="section-header">🕵️ Anomalous Engineers Detail</p>', unsafe_allow_html=True)
        anom_users = anom_df[anom_df["is_anomaly"]].sort_values("anomaly_score")
        st.dataframe(
            anom_users[["full_name", "practice", "level", "sessions",
                         "api_calls", "total_cost", "avg_turns", "anomaly_score"]]
            .style.format({"total_cost": "${:,.2f}", "anomaly_score": "{:.4f}",
                           "avg_turns": "{:.1f}"})
            .background_gradient(subset=["anomaly_score"], cmap="Reds_r"),
            use_container_width=True,
        )

    # ---- Clustering ----
    with tab3:
        st.markdown('<p class="section-header">🧩 Engineer Behavioural Segments (K-Means, k=4)</p>', unsafe_allow_html=True)
        cl_df = load_clusters()

        cluster_counts = cl_df["cluster_label"].value_counts()
        cols = st.columns(4)
        for i, (label, count) in enumerate(cluster_counts.items()):
            cols[i].metric(label, count)

        col1, col2 = st.columns(2)
        with col1:
            fig5 = px.scatter(
                cl_df, x="sessions", y="total_cost",
                color="cluster_label", size="avg_turns",
                size_max=30, hover_data=["user_email", "practice", "level"],
                color_discrete_sequence=PALETTE,
                labels={"cluster_label": "Segment"},
            )
            st.plotly_chart(fig_theme(fig5, 380), use_container_width=True)
        with col2:
            fig6 = px.box(
                cl_df, x="cluster_label", y="total_cost",
                color="cluster_label", color_discrete_sequence=PALETTE,
                points="all", labels={"cluster_label": "Segment", "total_cost": "Total Cost ($)"},
            )
            st.plotly_chart(fig_theme(fig6, 380), use_container_width=True)

        # Segment profile table
        profile = cl_df.groupby("cluster_label").agg(
            engineers=("user_email", "count"),
            avg_sessions=("sessions", "mean"),
            avg_cost=("total_cost", "mean"),
            avg_turns=("avg_turns", "mean"),
            avg_tools=("distinct_tools", "mean"),
            avg_score=("engagement_score", "mean"),
        ).round(2).reset_index()
        st.dataframe(
            profile.style.format({
                "avg_cost": "${:.2f}", "avg_score": "{:.1f}",
            }).background_gradient(subset=["avg_cost"], cmap="YlOrRd"),
            use_container_width=True,
        )

        st.markdown("""
        <div class="insight-box">
        💡 <b>Segmentation:</b> KMeans (k=4) on StandardScaler-normalised features: 
        sessions, avg_turns, total_cost, distinct_tools. 
        Cluster labels assigned by descending mean cost.
        </div>
        """, unsafe_allow_html=True)

    # ---- Error Patterns ----
    with tab4:
        st.markdown('<p class="section-header">🛑 API Error Summary</p>', unsafe_allow_html=True)
        err = load_errors()
        fig7 = px.bar(err, x="count", y="error_msg", orientation="h",
                      color="status_code", text="pct",
                      color_discrete_sequence=PALETTE,
                      labels={"count": "Error Count", "error_msg": ""})
        fig7.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig_theme(fig7, 320), use_container_width=True)

        et = load_error_trend()
        et["date"] = pd.to_datetime(et["date"])
        fig8 = px.bar(et, x="date", y="errors", color="status_code",
                      barmode="stack", color_discrete_sequence=PALETTE,
                      labels={"errors": "Error Count"})
        st.plotly_chart(fig_theme(fig8, 300), use_container_width=True)


# ===========================================================================
# PAGE: Statistical Deep Dive
# ===========================================================================
elif page == "stats":
    st.title("📊 Statistical Deep Dive")

    # ANOVA
    st.markdown('<p class="section-header">🧪 ANOVA: Cost per Session across Practices</p>', unsafe_allow_html=True)
    anova = load_anova()
    col1, col2, col3 = st.columns(3)
    col1.metric("F-Statistic", f"{anova['f_statistic']:.4f}")
    col2.metric("p-value", f"{anova['p_value']:.4f}")
    col3.metric("Significant (α=0.05)", "Yes ✅" if anova["significant"] else "No ❌")

    if not anova["significant"]:
        st.info("ℹ️ No statistically significant difference in cost across practices — usage is relatively uniform.")
    else:
        st.success("✅ Practices differ significantly in cost — see breakdown below.")

    # Spearman correlation
    st.markdown('<p class="section-header">📈 Spearman Correlation: Seniority vs Cost</p>', unsafe_allow_html=True)
    corr = load_level_corr()
    col4, col5, col6 = st.columns(3)
    col4.metric("Spearman ρ", f"{corr['spearman_rho']:.4f}")
    col5.metric("p-value", f"{corr['p_value']:.4f}")
    col6.metric("Direction", corr["direction"].capitalize())

    if corr["significant"]:
        st.success(f"✅ Significant {corr['direction']} correlation between level and cost per session.")
    else:
        st.info("ℹ️ No significant correlation found between seniority and cost per session.")

    # Cost box by practice
    st.markdown('<p class="section-header">📦 Cost Distribution by Practice (Box Plot)</p>', unsafe_allow_html=True)
    sess_costs = an._q("""
        SELECT e.practice, s.total_cost_usd
        FROM sessions s JOIN employees e ON e.email = s.user_email
        WHERE s.total_cost_usd > 0
    """)
    fig = px.box(sess_costs, x="practice", y="total_cost_usd",
                 color="practice", color_discrete_sequence=PALETTE, points="outliers",
                 labels={"total_cost_usd": "Cost per Session ($)", "practice": ""})
    st.plotly_chart(fig_theme(fig, 380), use_container_width=True)

    # Session duration descriptive stats
    st.markdown('<p class="section-header">⏱️ Session Duration Stats by Practice</p>', unsafe_allow_html=True)
    ss = load_session_stats()
    st.dataframe(
        ss.style.format({
            "mean": "{:.2f}", "median": "{:.2f}", "std": "{:.2f}",
            "p10": "{:.2f}", "p90": "{:.2f}",
            "skewness": "{:.3f}", "kurtosis": "{:.3f}",
        })
        .background_gradient(subset=["mean"], cmap="Blues"),
        use_container_width=True,
    )

    # Prompt length violin
    st.markdown('<p class="section-header">📝 Prompt Length Distribution by Practice</p>', unsafe_allow_html=True)
    prompts = an._q("""
        SELECT e.practice, p.prompt_length
        FROM user_prompts p JOIN employees e ON e.email = p.user_email
        WHERE p.prompt_length BETWEEN 1 AND 10000
    """)
    fig2 = px.violin(prompts, x="practice", y="prompt_length",
                     color="practice", color_discrete_sequence=PALETTE,
                     box=True, points=False,
                     labels={"prompt_length": "Prompt Length (chars)", "practice": ""})
    st.plotly_chart(fig_theme(fig2, 380), use_container_width=True)

    # Engagement score distribution
    st.markdown('<p class="section-header">🎯 Engineer Engagement Score Distribution</p>', unsafe_allow_html=True)
    eng = load_engagement()
    col7, col8 = st.columns(2)
    with col7:
        fig3 = px.histogram(eng, x="engagement_score", nbins=25,
                            color="practice", color_discrete_sequence=PALETTE,
                            labels={"engagement_score": "Engagement Score"})
        st.plotly_chart(fig_theme(fig3, 300), use_container_width=True)
    with col8:
        fig4 = px.scatter(eng, x="sessions", y="engagement_score",
                          color="practice", size="total_cost", size_max=25,
                          color_discrete_sequence=PALETTE,
                          hover_data=["user_email", "level"],
                          labels={"engagement_score": "Score", "sessions": "Sessions"})
        st.plotly_chart(fig_theme(fig4, 300), use_container_width=True)

    # Top engaged engineers
    st.markdown('<p class="section-header">🏅 Top Engaged Engineers</p>', unsafe_allow_html=True)
    st.dataframe(
        eng.head(15)[["user_email", "practice", "level", "sessions",
                       "avg_turns", "distinct_tools", "total_cost", "engagement_score"]]
        .style.format({
            "total_cost": "${:.2f}", "avg_turns": "{:.1f}", "engagement_score": "{:.1f}"
        })
        .background_gradient(subset=["engagement_score"], cmap="YlOrRd"),
        use_container_width=True,
    )


# ===========================================================================
# PAGE: Engineer Drill-Down
# ===========================================================================
elif page == "drilldown":
    st.title("🔍 Engineer Drill-Down")
    st.caption("Deep-dive profile for any individual engineer")

    emails = load_all_emails()
    selected = st.selectbox("Select engineer", emails,
                            format_func=lambda e: e)

    if selected:
        profile = an.user_drilldown(selected)
        if not profile:
            st.error("No data found for this engineer.")
        else:
            emp = profile["employee"]
            sessions_df  = profile["sessions"]
            model_df     = profile["model_mix"]
            tools_df     = profile["tools"]
            daily_df     = profile["daily"]
            errors_df    = profile["errors"]

            # Header card
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Practice",  emp.get("practice", "—"))
            col_b.metric("Level",     emp.get("level", "—"))
            col_c.metric("Location",  emp.get("location", "—"))
            col_d.metric("Sessions",  len(sessions_df))

            st.markdown("---")

            # KPIs
            total_cost   = sessions_df["total_cost_usd"].sum()
            total_turns  = sessions_df["num_turns"].sum()
            avg_duration = sessions_df["duration_min"].median()
            avg_api      = sessions_df["num_api_calls"].mean()
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Cost",      f"${total_cost:,.2f}")
            c2.metric("Total Turns",     f"{int(total_turns):,}")
            c3.metric("Median Session",  f"{avg_duration:.1f} min")
            c4.metric("Avg API/Session", f"{avg_api:.1f}")
            c5.metric("API Errors",      len(errors_df))

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<p class="section-header">📅 Daily Activity</p>', unsafe_allow_html=True)
                daily_df["date"] = pd.to_datetime(daily_df["date"])
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=daily_df["date"], y=daily_df["sessions"],
                                     name="Sessions", marker_color="#7c6fff", opacity=0.8), secondary_y=False)
                fig.add_trace(go.Scatter(x=daily_df["date"], y=daily_df["cost"],
                                         name="Cost ($)", line=dict(color="#ff9f43", width=2.5)), secondary_y=True)
                fig.update_yaxes(title_text="Sessions", secondary_y=False)
                fig.update_yaxes(title_text="Cost ($)", secondary_y=True)
                st.plotly_chart(fig_theme(fig, 300), use_container_width=True)

            with col2:
                st.markdown('<p class="section-header">🤖 Model Usage Mix</p>', unsafe_allow_html=True)
                if not model_df.empty:
                    fig2 = px.pie(model_df, values="calls", names="model",
                                  color_discrete_sequence=PALETTE, hole=0.45)
                    fig2.update_traces(textinfo="percent+label")
                    st.plotly_chart(fig_theme(fig2, 300), use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.markdown('<p class="section-header">🔧 Top Tools Used</p>', unsafe_allow_html=True)
                if not tools_df.empty:
                    fig3 = px.bar(tools_df.head(12), x="uses", y="tool_name",
                                  orientation="h", color="uses",
                                  color_continuous_scale="Blues", text_auto=",d")
                    fig3.update_traces(textposition="outside")
                    st.plotly_chart(fig_theme(fig3, 340), use_container_width=True)

            with col4:
                st.markdown('<p class="section-header">📐 Session Complexity</p>', unsafe_allow_html=True)
                fig4 = px.scatter(sessions_df, x="num_turns", y="total_cost_usd",
                                  size="num_api_calls", color="duration_min",
                                  size_max=20, color_continuous_scale="Viridis",
                                  labels={"num_turns": "Turns",
                                          "total_cost_usd": "Cost ($)",
                                          "duration_min": "Duration (min)"})
                st.plotly_chart(fig_theme(fig4, 340), use_container_width=True)

            st.markdown('<p class="section-header">📋 All Sessions</p>', unsafe_allow_html=True)
            st.dataframe(
                sessions_df[["start_time", "duration_min", "num_turns",
                              "num_api_calls", "num_tool_calls", "total_cost_usd", "total_tokens"]]
                .sort_values("start_time", ascending=False)
                .head(50)
                .style.format({
                    "total_cost_usd": "${:.4f}",
                    "duration_min":   "{:.1f} min",
                    "total_tokens":   "{:,}",
                })
                .background_gradient(subset=["total_cost_usd"], cmap="YlOrRd"),
                use_container_width=True,
            )

            if not errors_df.empty:
                st.markdown('<p class="section-header">🛑 API Errors</p>', unsafe_allow_html=True)
                st.dataframe(errors_df, use_container_width=True)


# ===========================================================================
# PAGE: Cohort & Retention
# ===========================================================================
elif page == "cohort":
    st.title("🔄 Cohort Analysis & Retention")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Retention Heatmap", "🪣 Session Funnel",
        "⏰ Session Cadence", "📈 Cluster Activity",
        "🆕 New vs Returning"
    ])

    with tab1:
        st.markdown('<p class="section-header">Weekly Engineer Retention Cohort</p>', unsafe_allow_html=True)
        retention = load_retention()
        fig = px.imshow(
            retention, text_auto=True,
            color_continuous_scale="RdYlGn",
            labels=dict(x="Weeks After First Session", y="Cohort", color="Retention %"),
            zmin=0, zmax=100, aspect="auto",
        )
        fig.update_xaxes(tickangle=0)
        st.plotly_chart(fig_theme(fig, 320), use_container_width=True)
        st.caption(
            "Each row is a cohort of engineers grouped by their first-session week. "
            "Week +0 = 100% by definition. Values show % still active N weeks later."
        )
        st.dataframe(retention.style.format("{:.1f}%")
                     .background_gradient(cmap="RdYlGn", vmin=0, vmax=100),
                     use_container_width=True)

    with tab2:
        st.markdown('<p class="section-header">Session Event Funnel</p>', unsafe_allow_html=True)
        funnel = load_funnel()
        fig2 = go.Figure(go.Funnel(
            y=funnel["stage"],
            x=funnel["count"],
            textinfo="value+percent initial",
            marker=dict(color=[
                "#7c6fff", "#6c8fff", "#5cafff", "#4ccfcf",
                "#3cef9f", "#2cff6f"
            ]),
        ))
        st.plotly_chart(fig_theme(fig2, 420), use_container_width=True)

        st.dataframe(
            funnel.style.format({"pct": "{:.1f}%", "drop_off_pct": "{:.1f}%",
                                 "count": "{:,}"})
            .background_gradient(subset=["pct"], cmap="RdYlGn"),
            use_container_width=True,
        )

    with tab3:
        st.markdown('<p class="section-header">Inter-Session Gap Distribution</p>', unsafe_allow_html=True)
        gaps = load_gaps()
        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.histogram(gaps, x="gap_hours", nbins=50,
                                color="practice", color_discrete_sequence=PALETTE,
                                labels={"gap_hours": "Hours Between Sessions"},
                                barmode="overlay", opacity=0.7, marginal="box")
            fig3.update_xaxes(range=[0, 120])
            st.plotly_chart(fig_theme(fig3, 340), use_container_width=True)
        with col2:
            fig4 = px.box(gaps, x="practice", y="gap_hours",
                          color="practice", color_discrete_sequence=PALETTE,
                          points="outliers",
                          labels={"gap_hours": "Gap (hours)", "practice": ""})
            fig4.update_yaxes(range=[0, 200])
            st.plotly_chart(fig_theme(fig4, 340), use_container_width=True)

        # Summary stats
        gap_stats = gaps.groupby("practice")["gap_hours"].agg(
            median="median", mean="mean", std="std",
            p10=lambda x: x.quantile(0.10), p90=lambda x: x.quantile(0.90)
        ).round(1).reset_index()
        st.dataframe(gap_stats.style.background_gradient(subset=["median"], cmap="Blues"),
                     use_container_width=True)

    with tab4:
        st.markdown('<p class="section-header">7-Day Rolling Sessions by User Segment</p>', unsafe_allow_html=True)
        rolling = load_rolling_clusters()
        fig5 = px.line(rolling, x="date", y="rolling_7d", color="cluster_label",
                       color_discrete_sequence=PALETTE, line_shape="spline",
                       labels={"rolling_7d": "7-day Rolling Sessions", "cluster_label": "Segment"})
        st.plotly_chart(fig_theme(fig5, 380), use_container_width=True)

        st.markdown('<p class="section-header">Cumulative Tool Adoption by Practice</p>', unsafe_allow_html=True)
        adoption = load_tool_adoption()
        all_prac = adoption[adoption["practice"] == "All"]
        by_prac  = adoption[adoption["practice"] != "All"]
        fig6 = px.line(by_prac, x="date", y="cumulative_tools",
                       color="practice", color_discrete_sequence=PALETTE, line_shape="spline",
                       labels={"cumulative_tools": "Distinct Tools Used", "practice": "Practice"})
        fig6.add_trace(go.Scatter(
            x=all_prac["date"], y=all_prac["cumulative_tools"],
            name="All Practices", line=dict(color="white", dash="dot", width=2.5),
        ))
        st.plotly_chart(fig_theme(fig6, 320), use_container_width=True)

    with tab5:
        st.markdown('<p class="section-header">New vs Returning Engineers — Daily</p>', unsafe_allow_html=True)
        nvr = load_new_returning()
        nvr["date"] = pd.to_datetime(nvr["date"])
        fig7 = px.bar(nvr, x="date", y="engineers", color="type",
                      color_discrete_map={"New": "#26de81", "Returning": "#7c6fff"},
                      barmode="stack",
                      labels={"engineers": "Engineers", "type": ""})
        st.plotly_chart(fig_theme(fig7, 360), use_container_width=True)

        # Activity profiles
        st.markdown('<p class="section-header">🦉 Night Owl vs 🐦 Early Bird Profiles</p>', unsafe_allow_html=True)
        ap = load_activity_profiles()
        col3, col4 = st.columns(2)
        with col3:
            fig8 = px.scatter(ap, x="night_owl_score", y="early_bird_score",
                              color="practice", size="sessions",
                              size_max=25, hover_data=["full_name", "level"],
                              color_discrete_sequence=PALETTE,
                              labels={"night_owl_score": "Night Owl % (after 20:00)",
                                      "early_bird_score": "Early Bird % (before 08:00)"})
            st.plotly_chart(fig_theme(fig8, 340), use_container_width=True)
        with col4:
            fig9 = px.histogram(ap, x="peak_hour", nbins=24, color="practice",
                                color_discrete_sequence=PALETTE, barmode="overlay",
                                labels={"peak_hour": "Most Common Active Hour"})
            st.plotly_chart(fig_theme(fig9, 340), use_container_width=True)

        top_owls = ap[["full_name", "practice", "level", "peak_hour",
                        "night_owl_score", "early_bird_score", "weekend_score"]].head(15)
        st.dataframe(top_owls.style.format(
            {"night_owl_score": "{:.1f}%", "early_bird_score": "{:.1f}%",
             "weekend_score": "{:.1f}%"}
        ).background_gradient(subset=["night_owl_score"], cmap="RdPu"),
                     use_container_width=True)


# ===========================================================================
# PAGE: Data Quality
# ===========================================================================
elif page == "quality":
    st.title("🛡️ Data Quality Report")
    st.caption("Automated validation of all 6 database tables across 10 checks")

    report = load_validation()
    score  = report["quality_score"]

    # Score gauge
    col_s, col_p, col_w, col_f = st.columns(4)
    color = "#26de81" if score >= 90 else "#fd9644" if score >= 70 else "#ff4444"
    col_s.metric("Quality Score", f"{score} / 100")
    col_p.metric("✅ Passed",  report["passed"])
    col_w.metric("⚠️ Warnings", report["warnings"])
    col_f.metric("❌ Failed",  report["failed"])

    # Score gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Data Health", "font": {"color": "#c0c0e0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#888"},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  60], "color": "#3a1010"},
                {"range": [60, 80], "color": "#3a2a10"},
                {"range": [80, 100],"color": "#103a10"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "value": 90},
        },
        number={"suffix": "%", "font": {"color": "#e0e0ff"}},
    ))
    fig.update_layout(height=280, **THEME)
    col_gauge, col_table = st.columns([1, 2])
    with col_gauge:
        st.plotly_chart(fig, use_container_width=True)
    with col_table:
        st.markdown('<p class="section-header">Check Results</p>', unsafe_allow_html=True)
        qdf = val.quality_summary_df(report)

        def _status_style(v):
            if v == "PASS":  return "background-color:#0d3b0d; color:#26de81; font-weight:600"
            if v == "WARN":  return "background-color:#3b2d0d; color:#fd9644; font-weight:600"
            return "background-color:#3b0d0d; color:#ff4444; font-weight:600"

        st.dataframe(
            qdf.style.applymap(_status_style, subset=["Status"])
               .set_properties(**{"font-size": "0.88rem"}),
            use_container_width=True, height=340
        )

    # Detail expanders for non-PASS checks
    non_pass = [r for r in report["results"] if r.status != "PASS"]
    if non_pass:
        st.markdown('<p class="section-header">⚠️ Issues Detail</p>', unsafe_allow_html=True)
        for r in non_pass:
            icon = "⚠️" if r.status == "WARN" else "❌"
            with st.expander(f"{icon} {r.check_name.replace('_',' ').title()} — {r.message}"):
                if r.detail is not None:
                    if isinstance(r.detail, dict):
                        for k, v in r.detail.items():
                            st.write(f"**{k}:** {v}")
                    elif isinstance(r.detail, list):
                        for item in r.detail:
                            st.write(f"• {item}")
                    else:
                        st.write(r.detail)
    else:
        st.success("🎉 All checks passed! The dataset is in excellent shape.")

    # Time-series decomposition on data quality page (meta-check)
    st.markdown("---")
    st.markdown('<p class="section-header">📉 Daily Cost — Trend / Seasonal / Residual Decomposition</p>', unsafe_allow_html=True)
    decomp = load_decompose()

    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         subplot_titles=("Actual vs Trend", "Seasonal Component", "Residual"),
                         vertical_spacing=0.08)
    fig2.add_trace(go.Scatter(x=decomp["date"], y=decomp["actual"],
                              name="Actual", line=dict(color="#7c6fff")), row=1, col=1)
    fig2.add_trace(go.Scatter(x=decomp["date"], y=decomp["trend"],
                              name="Trend", line=dict(color="#ff9f43", dash="dash", width=2.5)),
                   row=1, col=1)
    fig2.add_trace(go.Bar(x=decomp["date"], y=decomp["seasonal"],
                          name="Seasonal", marker_color="#26de81"), row=2, col=1)
    residual_color = np.where(decomp["residual_zscore"].abs() > 2.5, "#ff4444", "#aaaaff")
    fig2.add_trace(go.Bar(x=decomp["date"], y=decomp["residual"],
                          name="Residual", marker_color=residual_color.tolist()), row=3, col=1)
    fig2.update_layout(height=520, showlegend=True, **THEME)
    fig2.update_xaxes(showgrid=True, gridcolor="#2a2a4a")
    fig2.update_yaxes(showgrid=True, gridcolor="#2a2a4a")
    st.plotly_chart(fig2, use_container_width=True)

    ts = decomp.attrs.get("trend_strength", 0)
    ss = decomp.attrs.get("seasonal_strength", 0)
    c1, c2 = st.columns(2)
    c1.metric("Trend Strength", f"{ts:.3f}",
              help="1 − Var(residual)/Var(trend+residual). Closer to 1 = stronger trend.")
    c2.metric("Seasonal Strength", f"{ss:.3f}",
              help="1 − Var(residual)/Var(seasonal+residual). Closer to 1 = stronger weekly pattern.")

    # Model efficiency table
    st.markdown("---")
    st.markdown('<p class="section-header">⚡ Model Efficiency Index</p>', unsafe_allow_html=True)
    eff = load_efficiency()
    fig3 = px.bar(eff, x="model", y="efficiency_score",
                  color="efficiency_score", text_auto=".1f",
                  color_continuous_scale="RdYlGn",
                  labels={"efficiency_score": "Efficiency Score (0–100)"})
    fig3.update_traces(textposition="outside")
    fig3.update_layout(yaxis_range=[0, 110])
    st.plotly_chart(fig_theme(fig3, 300), use_container_width=True)
    st.caption(
        "Composite efficiency = 30% output yield + 30% cache leverage + 25% cost-per-output-token + 15% speed. "
        "All dimensions normalised [0,1] across models."
    )
    st.dataframe(
        eff[["model","calls","avg_cost","avg_output","avg_duration_s",
             "output_yield","cache_leverage","efficiency_score"]]
        .style.format({
            "avg_cost": "${:.5f}", "avg_output": "{:.0f} tok",
            "avg_duration_s": "{:.2f}s", "output_yield": "{:.3f}",
            "cache_leverage": "{:.3f}", "efficiency_score": "{:.1f}",
        })
        .background_gradient(subset=["efficiency_score"], cmap="RdYlGn"),
        use_container_width=True,
    )
