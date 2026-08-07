import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from digital_twin.config import TwinConfig
from digital_twin.simulator import OperatingPlan, simulate, kpis
from digital_twin.ml import make_training_data, train_surrogate, predict, detect_anomalies
from digital_twin.optimizer import optimize_plan

st.set_page_config(page_title="Baghewala Digital Twin", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.stApp{background:radial-gradient(circle at 15% 0%,#172e45 0,#09131f 42%,#061018 100%)}
[data-testid="stSidebar"]{background:#0c1d2b;border-right:1px solid #254359} h1,h2,h3{color:#f4fbff!important}
.hero{padding:1.6rem 1.8rem;border:1px solid #2f718c;border-radius:18px;background:linear-gradient(110deg,#102d43,#0e4b61);margin-bottom:1rem}.eyebrow{color:#7fe4df;font-weight:700;letter-spacing:.1em;font-size:.75rem}.hero p{color:#d1e6ee;margin-bottom:0;font-size:1.02rem}
.metric-card,.model-card,.feasibility-card{background:linear-gradient(165deg,#132638,#0f1f2e);border:1px solid #1e4a61;border-radius:14px;padding:15px;min-height:118px}.metric-label{color:#93b6c6;font-size:.78rem}.metric-value{font-size:1.75rem;font-weight:700;margin:6px 0}.metric-unit{font-size:.8rem;color:#9ab0bb;font-weight:400}.badge{display:inline-block;border-radius:20px;padding:4px 8px;font-size:.72rem;font-weight:700}.green{background:rgba(62,207,142,.17);color:#3ecf8e}.red{background:rgba(255,107,91,.17);color:#ff8072}.amber{background:rgba(240,166,58,.17);color:#f0b45a}.model-tag{display:inline-block;border-radius:5px;padding:3px 8px;font-size:.68rem;font-weight:800;margin-bottom:8px}.model-card{min-height:180px}.model-card p,.feasibility-card p{color:#a9c0cb;font-size:.85rem;line-height:1.5}.stButton>button{background:#17b6a4;color:#03151a;border:0;border-radius:9px;font-weight:700;width:100%}.stButton>button:hover{background:#63ddd1;color:#03151a}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_ai(config_items):
    config = TwinConfig(**dict(config_items))
    return train_surrogate(make_training_data(config, samples=240))


def readiness_score():
    """Transparent prototype-readiness score; never a probability of field success."""
    components = {"Data readiness": 25, "Engineering integration": 75, "Model validation": 30, "Operator workflow": 70}
    weights = {"Data readiness": 0.35, "Engineering integration": 0.25, "Model validation": 0.25, "Operator workflow": 0.15}
    return round(sum(components[key] * weights[key] for key in components)), components


def add_synthetic_lift_proxies(frame: pd.DataFrame) -> pd.DataFrame:
    """Portable demonstrator-only lift indicators; intentionally not field failure probabilities."""
    out = frame.copy()
    fillage = 1.0 - (out["viscosity_cp"] / 12_000.0) * (out["srp_spm"] / 8.0)
    out["fillage_proxy"] = out.get("fillage_proxy", fillage.clip(0.15, 1.0))
    stress = (1.0 - out["fillage_proxy"]) * (out["srp_spm"] / 6.0) * (out["stroke_m"] / 2.5)
    out["stress_index"] = out.get("stress_index", stress.clip(0.0, 1.0))
    out["synthetic_stress_risk"] = out["stress_index"].clip(0.0, 1.0)
    return out


def metric_card(label: str, value: float, unit: str, favorable: bool, note: str) -> str:
    tone = "green" if favorable else "red"
    arrow = "↑ favorable direction" if favorable else "↓ lower is better"
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value:,.1f}<span class="metric-unit"> {unit}</span></div><span class="badge {tone}">{arrow}</span><div class="metric-label" style="margin-top:8px">{note}</div></div>'

with st.sidebar:
    st.markdown("## Control room")
    st.caption("Synthetic scenario inputs")
    st.divider()
    steam = st.number_input("Steam rate (t/day)", min_value=20.0, max_value=100.0, value=55.0, step=1.0, help="tonnes per day")
    steam_days = st.number_input("Injection duration (days)", min_value=2, max_value=8, value=5, step=1, help="days per CSS cycle")
    cycle = st.number_input("CSS cycle length (days)", min_value=10, max_value=25, value=15, step=1, help="days")
    st.divider()
    spm = st.number_input("SRP speed (spm)", min_value=2.0, max_value=8.0, value=4.5, step=0.1, help="strokes per minute")
    stroke = st.number_input("SRP stroke (m)", min_value=1.5, max_value=4.0, value=2.5, step=0.1, help="metres")
    uptime = st.number_input("Pump uptime", min_value=0.75, max_value=0.99, value=0.94, step=0.01, format="%.2f")
    horizon = st.number_input("Simulation horizon (days)", min_value=15, max_value=90, value=45, step=1, help="days")
    st.divider()
    st.caption("Editable synthetic assumptions—not field instructions.")

config = TwinConfig()
plan = OperatingPlan(steam, steam_days, cycle, spm, stroke, uptime)
frame = add_synthetic_lift_proxies(detect_anomalies(simulate(config, plan, days=horizon)))
summary = kpis(frame)
summary.setdefault("avg_fillage_proxy", float(frame.fillage_proxy.mean()))
summary.setdefault("avg_stress_index", float(frame.stress_index.mean()))
model, quality = load_ai(tuple(config.as_dict().items()))

st.markdown("""<div class="hero"><div class="eyebrow">SIH 26120 · HEAVY-OIL OPERATIONS DEMONSTRATOR</div><h1>Baghewala Digital Twin</h1><p>Connect CSS steam injection, reservoir response, artificial lift and production outcomes in one transparent what-if simulation.</p></div>""", unsafe_allow_html=True)
st.warning("DEMONSTRATION MODEL — Outputs use synthetic data and simplified physics. They are not field-calibrated and must not guide live operations.")

metrics = [
    ("Average oil", summary["avg_oil_bpd"], "bpd", True, "synthetic production"),
    ("Cumulative oil", summary["cumulative_oil_bbl"], "bbl", True, "over selected horizon"),
    ("Steam–oil ratio", summary["steam_oil_ratio_t_per_bbl"], "t/bbl", False, "efficiency: lower is better"),
    ("Peak temperature", summary["peak_temperature_c"], "°C", True, "thermal response proxy"),
    ("Fillage proxy", summary["avg_fillage_proxy"] * 100, "%", True, "synthetic lift indicator"),
    ("Stress proxy", summary["avg_stress_index"] * 100, "%", False, "synthetic: lower is better"),
]
for column, card in zip(st.columns(3), metrics[:3]):
    column.markdown(metric_card(*card), unsafe_allow_html=True)
for column, card in zip(st.columns(3), metrics[3:]):
    column.markdown(metric_card(*card), unsafe_allow_html=True)

overview, intelligence, readiness, assumptions = st.tabs(["Simulation overview", "AI & optimization", "Readiness & models", "Model transparency"])
with overview:
    left, right = st.columns(2)
    with left:
        chart = px.line(frame, x="day", y=["production_bpd", "inflow_bpd", "pump_capacity_bpd"], labels={"value":"Rate (bpd)","variable":"Series"}, title="Production system performance", template="plotly_dark")
        chart.update_layout(legend_title_text="")
        st.plotly_chart(chart, use_container_width=True)
    with right:
        chart = go.Figure()
        chart.add_trace(go.Scatter(x=frame.day, y=frame.reservoir_temp_c, name="Temperature (°C)", line=dict(color="#f3a847", width=3)))
        chart.add_trace(go.Scatter(x=frame.day, y=frame.reservoir_pressure_kpa, name="Pressure (kPa)", yaxis="y2", line=dict(color="#55d6d0", width=3)))
        chart.update_layout(title="Lumped reservoir response", template="plotly_dark", yaxis=dict(title="Temperature (°C)"), yaxis2=dict(title="Pressure (kPa)", overlaying="y", side="right"), legend_title_text="")
        st.plotly_chart(chart, use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.area(frame, x="day", y="steam_rate_tpd", title="CSS steam schedule", labels={"steam_rate_tpd":"Steam (t/day)"}, template="plotly_dark", color_discrete_sequence=["#4fc3f7"]), use_container_width=True)
    with right:
        risk_chart = px.line(frame, x="day", y="synthetic_stress_risk", title="Synthetic artificial-lift stress score", labels={"synthetic_stress_risk":"Stress score","day":"Day"}, template="plotly_dark", color_discrete_sequence=["#ff6961"])
        risk_chart.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(risk_chart, use_container_width=True)

with intelligence:
    one, two, three, four = st.columns(4)
    one.metric("Physics simulation", f"{summary['avg_oil_bpd']:,.1f} bpd")
    two.metric("AI surrogate estimate", f"{predict(model, plan):,.1f} bpd")
    three.metric("Synthetic training fit", f"R² {quality['r2']:.3f}")
    four.metric("Peak stress score", f"{frame.synthetic_stress_risk.max():.0%}")
    st.caption(f"Anomaly screening identified {int(frame.anomaly.sum())} unusual synthetic sensor states across {horizon} simulated days. The stress score is simulator-trained and is not a field failure probability.")
    if st.button("Run 30-day optimization"):
        with st.spinner("Evaluating bounded CSS and SRP scenarios..."):
            best, best_kpis = optimize_plan(config)
        st.success("Synthetic recommendation generated")
        result, summary_column = st.columns([2, 1])
        with result:
            st.markdown(f"### Recommended scenario\n**CSS:** {best.steam_rate_tpd:.1f} t/day for {best.steam_days} days every {best.cycle_length_days} days  \\n**SRP:** {best.srp_spm:.1f} spm × {best.stroke_m:.1f} m stroke · {best.uptime:.0%} uptime")
        with summary_column:
            st.metric("Optimized oil", f"{best_kpis['avg_oil_bpd']:,.1f} bpd")
            st.metric("Risk-adjusted value", f"{best_kpis.get('risk_adjusted_net_value', best_kpis['net_value']):,.0f}")

with readiness:
    score, components = readiness_score()
    st.subheader("Three-model decision layer")
    model_columns = st.columns(3)
    model_cards = [
        ("MODEL 1", "#17b6a4", "Production prediction", "Random Forest surrogate maps CSS and SRP settings to synthetic average production. Retrain with approved historical production, CSS and lift records."),
        ("MODEL 2", "#ff6b5b", "Lift stress screening", "Uses viscosity, fillage proxy, SRP speed and stroke to highlight synthetic stress conditions. It is not a field failure forecast until validated on actual events."),
        ("MODEL 3", "#8b7cf6", "Joint optimizer", "Searches bounded CSS and SRP combinations for simulated economics while applying a synthetic lift-stress penalty."),
    ]
    for column, (tag, colour, title, text) in zip(model_columns, model_cards):
        column.markdown(f'<div class="model-card"><span class="model-tag" style="background:{colour}26;color:{colour}">{tag}</span><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("Feasibility and deployment readiness")
    left, right = st.columns([1, 2])
    with left:
        tone = "green" if score >= 70 else "amber" if score >= 45 else "red"
        label = "Field-pilot ready" if score >= 70 else "Prototype-stage" if score >= 45 else "Needs validation"
        st.markdown(f'<div class="feasibility-card"><div class="metric-label">PROTOTYPE READINESS</div><div class="metric-value">{score}%</div><span class="badge {tone}">{label}</span><p>This is a transparent maturity estimate—not a probability of field success.</p></div>', unsafe_allow_html=True)
    with right:
        readiness_frame = pd.DataFrame({"Area": list(components), "Readiness": list(components.values())})
        readiness_chart = px.bar(readiness_frame, x="Readiness", y="Area", orientation="h", range_x=[0, 100], title="What determines feasibility", template="plotly_dark", color="Readiness", color_continuous_scale=["#ff6961", "#f3a847", "#17b6a4"])
        readiness_chart.update_layout(coloraxis_showscale=False)
        st.plotly_chart(readiness_chart, use_container_width=True)
    st.markdown("### Path to field feasibility")
    st.markdown("1. **Data readiness:** ingest approved historical CSS, production, VFD/SRP, pump-card and failure records.  \\n2. **Engineering validation:** calibrate the reservoir, wellbore and pump relationships against field specialists.  \\n3. **Model validation:** use time-separated holdout data, error targets and human review before any recommendation.  \\n4. **Controlled pilot:** begin with advisory-only operation and documented operator override.")

with assumptions:
    st.subheader("Transparent simplified model")
    st.markdown("""- **Thermal and pressure response:** steam input plus first-order relaxation to a baseline.
- **Viscosity and mobility:** exponential temperature sensitivity; mobility varies inversely with viscosity.
- **Well inflow:** temperature-adjusted productivity index × drawdown.
- **Surface production:** the lesser of reservoir inflow and SRP capacity, adjusted for uptime.
- **Model 1 — production AI:** Random Forest surrogate trained only on synthetic simulator scenarios.
- **Model 2 — lift stress AI:** Random Forest classifier trained on simulator-generated stress labels using viscosity, fillage proxy, SRP speed and stroke. It is not a failure-prediction model until trained and validated on field events.
- **Model 3 — joint optimizer:** bounded CSS/SRP search that maximizes simulated economics less a clearly labeled synthetic stress penalty.
- **Anomalies:** Isolation Forest flags unusual multi-sensor patterns in the current synthetic run.""")
    st.json(config.as_dict())
