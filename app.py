import streamlit as st
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
[data-testid="stMetric"]{background:rgba(15,45,62,.82);border:1px solid #24526b;border-radius:14px;padding:13px}.stButton>button{background:#17b6a4;color:#03151a;border:0;border-radius:9px;font-weight:700;width:100%}.stButton>button:hover{background:#63ddd1;color:#03151a}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_ai(config_items):
    config = TwinConfig(**dict(config_items))
    return train_surrogate(make_training_data(config, samples=240))

with st.sidebar:
    st.markdown("## Control room")
    st.caption("Synthetic scenario inputs")
    st.divider()
    steam = st.slider("Steam rate", 20.0, 100.0, 55.0, 1.0, help="tonnes per day")
    steam_days = st.slider("Injection duration", 2, 8, 5, help="days per CSS cycle")
    cycle = st.slider("CSS cycle length", 10, 25, 15, help="days")
    st.divider()
    spm = st.slider("SRP speed", 2.0, 8.0, 4.5, 0.1, help="strokes per minute")
    stroke = st.slider("SRP stroke", 1.5, 4.0, 2.5, 0.1, help="metres")
    uptime = st.slider("Pump uptime", 0.75, 0.99, 0.94, 0.01)
    horizon = st.slider("Simulation horizon", 15, 90, 45, help="days")
    st.divider()
    st.caption("Editable synthetic assumptions—not field instructions.")

config = TwinConfig()
plan = OperatingPlan(steam, steam_days, cycle, spm, stroke, uptime)
frame = detect_anomalies(simulate(config, plan, days=horizon))
summary = kpis(frame)
model, quality = load_ai(tuple(config.as_dict().items()))

st.markdown("""<div class="hero"><div class="eyebrow">SIH 26120 · HEAVY-OIL OPERATIONS DEMONSTRATOR</div><h1>Baghewala Digital Twin</h1><p>Connect CSS steam injection, reservoir response, artificial lift and production outcomes in one transparent what-if simulation.</p></div>""", unsafe_allow_html=True)
st.warning("DEMONSTRATION MODEL — Outputs use synthetic data and simplified physics. They are not field-calibrated and must not guide live operations.")

metrics = [("Average oil", summary["avg_oil_bpd"], "bpd"), ("Cumulative oil", summary["cumulative_oil_bbl"], "bbl"), ("Steam–oil ratio", summary["steam_oil_ratio_t_per_bbl"], "t/bbl"), ("Peak temperature", summary["peak_temperature_c"], "°C"), ("Simulated net value", summary["net_value"], "units")]
for column, (name, value, unit) in zip(st.columns(5), metrics):
    column.metric(name, f"{value:,.1f}", unit)

overview, intelligence, assumptions = st.tabs(["Simulation overview", "AI & optimization", "Model transparency"])
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
        st.plotly_chart(px.scatter(frame, x="day", y="production_bpd", color="anomaly", title="Sensor-state anomaly screening", labels={"production_bpd":"Oil rate (bpd)"}, template="plotly_dark", color_discrete_map={False:"#58d6b2", True:"#ff6961"}), use_container_width=True)

with intelligence:
    one, two, three = st.columns(3)
    one.metric("Physics simulation", f"{summary['avg_oil_bpd']:,.1f} bpd")
    two.metric("AI surrogate estimate", f"{predict(model, plan):,.1f} bpd")
    three.metric("Synthetic training fit", f"R² {quality['r2']:.3f}")
    st.caption(f"Anomaly screening identified {int(frame.anomaly.sum())} unusual synthetic sensor states across {horizon} simulated days.")
    if st.button("Run 30-day optimization"):
        with st.spinner("Evaluating bounded CSS and SRP scenarios..."):
            best, best_kpis = optimize_plan(config)
        st.success("Synthetic recommendation generated")
        result, summary_column = st.columns([2, 1])
        with result:
            st.markdown(f"### Recommended scenario\n**CSS:** {best.steam_rate_tpd:.1f} t/day for {best.steam_days} days every {best.cycle_length_days} days  \\n**SRP:** {best.srp_spm:.1f} spm × {best.stroke_m:.1f} m stroke · {best.uptime:.0%} uptime")
        with summary_column:
            st.metric("Optimized oil", f"{best_kpis['avg_oil_bpd']:,.1f} bpd")
            st.metric("Optimized net value", f"{best_kpis['net_value']:,.0f}")

with assumptions:
    st.subheader("Transparent simplified model")
    st.markdown("""- **Thermal and pressure response:** steam input plus first-order relaxation to a baseline.
- **Viscosity and mobility:** exponential temperature sensitivity; mobility varies inversely with viscosity.
- **Well inflow:** temperature-adjusted productivity index × drawdown.
- **Surface production:** the lesser of reservoir inflow and SRP capacity, adjusted for uptime.
- **AI:** Random Forest surrogate trained only on synthetic simulator scenarios.
- **Anomalies:** Isolation Forest flags unusual multi-sensor patterns in the current synthetic run.""")
    st.json(config.as_dict())
