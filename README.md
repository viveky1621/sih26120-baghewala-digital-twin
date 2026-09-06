# SIH 26120 — Baghewala Heavy-Oil Digital Twin (Prototype)

> **Demonstration only.** This project uses synthetic data and simplified engineering relationships. It is not field-calibrated, not a reservoir simulator, and must not be used for operational decisions without validation against approved field data and domain review.

## What it demonstrates

The prototype links a cyclic-steam-stimulation (CSS) schedule to a lumped reservoir response, temperature-dependent heavy-oil viscosity, productivity/inflow, sucker-rod-pump (SRP) capacity, surface production, predictive ML, anomaly detection, and an operating-point optimizer.

## Assumptions (deliberately configurable)

- One well and one lumped near-wellbore reservoir volume; no spatial steam-front model.
- Steam rate adds heat and a modest pressure-support term. Heat and pressure decay to base conditions between cycles.
- Heavy-oil viscosity follows an exponential temperature sensitivity. Mobility is inverse viscosity.
- Inflow uses a temperature-adjusted productivity index multiplied by drawdown.
- SRP capacity is proportional to stroke length, strokes per minute, volumetric efficiency and uptime.
- Surface rate is the lesser of reservoir inflow and pump capacity, adjusted by uptime.
- The ML model learns the simulator's synthetic behavior; it is a surrogate, not independent field truth.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

The dashboard supports scenario controls, a 30-day simulation, ML prediction versus physics, anomaly flags, and a constrained optimization recommendation.

## Project layout

- `digital_twin/config.py` — all synthetic physical and operational assumptions.
- `digital_twin/simulator.py` — coupled daily engineering simulation.
- `digital_twin/ml.py` — synthetic training-data generation, random-forest surrogate, anomaly detector.
- `digital_twin/optimizer.py` — bounded operating-point optimization.
- `app.py` — Streamlit presentation dashboard.
- `tests/test_smoke.py` — quick reproducibility checks.

