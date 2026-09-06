import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from .simulator import OperatingPlan, simulate

FEATURES = ["steam_rate_tpd", "steam_days", "cycle_length_days", "srp_spm", "stroke_m", "uptime"]


def make_training_data(config, samples: int = 240, seed: int = 7) -> pd.DataFrame:
    rng, rows = np.random.default_rng(seed), []
    for _ in range(samples):
        plan = OperatingPlan(float(rng.uniform(20, 100)), int(rng.integers(2, 9)), int(rng.integers(10, 26)), float(rng.uniform(2, 8)), float(rng.uniform(1.5, 4)), float(rng.uniform(0.75, 0.99)))
        run = simulate(config, plan, days=30, seed=int(rng.integers(1_000_000)), add_noise=True)
        rows.append({"steam_rate_tpd": plan.steam_rate_tpd, "steam_days": plan.steam_days, "cycle_length_days": plan.cycle_length_days, "srp_spm": plan.srp_spm, "stroke_m": plan.stroke_m, "uptime": plan.uptime, "avg_oil_bpd": run.production_bpd.mean(), "net_value": run.net_value_per_day.sum()})
    return pd.DataFrame(rows)


def train_surrogate(data: pd.DataFrame):
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=12, n_jobs=-1)
    model.fit(data[FEATURES], data["avg_oil_bpd"])
    fitted = model.predict(data[FEATURES])
    return model, {"mae_bpd": float(mean_absolute_error(data.avg_oil_bpd, fitted)), "r2": float(r2_score(data.avg_oil_bpd, fitted))}


def predict(model, plan: OperatingPlan) -> float:
    values = pd.DataFrame([[plan.steam_rate_tpd, plan.steam_days, plan.cycle_length_days, plan.srp_spm, plan.stroke_m, plan.uptime]], columns=FEATURES)
    return float(model.predict(values)[0])


def detect_anomalies(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["reservoir_temp_c", "reservoir_pressure_kpa", "viscosity_cp", "production_bpd", "inflow_bpd"]
    out = frame.copy()
    out["anomaly"] = IsolationForest(contamination=0.08, random_state=21).fit_predict(out[columns]) == -1
    return out
