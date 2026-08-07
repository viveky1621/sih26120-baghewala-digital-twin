from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import TwinConfig


@dataclass
class OperatingPlan:
    steam_rate_tpd: float = 55.0
    steam_days: int = 5
    cycle_length_days: int = 15
    srp_spm: float = 4.5
    stroke_m: float = 2.5
    uptime: float = 0.94


def simulate(config: TwinConfig, plan: OperatingPlan, days: int = 60, seed: int = 42, add_noise: bool = False) -> pd.DataFrame:
    """Daily lumped CSS-reservoir-SRP simulation; entirely synthetic."""
    rng = np.random.default_rng(seed)
    temp, pressure, records = config.initial_temp_c, config.initial_pressure_kpa, []
    for day in range(1, days + 1):
        steam_on = ((day - 1) % max(plan.cycle_length_days, 1)) < min(plan.steam_days, plan.cycle_length_days)
        steam = plan.steam_rate_tpd if steam_on else 0.0
        temp += config.heat_gain_c_per_tonne * steam - config.thermal_decay_per_day * (temp - config.ambient_temp_c)
        pressure += config.pressure_gain_kpa_per_tonne * steam - config.pressure_decay_per_day * (pressure - config.base_pressure_kpa)
        viscosity = config.viscosity_at_30c_cp * np.exp(-config.viscosity_temp_sensitivity * (temp - 30.0))
        mobility = config.viscosity_at_30c_cp / max(viscosity, 1.0)
        pi = config.base_productivity_bpd_per_kpa * mobility
        inflow = pi * max(pressure - config.tubing_head_pressure_kpa, 0.0)
        pump_capacity = config.pump_capacity_factor * plan.srp_spm * plan.stroke_m * 100.0
        # Synthetic artificial-lift health proxies. They are not measured fillage or failure data.
        fillage = float(np.clip(1.0 - (viscosity / 12_000.0) * (plan.srp_spm / 8.0), 0.15, 1.0))
        stress_index = float(np.clip((1.0 - fillage) * (plan.srp_spm / 6.0) * (plan.stroke_m / 2.5), 0.0, 1.0))
        production = min(inflow, pump_capacity * fillage) * plan.uptime
        if add_noise:
            production *= max(0.0, 1 + rng.normal(0, 0.035))
        pressure -= config.reservoir_depletion_kpa_per_bbl * production
        steam_cost = steam * config.steam_energy_cost_per_tonne
        net = production * (config.oil_value_per_bbl - config.lifting_cost_per_bbl) - steam_cost
        records.append({"day": day, "steam_on": steam_on, "steam_rate_tpd": steam, "reservoir_temp_c": temp, "reservoir_pressure_kpa": pressure, "viscosity_cp": viscosity, "mobility_multiplier": mobility, "productivity_index": pi, "inflow_bpd": inflow, "pump_capacity_bpd": pump_capacity, "fillage_proxy": fillage, "stress_index": stress_index, "production_bpd": production, "srp_spm": plan.srp_spm, "stroke_m": plan.stroke_m, "steam_cost": steam_cost, "net_value_per_day": net})
    return pd.DataFrame(records)


def kpis(frame: pd.DataFrame) -> dict:
    oil = frame.production_bpd.sum()
    return {"avg_oil_bpd": float(frame.production_bpd.mean()), "cumulative_oil_bbl": float(oil), "steam_tonnes": float(frame.steam_rate_tpd.sum()), "steam_oil_ratio_t_per_bbl": float(frame.steam_rate_tpd.sum() / max(oil, 1e-9)), "net_value": float(frame.net_value_per_day.sum()), "peak_temperature_c": float(frame.reservoir_temp_c.max()), "avg_fillage_proxy": float(frame.fillage_proxy.mean()), "avg_stress_index": float(frame.stress_index.mean())}
