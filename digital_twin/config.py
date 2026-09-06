from dataclasses import dataclass, asdict


@dataclass
class TwinConfig:
    """Synthetic, editable assumptions for the demonstration model."""
    ambient_temp_c: float = 28.0
    base_pressure_kpa: float = 1850.0
    initial_pressure_kpa: float = 1950.0
    initial_temp_c: float = 35.0
    viscosity_at_30c_cp: float = 8500.0
    viscosity_temp_sensitivity: float = 0.037
    base_productivity_bpd_per_kpa: float = 0.12
    heat_gain_c_per_tonne: float = 0.11
    thermal_decay_per_day: float = 0.10
    pressure_gain_kpa_per_tonne: float = 0.8
    pressure_decay_per_day: float = 0.16
    reservoir_depletion_kpa_per_bbl: float = 0.015
    tubing_head_pressure_kpa: float = 350.0
    pump_capacity_factor: float = 0.095
    default_uptime: float = 0.94
    steam_energy_cost_per_tonne: float = 3.0
    lifting_cost_per_bbl: float = 0.80
    oil_value_per_bbl: float = 65.0

    def as_dict(self):
        return asdict(self)
