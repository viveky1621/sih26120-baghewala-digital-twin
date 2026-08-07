from scipy.optimize import differential_evolution
from .simulator import OperatingPlan, simulate, kpis


def optimize_plan(config, days: int = 30):
    """Maximize synthetic net value under bounded demonstration limits."""
    bounds = [(20, 100), (2, 8), (10, 25), (2, 8), (1.5, 4), (0.80, 0.99)]
    def objective(x):
        plan = OperatingPlan(x[0], int(round(x[1])), int(round(x[2])), x[3], x[4], x[5])
        run = simulate(config, plan, days=days)
        economics = kpis(run)["net_value"]
        synthetic_risk_penalty = run.stress_index.sum() * config.synthetic_lift_risk_cost_per_day
        return -(economics - synthetic_risk_penalty)
    result = differential_evolution(objective, bounds, seed=11, polish=True)
    x = result.x
    plan = OperatingPlan(float(x[0]), int(round(x[1])), int(round(x[2])), float(x[3]), float(x[4]), float(x[5]))
    run = simulate(config, plan, days=days)
    result_kpis = kpis(run)
    result_kpis["risk_adjusted_net_value"] = result_kpis["net_value"] - run.stress_index.sum() * config.synthetic_lift_risk_cost_per_day
    return plan, result_kpis
