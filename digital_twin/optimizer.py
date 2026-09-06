from scipy.optimize import differential_evolution
from .simulator import OperatingPlan, simulate, kpis


def optimize_plan(config, days: int = 30):
    """Maximize synthetic net value under bounded demonstration limits."""
    bounds = [(20, 100), (2, 8), (10, 25), (2, 8), (1.5, 4), (0.80, 0.99)]
    def objective(x):
        plan = OperatingPlan(x[0], int(round(x[1])), int(round(x[2])), x[3], x[4], x[5])
        return -kpis(simulate(config, plan, days=days))["net_value"]
    result = differential_evolution(objective, bounds, seed=11, polish=True)
    x = result.x
    plan = OperatingPlan(float(x[0]), int(round(x[1])), int(round(x[2])), float(x[3]), float(x[4]), float(x[5]))
    return plan, kpis(simulate(config, plan, days=days))
