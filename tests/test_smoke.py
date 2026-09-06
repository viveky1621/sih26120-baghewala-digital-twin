from digital_twin.config import TwinConfig
from digital_twin.simulator import OperatingPlan, simulate, kpis
from digital_twin.ml import make_training_data, train_surrogate, predict, detect_anomalies
from digital_twin.optimizer import optimize_plan


def test_simulation_and_ml():
    config, plan = TwinConfig(), OperatingPlan()
    run = simulate(config, plan, days=20)
    assert len(run) == 20
    assert (run.production_bpd >= 0).all()
    assert kpis(run)["cumulative_oil_bbl"] > 0
    model, _ = train_surrogate(make_training_data(config, samples=30))
    assert predict(model, plan) > 0
    assert "anomaly" in detect_anomalies(run).columns


def test_optimizer_returns_valid_plan():
    plan, result = optimize_plan(TwinConfig(), days=10)
    assert plan.steam_days < plan.cycle_length_days
    assert result["net_value"] > 0
