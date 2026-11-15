from src.model import m4_pit_runtime


def test_runtime_m4_loader_fails_closed_when_not_configured(monkeypatch):
    monkeypatch.delenv("FHM_M4_PIT_INPUT_DIR", raising=False)
    snapshot, audit, status = m4_pit_runtime.build_runtime_m4_pit_snapshot(
        decision_time="2026-11-14T15:00:00+08:00",
    )
    assert snapshot.empty and audit.empty
    assert status["status"] == "NOT_CONFIGURED"
    assert status["builder_invoked"] is False
    assert status["historical_pit_eligible"] is False


def test_runtime_m4_loader_reports_missing_inputs(tmp_path):
    snapshot, audit, status = m4_pit_runtime.build_runtime_m4_pit_snapshot(
        decision_time="2026-11-14T15:00:00+08:00",
        input_dir=tmp_path,
        expected_universe_size=100,
    )
    assert snapshot.empty and audit.empty
    assert status["status"] == "MISSING_CERTIFIED_INPUTS"
    assert len(status["missing_inputs"]) == 4


def test_runtime_m4_loader_requires_declared_universe_size(tmp_path):
    snapshot, audit, status = m4_pit_runtime.build_runtime_m4_pit_snapshot(
        decision_time="2026-11-14T15:00:00+08:00",
        input_dir=tmp_path,
    )
    assert snapshot.empty and audit.empty
    assert status["status"] == "INVALID_CONFIGURATION"
    assert status["builder_invoked"] is False


def test_runtime_m4_loader_passes_only_explicit_archives_to_builder(tmp_path, monkeypatch):
    import pandas as pd

    for name in ("stock_bars", "stock_returns", "industry_returns", "market_returns"):
        pd.DataFrame({"historical_pit_eligible": [True]}).to_csv(
            tmp_path / f"{name}.csv", index=False
        )
    expected = pd.DataFrame([{
        "symbol": "000001", "historical_pit_eligible": True,
        "availability_basis": "derived_from_certified_pit_inputs",
    }])
    expected_audit = pd.DataFrame([{"factor_name": "price_structure_pct", "status": "ranked"}])
    captured = {}

    def fake_builder(stock_bars, stock_returns, industry_returns, market_returns, **kwargs):
        captured["inputs"] = (stock_bars, stock_returns, industry_returns, market_returns)
        captured["kwargs"] = kwargs
        return expected, expected_audit

    monkeypatch.setattr(m4_pit_runtime, "build_m4_pit_factor_snapshot", fake_builder)
    snapshot, audit, status = m4_pit_runtime.build_runtime_m4_pit_snapshot(
        decision_time="2026-11-14T15:00:00+08:00",
        input_dir=tmp_path,
        expected_universe_size=1,
    )
    assert snapshot.equals(expected)
    assert audit.equals(expected_audit)
    assert status["status"] == "BUILT"
    assert status["builder_invoked"] is True
    assert status["historical_pit_eligible"] is True
    assert captured["kwargs"]["expected_universe_size"] == 1
