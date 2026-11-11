import pandas as pd

from src import fhm100


def test_m3_ready_state_is_exposed_for_m5(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fhm100, "assemble_m3_history",
        lambda *args, **kwargs: {
            "status": "READY", "reason": None, "market_state": "RISK_ON",
            "position_scale": 1.0, "valid_sessions": 119,
            "latest_trade_date": "2026-11-10",
            "historical_backtest_eligible": False,
        },
    )

    result = fhm100.load_m3_market_context(
        decision_time=pd.Timestamp("2026-11-11T09:00:00Z"), archive_root=tmp_path
    )

    assert result["status"] == "READY"
    assert result["market_state"] == "RISK_ON"
    assert result["position_scale"] == 1.0
    assert result["historical_backtest_eligible"] is False


def test_m3_pending_state_is_not_passed_to_m5(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fhm100, "assemble_m3_history",
        lambda *args, **kwargs: {
            "status": "STATE_CONFIRMATION_PENDING",
            "reason": "two_consecutive_sessions_not_yet_confirmed",
            "market_state": "INSUFFICIENT_DATA", "position_scale": None,
            "valid_sessions": 119,
        },
    )

    result = fhm100.load_m3_market_context(archive_root=tmp_path)

    assert result["status"] == "STATE_CONFIRMATION_PENDING"
    assert result["market_state"] is None
    assert result["position_scale"] is None


def test_m3_missing_archive_and_assembly_errors_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fhm100, "assemble_m3_history",
        lambda *args, **kwargs: {"status": "INSUFFICIENT_DATA", "reason": "index_archive_missing"},
    )
    missing = fhm100.load_m3_market_context(archive_root=tmp_path)
    assert missing["market_state"] is None
    assert missing["status"] == "INSUFFICIENT_DATA"

    def broken(*args, **kwargs):
        raise ValueError("corrupt archive")
    monkeypatch.setattr(fhm100, "assemble_m3_history", broken)
    failed = fhm100.load_m3_market_context(archive_root=tmp_path)
    assert failed["status"] == "INVALID_DATA"
    assert failed["market_state"] is None
    assert "corrupt archive" in failed["reason"]
