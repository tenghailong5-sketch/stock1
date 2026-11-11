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


def test_universe_provider_failure_creates_explicit_empty_scan():
    error = RuntimeError("Push2RateLimitError: temporary provider rate limit")
    watchlist, errors = fhm100.empty_scan_result(error)

    assert watchlist.empty
    assert "技术观察通过" in watchlist.columns
    assert len(errors) == 1
    assert errors.iloc[0]["代码"] == "UNIVERSE"
    assert "temporary provider rate limit" in errors.iloc[0]["原因"]


def test_m5_blocker_summary_counts_reasons_by_strategy():
    frame = pd.DataFrame([
        {"strategy": "A", "reasons": ["missing_factor:x", "missing_gate:y"]},
        {"strategy": "A", "reasons": ["missing_factor:x"]},
        {"strategy": "C", "reasons": ["missing_factor:z"]},
    ])

    summary = fhm100.summarize_m5_blockers(frame)

    counts = {
        (row.strategy, row.blocker): row.count
        for row in summary.itertuples(index=False)
    }
    assert counts[("A", "missing_factor:x")] == 2
    assert counts[("A", "missing_gate:y")] == 1
    assert counts[("C", "missing_factor:z")] == 1


def test_m5_blocker_summary_handles_empty_diagnostics():
    summary = fhm100.summarize_m5_blockers(pd.DataFrame())
    assert summary.empty
    assert list(summary.columns) == ["strategy", "blocker", "count"]
