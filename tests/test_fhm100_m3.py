import json

import pandas as pd

from src import fhm100


def test_m3_ready_state_is_exposed_for_m5(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fhm100, "assemble_m3_history",
        lambda *args, **kwargs: {
            "status": "READY", "reason": None, "market_state": "RISK_ON",
            "position_scale": 1.0, "market_environment_score": 92.5,
            "valid_sessions": 119,
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
    assert result["market_environment_score"] == 92.5
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


def test_m3_environment_factor_is_only_passed_to_m5_when_ready():
    watchlist = pd.DataFrame({"代码": ["000001", "000002"]})
    ready = {
        "market_state": "RISK_ON",
        "market_environment_score": 92.5,
    }
    frame = fhm100.build_m5_diagnostic_input(watchlist, ready)
    assert frame["market_environment"].tolist() == [92.5, 92.5]
    assert frame["market_state"].tolist() == ["RISK_ON", "RISK_ON"]

    pending = {
        "market_state": None,
        "market_environment_score": None,
    }
    frame = fhm100.build_m5_diagnostic_input(watchlist, pending)
    assert "market_environment" not in frame.columns
    assert frame["market_state"].isna().all()


def test_m3_environment_factor_rejects_out_of_range_or_non_numeric_values(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fhm100, "assemble_m3_history",
        lambda *args, **kwargs: {
            "status": "READY", "reason": None, "market_state": "RISK_ON",
            "position_scale": 1.0, "market_environment_score": 101.0,
            "valid_sessions": 119, "latest_trade_date": "2026-11-10",
        },
    )
    result = fhm100.load_m3_market_context(
        decision_time=pd.Timestamp("2026-11-11T09:00:00Z"), archive_root=tmp_path
    )
    assert result["market_state"] == "RISK_ON"
    assert result["market_environment_score"] is None


def _write_forward_industry_batch(tmp_path, source_ids):
    root = tmp_path / "industry_membership"
    root.mkdir(parents=True, exist_ok=True)
    symbols = ["000001", "600000", "000002"]
    rows = []
    for index, symbol in enumerate(symbols):
        rows.append({
            "symbol": symbol,
            "industry_code": "EM::银行" if index < 2 else "EM::房地产",
            "industry_name": "银行" if index < 2 else "房地产",
            "snapshot_id": "single-complete-batch",
            "expected_industry_count": 2,
            "observed_at": "2026-11-13T12:00:00+00:00",
            "available_time": "2026-11-13T12:00:00+00:00",
            "availability_basis": "retrieval_time_only",
            "historical_pit_eligible": False,
            "source_id": source_ids[index],
            "snapshot_scope": "current_membership_observation_only",
            "source_revision": f"revision-{index}",
        })
    (root / "2026-11-13.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_forward_industry_audit_accepts_baostock_taxonomy(tmp_path):
    _write_forward_industry_batch(
        tmp_path,
        ["baostock.query_stock_industry"] * 3,
    )
    selected, audit = fhm100.audit_forward_industry_membership(
        tmp_path,
        ["000001", "600000", "000002"],
        decision_time=pd.Timestamp("2026-11-13T13:00:00Z"),
    )
    assert audit["status"] == "ready_forward_only"
    assert audit["selected_symbol_count"] == 3
    assert set(selected["source_id"]) == {"baostock.query_stock_industry"}
    assert audit["historical_pit_eligible"] is False


def test_forward_industry_audit_accepts_one_combined_eastmoney_batch(tmp_path):
    _write_forward_industry_batch(
        tmp_path,
        [
            "akshare.stock_board_industry_cons_em",
            "akshare.stock_individual_info_em",
            "akshare.stock_board_industry_cons_em",
        ],
    )
    selected, audit = fhm100.audit_forward_industry_membership(
        tmp_path,
        ["000001", "600000", "000002"],
        decision_time=pd.Timestamp("2026-11-13T13:00:00Z"),
    )
    assert audit["status"] == "ready_forward_only"
    assert audit["selected_symbol_count"] == 3
    assert set(selected["source_id"]) == {
        "akshare.stock_board_industry_cons_em",
        "akshare.stock_individual_info_em",
    }
    assert audit["historical_pit_eligible"] is False
