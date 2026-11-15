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



def test_forward_industry_audit_accepts_official_sw2021_snapshot(tmp_path):
    _write_forward_industry_batch(
        tmp_path,
        ["swsresearch.sw2021_stock_classification"] * 3,
    )
    selected, audit = fhm100.audit_forward_industry_membership(
        tmp_path,
        ["000001", "600000", "000002"],
        decision_time=pd.Timestamp("2026-11-13T13:00:00Z"),
    )
    assert audit["status"] == "ready_forward_only"
    assert audit["selected_symbol_count"] == 3
    assert set(selected["source_id"]) == {"swsresearch.sw2021_stock_classification"}
    assert audit["historical_pit_eligible"] is False



def test_m5_maps_observed_industry_structure_and_trade_quality_without_filling_unknowns():
    watchlist = pd.DataFrame([
        {
            "代码": "000001", "sw_history_industry_code": "801010",
            "CLV": 0.9, "20日平均成交额": 200_000_000,
        },
        {
            "代码": "000002", "sw_history_industry_code": "801020",
            "CLV": 0.4, "20日平均成交额": 80_000_000,
        },
    ])
    mainline = pd.DataFrame([
        {
            "industry_code": "801010", "trade_date": "2026-11-13",
            "strength_percentile": 0.92,
        },
        {
            "industry_code": "801020", "trade_date": "2026-11-13",
            "strength_percentile": 0.35,
        },
    ])
    market = {
        "market_state": "RISK_ON",
        "market_environment_score": 88.0,
    }

    frame = fhm100.build_m5_diagnostic_input(watchlist, market, mainline)
    first = frame.set_index("symbol").loc["000001"]
    second = frame.set_index("symbol").loc["000002"]

    assert first["industry_relative_strength"] == 92.0
    assert first["industry_strength"] == 92.0
    assert bool(first["industry_top_30pct"])
    assert bool(first["amount_ma20_ge_100m"])
    assert first["market_environment"] == 88.0
    assert first["m5_factor_scope"] == "bounded_live_research_pool_percentile"
    assert bool(second["industry_not_weak"])
    assert not bool(second["amount_ma20_ge_100m"])

    # No fabricated cross-session residual, event, pullback, or risk inputs.
    for key in (
        "stock_residual_top_20pct", "breakout_or_confirmed_retest",
        "no_high_volume_stall_or_repeated_rejection",
        "pre_pullback_residual_positive", "pullback_between_0_5_and_4pct",
        "pullback_volume_le_80pct_prior5d", "stop_or_reversal_confirmed",
        "no_material_negative_event", "liquidity_and_risk_pass",
    ):
        assert frame[key].isna().all()


def test_m5_maps_only_lineage_valid_pit_factor_snapshot():
    watchlist = pd.DataFrame([{"代码": "000001", "CLV": 0.8, "20日平均成交额": 150_000_000}])
    snapshot = pd.DataFrame([{
        "symbol": "000001", "trade_date": "2026-11-13",
        "available_time": "2026-11-13T08:00:00Z",
        "source_id": "pit.factor_builder", "source_revision": "sha256:abc",
        "historical_pit_eligible": True,
        "industry_relative_strength_pct": 91.0,
        "stock_residual_strength_pct": 88.0,
        "price_structure_pct": 82.0, "trade_quality_pct": 76.0,
        "pre_pullback_strength_pct": 85.0,
        "pullback_reversal_structure_pct": 79.0,
        "industry_top_30pct": True, "stock_residual_top_20pct": True,
        "breakout_or_confirmed_retest": True,
        "no_high_volume_stall_or_repeated_rejection": True,
        "pre_pullback_residual_positive": True,
        "pullback_between_0_5_and_4pct": True,
        "pullback_volume_le_80pct_prior5d": True,
        "stop_or_reversal_confirmed": True, "industry_not_weak": True,
        "no_material_negative_event": True, "liquidity_and_risk_pass": True,
    }])
    frame = fhm100.build_m5_diagnostic_input(
        watchlist, {"market_state": "RISK_ON", "market_environment_score": 90.0},
        pit_factor_snapshot=snapshot,
        decision_time="2026-11-14T09:00:00+08:00",
    )
    row = frame.iloc[0]
    assert row["industry_relative_strength"] == 91.0
    assert row["stock_residual_strength"] == 88.0
    assert row["price_structure"] == 82.0
    assert bool(row["stock_residual_top_20pct"])
    assert row["m5_pit_factor_provenance"] == "verified_pit_snapshot"


def test_m5_rejects_future_or_retrieval_only_pit_factor_rows():
    watchlist = pd.DataFrame([{"代码": "000001"}])
    snapshot = pd.DataFrame([{
        "symbol": "000001", "trade_date": "2026-11-13",
        "available_time": "2026-11-15T00:00:00Z",
        "source_id": "pit.factor_builder", "source_revision": "rev1",
        "historical_pit_eligible": True,
        "industry_relative_strength_pct": 99.0,
    }])
    frame = fhm100.build_m5_diagnostic_input(
        watchlist, {"market_state": "RISK_ON"},
        pit_factor_snapshot=snapshot,
        decision_time="2026-11-14T09:00:00+08:00",
    )
    assert pd.isna(frame.loc[0, "industry_relative_strength"])
    assert frame.loc[0, "m5_pit_factor_provenance"] == "missing_or_ineligible_pit_snapshot"


def test_m5_industry_factor_fails_closed_on_ambiguous_latest_industry_rows():
    watchlist = pd.DataFrame([
        {
            "代码": "000001", "sw_history_industry_code": "801010",
            "CLV": 0.8, "20日平均成交额": 150_000_000,
        },
    ])
    mainline = pd.DataFrame([
        {
            "industry_code": "801010", "trade_date": "2026-11-13",
            "strength_percentile": 0.9,
        },
        {
            "industry_code": "801010", "trade_date": "2026-11-13",
            "strength_percentile": 0.95,
        },
    ])
    frame = fhm100.build_m5_diagnostic_input(
        watchlist, {"market_state": "RISK_ON", "market_environment_score": 80},
        mainline,
    )
    assert pd.isna(frame.loc[0, "industry_relative_strength"])
    assert frame.loc[0, "industry_top_30pct"] is None
    assert frame.loc[0, "industry_factor_provenance"] == "missing_or_ambiguous_industry_mapping"

def test_m5_diagnostic_input_handles_non_unique_watchlist_index():
    watchlist = pd.DataFrame([
        {
            "代码": "000001", "sw_history_industry_code": "801010",
            "CLV": 0.9, "20日平均成交额": 200_000_000,
        },
        {
            "代码": "000002", "sw_history_industry_code": "801020",
            "CLV": 0.4, "20日平均成交额": 80_000_000,
        },
    ], index=[7, 7])
    mainline = pd.DataFrame([
        {
            "industry_code": "801010", "trade_date": "2026-11-13",
            "strength_percentile": 0.92,
        },
        {
            "industry_code": "801020", "trade_date": "2026-11-13",
            "strength_percentile": 0.35,
        },
    ])

    frame = fhm100.build_m5_diagnostic_input(
        watchlist,
        {"market_state": "RISK_ON", "market_environment_score": 88.0},
        mainline,
    )

    assert frame["symbol"].tolist() == ["000001", "000002"]
    assert frame["industry_relative_strength"].tolist() == [92.0, 35.0]
    assert frame["price_structure"].tolist() == [100.0, 50.0]
    assert frame["trade_quality"].tolist() == [100.0, 50.0]

def test_m5_negative_policy_risk_blocks_only_explicitly_affected_industry():
    watchlist = pd.DataFrame({
        "代码": ["000001", "000002"],
        "sw_history_industry_code": ["801010", "801020"],
        "CLV": [0.8, 0.5],
        "20日平均成交额": [200_000_000, 120_000_000],
    })
    mainline = pd.DataFrame({
        "industry_code": ["801010", "801020"],
        "trade_date": ["2026-11-16", "2026-11-16"],
        "strength_percentile": [0.9, 0.7],
        "negative_policy_event_component": [True, pd.NA],
    })

    result = fhm100.build_m5_diagnostic_input(
        watchlist,
        {"market_state": "RISK_ON", "market_environment_score": 80.0},
        mainline,
    )

    assert result["no_material_negative_event"].tolist() == [False, None]

