import pandas as pd

from src.model.live_decision_integration import (
    aggregate_archived_stock_turnover, build_live_decision_gate_audit,
)


def _inputs():
    watchlist = pd.DataFrame([{"代码": "600000", "最新价": 20.0, "当日成交额": 1e9}])
    m5 = pd.DataFrame([
        {"symbol": "600000", "strategy": strategy, "score": 90.0, "eligible": True}
        for strategy in ("A", "B", "C")
    ])
    market = {"status": "READY", "market_state": "RISK_ON", "position_scale": 1.0}
    mainline = pd.DataFrame([{
        "industry_code": "801010", "trade_date": "2026-11-13",
        "mainline_phase": "IGNITION", "persistence_ready": True,
        "historical_pit_eligible": False, "stock_turnover_pit_status": "not_ready",
        "stock_turnover_live_status": "ready",
        "available_time": "2026-11-14T00:30:00Z",
        "source_id": "sw-industry-api", "source_revision": "rev-main-1",
        "availability_basis": "retrieval_time_only", "snapshot_scope": "live_industry_cross_section",
    }])
    membership = pd.DataFrame([{
        "symbol": "600000", "industry_code": "801010",
        "historical_pit_eligible": False,
        "availability_basis": "retrieval_time_only",
        "available_time": "2026-11-14T00:30:00Z",
        "source_id": "sw-membership-api", "source_revision": "rev-member-1",
        "snapshot_scope": "current_membership_observation",
        "valid_from": "2020-01-01",
        "valid_to": None,
    }])
    risk = {"600000": {
        "entry_price": 20.0, "stop_price": 19.0, "quantity": 500,
        "planned_risk": 600.0, "reward_risk": 2.0, "minimum_reward_risk": 1.5,
        "existing_portfolio_risk": 0.0, "existing_position_count": 0,
        "available_risk_budget": 1000.0,
        "same_industry_or_corr_gt_06": False,
    }}
    return watchlist, m5, market, mainline, membership, risk


def test_all_four_gates_can_authorize_only_with_complete_inputs():
    watchlist, m5, market, mainline, membership, risk = _inputs()
    rows, audit = build_live_decision_gate_audit(
        watchlist, m5, market, mainline, membership,
        decision_time="2026-11-14T09:00:00+08:00",
        risk_context_by_symbol=risk,
        minute_confirmation_fetcher=lambda symbol, captured_at=None: {
            "status": "CONFIRMED_15M_BREAKOUT", "entry_ready": True,
            "reason": "all_15m_checks_passed",
        },
    )
    assert rows["allowed"].all()
    assert audit["allowed_count"] == 3
    assert audit["formal_signal_emitted"] is False


def test_current_diagnostic_mainline_and_missing_m8_inputs_block_all_entries():
    watchlist, m5, market, mainline, membership, risk = _inputs()
    mainline["stock_turnover_live_status"] = "not_ready"
    mainline["source_revision"] = ""
    rows, audit = build_live_decision_gate_audit(
        watchlist, m5, market, mainline, membership,
        decision_time="2026-11-14T09:00:00+08:00",
        minute_confirmation_fetcher=lambda symbol, captured_at=None: {
            "status": "CONFIRMED_15M_BREAKOUT", "entry_ready": True,
            "reason": "all_15m_checks_passed",
        },
    )
    assert not rows["allowed"].any()
    assert rows["reasons"].map(lambda x: "industry_live_snapshot_not_ready" in x or "stock_turnover_live_not_ready" in x).all()
    assert audit["blocked_count"] == 3


def test_forward_only_membership_snapshot_can_authorize_live_but_never_pit():
    watchlist, m5, market, mainline, membership, risk = _inputs()
    membership["historical_pit_eligible"] = False
    membership["availability_basis"] = "retrieval_time_only"
    rows, audit = build_live_decision_gate_audit(
        watchlist, m5, market, mainline, membership,
        decision_time="2026-11-14T09:00:00+08:00",
        risk_context_by_symbol=risk,
        minute_confirmation_fetcher=lambda symbol, captured_at=None: {
            "status": "CONFIRMED_15M_BREAKOUT", "entry_ready": True,
            "reason": "all_15m_checks_passed",
        },
    )
    assert rows["allowed"].all()
    assert rows["historical_pit_eligible"].eq(False).all()
    assert rows["industry_live_data_eligible"].eq(True).all()
    assert audit["decision_mode"] == "live_forward"


def test_stale_or_future_live_industry_snapshots_are_rejected():
    for available_time in ("2026-11-01T00:00:00Z", "2026-11-14T02:00:01Z"):
        watchlist, m5, market, mainline, membership, risk = _inputs()
        membership["available_time"] = available_time
        rows, _ = build_live_decision_gate_audit(
            watchlist, m5, market, mainline, membership,
            decision_time="2026-11-14T09:00:00+08:00", risk_context_by_symbol=risk,
            minute_confirmation_fetcher=lambda symbol, **kwargs: {
                "status": "CONFIRMED_15M_BREAKOUT", "entry_ready": True,
            },
        )
        assert not rows["allowed"].any()


def test_15m_fetcher_receives_the_frozen_decision_cutoff():
    watchlist, m5, market, mainline, membership, risk = _inputs()
    seen = []

    def fetcher(symbol, *, captured_at=None):
        seen.append((symbol, pd.Timestamp(captured_at)))
        return {
            "status": "CONFIRMED_15M_BREAKOUT", "entry_ready": True,
            "reason": "all_15m_checks_passed",
        }

    build_live_decision_gate_audit(
        watchlist, m5, market, mainline, membership,
        decision_time="2026-11-14T09:00:00+08:00",
        risk_context_by_symbol=risk,
        minute_confirmation_fetcher=fetcher,
    )
    assert seen == [("600000", pd.Timestamp("2026-11-14T09:00:00+08:00"))]


def test_expired_membership_interval_cannot_authorize_entry():
    watchlist, m5, market, mainline, membership, risk = _inputs()
    membership["valid_to"] = "2026-11-01"
    rows, audit = build_live_decision_gate_audit(
        watchlist, m5, market, mainline, membership,
        decision_time="2026-11-14T09:00:00+08:00",
        risk_context_by_symbol=risk,
        minute_confirmation_fetcher=lambda symbol, captured_at=None: {
            "status": "CONFIRMED_15M_BREAKOUT", "entry_ready": True,
            "reason": "all_15m_checks_passed",
        },
    )
    assert not rows["allowed"].any()
    assert rows["reasons"].map(
        lambda reasons: "industry_live_snapshot_not_ready" in reasons
    ).all()


def test_missing_15m_confirmation_blocks_entry_even_when_other_inputs_are_ready():
    watchlist, m5, market, mainline, membership, risk = _inputs()
    rows, audit = build_live_decision_gate_audit(
        watchlist, m5, market, mainline, membership,
        decision_time="2026-11-14T09:00:00+08:00",
        risk_context_by_symbol=risk,
        minute_confirmation_fetcher=lambda symbol, captured_at=None: {
            "status": "NO_15M_CONFIRMATION", "entry_ready": False,
            "reason": "no_breakout",
        },
    )
    assert not rows["allowed"].any()
    assert rows["reasons"].map(lambda x: "entry_not_confirmed" in x).all()


def test_archived_retrieval_only_stock_bars_never_become_historical_pit(tmp_path):
    from pathlib import Path

    root = Path(tmp_path)
    (root / "stock_daily").mkdir()
    (root / "industry_membership").mkdir()
    pd.DataFrame([{
        "symbol": "600000", "trade_date": "2026-11-13", "amount": 1000000.0,
        "available_time": "2026-11-14T00:00:00Z",
        "historical_pit_eligible": False,
        "availability_basis": "retrieval_time_only",
    }]).to_json(root / "stock_daily" / "600000.jsonl", orient="records", lines=True)
    pd.DataFrame([{
        "symbol": "600000", "industry_code": "801010",
        "valid_from": "2020-01-01", "valid_to": None,
        "available_time": "2026-11-14T00:00:00Z",
        "source_id": "test", "source_revision": "rev1",
        "snapshot_scope": "current_membership_observation_only",
        "historical_pit_eligible": False,
        "availability_basis": "retrieval_time_only",
    }]).to_json(
        root / "industry_membership" / "snapshot.jsonl",
        orient="records", lines=True,
    )
    result, audit = aggregate_archived_stock_turnover(
        root, ["600000"], trade_date="2026-11-13",
        decision_time="2026-11-14T08:00:00+08:00",
    )
    assert result.empty
    assert audit["status"] == "incomplete_pit_stock_bars"
    assert audit["stock_bar_coverage"] == 0.0
    assert audit["historical_pit_eligible"] is False



def test_15m_confirmation_scan_supports_all_candidates_without_fixed_cap():
    watchlist = pd.DataFrame([
        {"代码": "600000", "最新价": 20.0, "当日成交额": 1e9},
        {"代码": "000001", "最新价": 10.0, "当日成交额": 8e8},
        {"代码": "300001", "最新价": 30.0, "当日成交额": 7e8},
    ])
    seen = []

    def fetcher(symbol, *, captured_at=None):
        seen.append(symbol)
        return {"status": "CHECKED", "entry_ready": False, "reason": "test"}

    rows, audit = build_live_decision_gate_audit(
        watchlist, pd.DataFrame([
            {"symbol": symbol, "strategy": "A", "score": 90.0, "eligible": True}
            for symbol in ("600000", "000001", "300001")
        ]), {"status": "UNKNOWN"}, pd.DataFrame(),
        pd.DataFrame(), decision_time="2026-11-14T09:00:00+08:00",
        minute_confirmation_fetcher=fetcher, minute_scan_limit=None,
    )

    assert seen == ["600000", "000001", "300001"]
    assert audit["minute_scanned_count"] == 3
    assert audit["minute_scan_limit"] == "ALL"




def test_15m_scan_skips_symbols_without_any_m5_eligible_strategy():
    watchlist = pd.DataFrame([
        {"代码": "600000", "当日成交额": 1e9},
        {"代码": "000001", "当日成交额": 8e8},
    ])
    m5 = pd.DataFrame([
        {"symbol": "600000", "strategy": "A", "score": 90.0, "eligible": True},
        {"symbol": "000001", "strategy": "A", "score": 70.0, "eligible": False},
    ])
    seen = []
    rows, audit = build_live_decision_gate_audit(
        watchlist, m5, {"status": "UNKNOWN"}, pd.DataFrame(), pd.DataFrame(),
        decision_time="2026-11-14T09:00:00+08:00",
        minute_confirmation_fetcher=lambda symbol, **kwargs: (
            seen.append(symbol) or {"status": "CHECKED", "entry_ready": False, "reason": "test"}
        ),
        minute_scan_limit=None,
    )
    assert seen == ["600000"]
    assert audit["m5_eligible_symbol_count"] == 1
    assert audit["minute_scanned_count"] == 1
    blocked = rows.loc[rows["symbol"] == "000001"]
    assert blocked["minute_reason"].eq("m5_strategy_not_eligible").all()

def test_15m_full_universe_uses_bounded_parallel_workers(monkeypatch):
    from src.model import live_decision_integration as integration

    calls = {}

    class RecordingExecutor:
        def __init__(self, max_workers):
            calls["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, fn, values):
            return map(fn, values)

    monkeypatch.setattr(integration, "ThreadPoolExecutor", RecordingExecutor)
    watchlist = pd.DataFrame([
        {"代码": "600000", "当日成交额": 100},
        {"代码": "000001", "当日成交额": 90},
    ])
    m5 = pd.DataFrame([
        {"symbol": "600000", "strategy": "A", "eligible": True, "score": 72},
        {"symbol": "000001", "strategy": "A", "eligible": True, "score": 68},
    ])
    rows, audit = integration.build_live_decision_gate_audit(
        watchlist, m5, {"status": "UNKNOWN"},
        pd.DataFrame(), pd.DataFrame(),
        decision_time="2026-11-14T09:00:00+08:00",
        minute_confirmation_fetcher=lambda symbol, **kwargs: {
            "status": "CHECKED", "entry_ready": False
        },
        minute_scan_limit=None, minute_scan_workers=99,
    )

    assert calls["max_workers"] == 8
    assert audit["minute_scanned_count"] == 2
    assert audit["minute_scan_limit"] == "ALL"

def test_missing_m8_plan_does_not_substitute_latest_quote_as_entry_price():
    watchlist, m5, market, mainline, membership, risk = _inputs()
    rows, audit = build_live_decision_gate_audit(
        watchlist, m5, market, mainline, membership,
        decision_time="2026-11-14T09:00:00+08:00",
        minute_confirmation_fetcher=lambda symbol, captured_at=None: {
            "status": "CONFIRMED_15M_BREAKOUT", "entry_ready": True,
            "reason": "all_15m_checks_passed",
        },
    )
    assert not rows["allowed"].any()
    assert rows["m8_risk_context_complete"].eq(False).all()
    assert rows["reasons"].map(lambda x: "invalid_entry_price" in x).all()
    assert audit["m8_risk_context_complete_symbol_count"] == 0
    assert audit["m8_risk_context_missing_symbol_count"] == 1



def test_operator_risk_context_file_loads_explicit_per_symbol_plan(tmp_path):
    import json
    from src.model.live_decision_integration import load_risk_context_file

    path = tmp_path / "risk.json"
    path.write_text(json.dumps({
        "600000": {
            "entry_price": 20.0, "stop_price": 19.0, "quantity": 100,
            "planned_risk": 120.0, "reward_risk": 2.0,
            "minimum_reward_risk": 1.5, "existing_portfolio_risk": 0.0,
            "existing_position_count": 0, "available_risk_budget": 500.0,
        }
    }), encoding="utf-8")
    plans, audit = load_risk_context_file(str(path))
    assert "600000" in plans
    assert plans["600000"]["available_risk_budget"] == 500.0
    assert audit["status"] == "LOADED"
    assert load_risk_context_file(None)[1]["status"] == "NOT_CONFIGURED"
