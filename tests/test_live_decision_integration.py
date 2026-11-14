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
        "historical_pit_eligible": True, "stock_turnover_pit_status": "ready",
    }])
    membership = pd.DataFrame([{
        "symbol": "600000", "industry_code": "801010",
        "historical_pit_eligible": True,
    }])
    risk = {"600000": {
        "entry_price": 20.0, "stop_price": 19.0, "quantity": 500,
        "planned_risk": 600.0, "reward_risk": 2.0, "minimum_reward_risk": 1.5,
        "existing_portfolio_risk": 0.0, "existing_position_count": 0,
        "same_industry_or_corr_gt_06": False,
    }}
    return watchlist, m5, market, mainline, membership, risk


def test_all_four_gates_can_authorize_only_with_complete_inputs():
    watchlist, m5, market, mainline, membership, risk = _inputs()
    rows, audit = build_live_decision_gate_audit(
        watchlist, m5, market, mainline, membership,
        decision_time="2026-11-14T09:00:00+08:00",
        risk_context_by_symbol=risk,
        minute_confirmation_fetcher=lambda symbol: {
            "status": "CONFIRMED_15M_BREAKOUT", "entry_ready": True,
            "reason": "all_15m_checks_passed",
        },
    )
    assert rows["allowed"].all()
    assert audit["allowed_count"] == 3
    assert audit["formal_signal_emitted"] is False


def test_current_diagnostic_mainline_and_missing_m8_inputs_block_all_entries():
    watchlist, m5, market, mainline, membership, risk = _inputs()
    mainline["historical_pit_eligible"] = False
    mainline["stock_turnover_pit_status"] = "not_ready"
    rows, audit = build_live_decision_gate_audit(
        watchlist, m5, market, mainline, membership,
        decision_time="2026-11-14T09:00:00+08:00",
        minute_confirmation_fetcher=lambda symbol: {
            "status": "CONFIRMED_15M_BREAKOUT", "entry_ready": True,
            "reason": "all_15m_checks_passed",
        },
    )
    assert not rows["allowed"].any()
    assert rows["reasons"].map(lambda x: "industry_history_not_pit_eligible" in x).all()
    assert audit["blocked_count"] == 3


def test_missing_15m_confirmation_blocks_entry_even_when_other_inputs_are_ready():
    watchlist, m5, market, mainline, membership, risk = _inputs()
    rows, audit = build_live_decision_gate_audit(
        watchlist, m5, market, mainline, membership,
        decision_time="2026-11-14T09:00:00+08:00",
        risk_context_by_symbol=risk,
        minute_confirmation_fetcher=lambda symbol: {
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
