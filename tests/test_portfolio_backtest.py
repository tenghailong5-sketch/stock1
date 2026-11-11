from datetime import date

import pandas as pd
import pytest

from src.model.execution import ExecutionCosts
from src.model.portfolio_backtest import _risk, run_portfolio_backtest


DECISION = pd.Timestamp("2026-01-05T15:00:00+08:00")
AVAILABLE = pd.Timestamp("2026-01-05T14:59:00+08:00")


def candidate(symbol="000001", score=90.0, stop=9.5, strategy="A"):
    return {
        "symbol": symbol, "decision_time": DECISION, "available_time": AVAILABLE,
        "historical_pit_eligible": True, "source_revision": "candidate-r1-" + symbol + strategy,
        "strategy": strategy, "score": score, "eligible": True,
        "corporate_actions_resolved": True, "universe_complete": True,
        "delisted_included": True, "ruleset_version": "FHM-100-test",
        "stop_price": stop,
    }


def bar(symbol, day, *, open_=10.0, high=10.2, low=9.8, close=10.0,
        volume=100000, limit_down=False):
    return {
        "symbol": symbol, "trade_date": day,
        "available_time": pd.Timestamp(str(day) + "T15:01:00+08:00"),
        "historical_pit_eligible": True, "source_revision": f"bar-{symbol}-{day}",
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        "suspended": False, "limit_up_locked": False, "limit_down_locked": limit_down,
        "corporate_actions_resolved": True, "universe_complete": True,
        "delisted_included": True,
    }


def frame(rows):
    return pd.DataFrame(rows)


def test_signal_at_close_enters_next_session_and_stops_next_day():
    candidates = frame([candidate()])
    bars = frame([
        bar("000001", "2026-01-06", open_=10.0, high=10.2, low=9.8, close=10.1),
        bar("000001", "2026-01-07", open_=9.4, high=9.6, low=9.2, close=9.3),
    ])
    result = run_portfolio_backtest(candidates, bars)
    assert result["audit_passed"]
    assert result["fills"][0]["date"] == "2026-01-06"
    assert result["fills"][0]["side"] == "buy"
    assert result["trades"][0]["entry_date"] == "2026-01-06"
    assert result["trades"][0]["exit_date"] == "2026-01-07"
    assert result["trades"][0]["exit_reason"] == "protective_stop"
    assert result["ending_equity"] < result["initial_cash"]
    assert result["metrics"]["trade_count"] == 1
    assert result["metrics"]["losses"] == 1
    assert result["metrics"]["max_drawdown_pct"] > 0


def test_t1_prevents_same_session_stop_exit():
    candidates = frame([candidate()])
    bars = frame([
        bar("000001", "2026-01-06", open_=10.0, high=10.2, low=9.0, close=9.2),
        bar("000001", "2026-01-07", open_=9.4, high=9.6, low=9.2, close=9.3),
    ])
    result = run_portfolio_backtest(candidates, bars)
    assert result["fills"][0]["side"] == "buy"
    assert len(result["trades"]) == 1
    assert result["trades"][0]["exit_date"] == "2026-01-07"


def test_locked_limit_down_keeps_stopped_position_open():
    candidates = frame([candidate()])
    bars = frame([
        bar("000001", "2026-01-06"),
        bar("000001", "2026-01-07", open_=9.4, high=9.5, low=9.2, close=9.3, limit_down=True),
    ])
    result = run_portfolio_backtest(candidates, bars)
    assert len(result["trades"]) == 0
    assert result["open_positions"][0]["symbol"] == "000001"
    assert result["fills"][-1]["reason"] == "limit_down_locked_no_sell_fill"


def test_triggered_stop_retries_after_limit_down_block_even_if_price_rebounds():
    candidates = frame([candidate()])
    bars = frame([
        bar("000001", "2026-01-06"),
        bar("000001", "2026-01-07", open_=9.4, high=9.5, low=9.2, close=9.3, limit_down=True),
        bar("000001", "2026-01-08", open_=9.8, high=10.0, low=9.7, close=9.9),
    ])
    result = run_portfolio_backtest(candidates, bars)
    assert len(result["trades"]) == 1
    assert result["trades"][0]["exit_date"] == "2026-01-08"
    assert result["fills"][-1]["side"] == "sell"
    assert result["fills"][-1]["filled"] is True
    # The pending market exit uses the next session's opening price, not
    # the old stop trigger price, when the market has rebounded.
    assert result["fills"][-1]["fill_price"] > 9.5


def test_partial_stop_fills_remain_pending_and_are_not_counted_as_completed():
    candidates = frame([candidate(stop=9.5)])
    bars = frame([
        bar("000001", "2026-01-06", volume=100000),
        bar("000001", "2026-01-07", open_=9.4, high=9.5, low=9.2, close=9.3, volume=2000),
        bar("000001", "2026-01-08", open_=9.3, high=9.4, low=9.1, close=9.2, volume=2000),
        bar("000001", "2026-01-09", open_=9.2, high=9.3, low=9.0, close=9.1, volume=2000),
    ])
    result = run_portfolio_backtest(candidates, bars)
    sell_fills = [fill for fill in result["fills"] if fill["side"] == "sell" and fill["filled"]]
    assert len(sell_fills) >= 2
    assert all(fill["quantity"] == 100 for fill in sell_fills)
    assert result["open_positions"][0]["stop_exit_pending"] is True
    assert result["metrics"]["trade_count"] == 0
    assert result["metrics"]["win_rate_pct"] is None


def test_partial_exit_fills_count_as_one_trade_only_after_position_closes():
    candidates = frame([candidate(stop=9.5)])
    bars = frame([
        bar("000001", "2026-01-06", volume=100000),
        bar("000001", "2026-01-07", open_=9.4, high=9.5, low=9.2, close=9.3, volume=2000),
        bar("000001", "2026-01-08", open_=9.3, high=9.4, low=9.1, close=9.2, volume=2000),
        bar("000001", "2026-01-09", open_=9.2, high=9.3, low=9.0, close=9.1, volume=2000),
        bar("000001", "2026-01-12", open_=9.0, high=9.1, low=8.9, close=9.0, volume=100000),
    ])
    result = run_portfolio_backtest(candidates, bars)
    assert result["open_positions"] == []
    assert len([f for f in result["fills"] if f["side"] == "sell" and f["filled"]]) >= 4
    assert result["metrics"]["trade_count"] == 1
    assert result["metrics"]["losses"] == 1



def test_retrieval_time_only_dataset_is_blocked_before_simulation():
    candidates = frame([candidate()])
    bars = frame([bar("000001", "2026-01-06")])
    bars.loc[0, "historical_pit_eligible"] = False
    with pytest.raises(ValueError, match="historical backtest blocked"):
        run_portfolio_backtest(candidates, bars)


def test_entry_is_skipped_when_stop_is_not_below_slippage_adjusted_entry():
    candidates = frame([candidate(stop=10.02)])
    bars = frame([bar("000001", "2026-01-06", open_=10.0)])
    result = run_portfolio_backtest(candidates, bars)
    assert result["fills"] == []
    assert result["skipped"][0]["reason"] == "stop_not_below_entry"


def test_duplicate_strategy_signals_do_not_double_buy_same_symbol():
    candidates = frame([
        candidate(symbol="000001", score=91.0, strategy="A"),
        candidate(symbol="000001", score=90.0, strategy="C"),
    ])
    bars = frame([bar("000001", "2026-01-06")])
    result = run_portfolio_backtest(candidates, bars)
    assert len([f for f in result["fills"] if f["side"] == "buy"]) == 1


def test_invalid_volume_fraction_is_rejected():
    candidates = frame([candidate()])
    bars = frame([bar("000001", "2026-01-06", volume=100.5)])
    with pytest.raises(ValueError, match="bar volume"):
        run_portfolio_backtest(candidates, bars)



def test_portfolio_never_opens_more_than_two_positions():
    candidates = frame([
        candidate(symbol="000001", score=93.0),
        candidate(symbol="000002", score=92.0),
        candidate(symbol="000003", score=91.0),
    ])
    bars = frame([
        bar("000001", "2026-01-06"),
        bar("000002", "2026-01-06"),
        bar("000003", "2026-01-06"),
    ])
    result = run_portfolio_backtest(candidates, bars)
    buys = [fill for fill in result["fills"] if fill["side"] == "buy" and fill["filled"]]
    assert len(buys) == 2
    assert len(result["open_positions"]) == 2
    assert result["skipped"][-1]["reason"] == "position_sizing_rejected"


def test_open_position_risk_includes_paid_buy_fees_and_minimum_sell_commission():
    position = {
        "shares": 100, "entry_price": 10.0, "stop_price": 9.0,
        "buy_cost": 5.5,
    }
    risk = _risk(position, ExecutionCosts())
    assert risk > 111.0
    assert risk == pytest.approx(100 * (10.0 - 9.0 * 0.999)
                                 + 5.5 + 5.0
                                 + 100 * 9.0 * 0.999 * (5.0 + 0.1) / 10000)


def test_custom_execution_costs_reduce_position_size_conservatively():
    candidates = frame([candidate(stop=9.0)])
    bars = frame([
        bar("000001", "2026-01-06", open_=10.0, high=10.2, low=9.8, close=10.0),
        bar("000001", "2026-01-07", open_=10.0, high=10.2, low=9.8, close=10.0),
    ])
    low_cost = run_portfolio_backtest(candidates, bars, costs=ExecutionCosts())
    high_costs = ExecutionCosts(commission_bps=10, minimum_commission=10,
                                sell_stamp_duty_bps=10, slippage_bps=30,
                                transfer_fee_bps=1)
    high_cost = run_portfolio_backtest(candidates, bars, costs=high_costs)
    assert low_cost["open_positions"][0]["shares"] > high_cost["open_positions"][0]["shares"]
