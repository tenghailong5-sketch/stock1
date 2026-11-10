from datetime import date

import pandas as pd
import pytest

from src.model.portfolio_backtest import run_portfolio_backtest


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


def test_retrieval_time_only_dataset_is_blocked_before_simulation():
    candidates = frame([candidate()])
    bars = frame([bar("000001", "2026-01-06")])
    bars.loc[0, "historical_pit_eligible"] = False
    with pytest.raises(ValueError, match="historical backtest blocked"):
        run_portfolio_backtest(candidates, bars)


def test_entry_is_skipped_when_stop_is_not_below_next_open():
    candidates = frame([candidate(stop=10.0)])
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
