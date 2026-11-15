from datetime import date

import pandas as pd
import pytest

import src.model.portfolio_backtest_runner as runner
from src.model.portfolio_backtest import run_portfolio_backtest
from src.model.run_checkpoint import CheckpointConflictError, SQLiteCheckpointStore


def _inputs():
    candidate = {
        "symbol": "000001",
        "decision_time": pd.Timestamp("2026-01-05T15:00:00+08:00"),
        "available_time": pd.Timestamp("2026-01-05T14:59:00+08:00"),
        "historical_pit_eligible": True,
        "source_revision": "candidate-r1",
        "strategy": "A",
        "score": 90.0,
        "eligible": True,
        "corporate_actions_resolved": True,
        "universe_complete": True,
        "delisted_included": True,
        "ruleset_version": "FHM-100-test",
        "stop_price": 9.5,
        "available_risk_budget": 5000.0,
        "market_risk_scale": 1.0,
        "m8_trade_allowed": True,
    }
    bars = []
    for day, open_, high, low, close in (
        ("2026-01-06", 10.0, 10.2, 9.8, 10.0),
        ("2026-01-07", 9.4, 9.5, 9.2, 9.3),
        ("2026-01-08", 9.8, 10.0, 9.7, 9.9),
    ):
        bars.append({
            "symbol": "000001",
            "trade_date": day,
            "available_time": pd.Timestamp(day + "T15:01:00+08:00"),
            "historical_pit_eligible": True,
            "source_revision": "bar-" + day,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100000,
            "suspended": False,
            "limit_up_locked": False,
            "limit_down_locked": False,
            "corporate_actions_resolved": True,
            "universe_complete": True,
            "delisted_included": True,
        })
    return pd.DataFrame([candidate]), pd.DataFrame(bars)


def _run_kwargs(tmp_path, run_id):
    return {
        "checkpoint_path": str(tmp_path / "resume.sqlite"),
        "run_id": run_id,
        "batch_size": 1,
    }


def test_interrupted_batch_is_retried_and_matches_continuous_run(tmp_path, monkeypatch):
    candidates, bars = _inputs()
    expected = run_portfolio_backtest(candidates, bars)
    original = runner.run_portfolio_backtest
    interrupted = {"raised": False}

    def fail_once_on_second_batch(*args, **kwargs):
        if kwargs.get("_date_start") == date(2026, 1, 7) and not interrupted["raised"]:
            interrupted["raised"] = True
            raise RuntimeError("simulated process interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "run_portfolio_backtest", fail_once_on_second_batch)
    kwargs = _run_kwargs(tmp_path, "interrupted-run")
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        runner.run_checkpointed_portfolio_backtest(candidates, bars, **kwargs)

    store = SQLiteCheckpointStore(kwargs["checkpoint_path"])
    partial = store.summary("interrupted-run", "portfolio_backtest")
    assert partial == {"total": 3, "pending": 1, "running": 0, "completed": 1, "failed": 1}

    monkeypatch.setattr(runner, "run_portfolio_backtest", original)
    resumed = runner.run_checkpointed_portfolio_backtest(candidates, bars, **kwargs)
    for key in (
        "ending_cash", "ending_equity", "total_return_pct", "realized_pnl",
        "unrealized_pnl", "open_positions", "fills", "trades", "skipped",
        "daily_equity", "metrics",
    ):
        assert resumed[key] == expected[key]
    assert resumed["checkpoint"]["completed"] == 3
    assert resumed["checkpoint"]["failed"] == 0


def test_changed_candidate_input_is_rejected_for_existing_run_id(tmp_path):
    candidates, bars = _inputs()
    kwargs = _run_kwargs(tmp_path, "changed-input")
    runner.run_checkpointed_portfolio_backtest(candidates, bars, **kwargs)

    changed = candidates.copy()
    changed["score"] = changed["score"].astype(object)
    changed.loc[0, "score"] = "90.0"
    with pytest.raises(CheckpointConflictError, match="run input fingerprint changed"):
        runner.run_checkpointed_portfolio_backtest(changed, bars, **kwargs)


def test_frame_fingerprint_distinguishes_numeric_and_string_values():
    numeric = pd.DataFrame({"value": [1]})
    string = pd.DataFrame({"value": ["1"]})
    assert runner._frame_fingerprint(numeric) != runner._frame_fingerprint(string)
