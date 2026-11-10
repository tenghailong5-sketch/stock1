import pytest

from src.model.backtest_metrics import calculate_backtest_metrics


def equity(*values):
    return [{"date": f"2026-01-0{i + 1}", "equity": value} for i, value in enumerate(values)]


def test_drawdown_uses_daily_equity_peak_to_trough():
    result = calculate_backtest_metrics(equity(100000, 110000, 99000, 105000), [], 100000)
    assert result["max_drawdown_pct"] == 10.0
    assert result["max_drawdown_peak_date"] == "2026-01-02"
    assert result["max_drawdown_trough_date"] == "2026-01-03"
    assert result["total_return_pct"] == 5.0


def test_trade_statistics_include_breakeven_in_win_rate_denominator():
    trades = [
        {"exit_date": "2026-01-03", "realized_pnl": 100},
        {"exit_date": "2026-01-04", "realized_pnl": -50},
        {"exit_date": "2026-01-05", "realized_pnl": 0},
    ]
    result = calculate_backtest_metrics(equity(100000, 100050), trades, 100000)
    assert result["trade_count"] == 3
    assert result["wins"] == 1
    assert result["losses"] == 1
    assert result["breakeven_trades"] == 1
    assert result["win_rate_pct"] == pytest.approx(33.3333)
    assert result["profit_factor"] == 2.0
    assert result["payoff_ratio"] == 2.0
    assert result["expectancy_per_trade"] == pytest.approx(16.67)


def test_loss_streak_is_chronological_and_break_even_resets_it():
    trades = [
        {"exit_date": "2026-01-04", "realized_pnl": -3},
        {"exit_date": "2026-01-02", "realized_pnl": -1},
        {"exit_date": "2026-01-03", "realized_pnl": 0},
        {"exit_date": "2026-01-05", "realized_pnl": -2},
        {"exit_date": "2026-01-06", "realized_pnl": -4},
    ]
    result = calculate_backtest_metrics(equity(100, 95), trades, 100)
    assert result["max_consecutive_losses"] == 2


def test_no_completed_trades_returns_explicit_undefined_statistics():
    result = calculate_backtest_metrics(equity(100, 105), [], 100)
    assert result["trade_count"] == 0
    assert result["win_rate_pct"] is None
    assert result["profit_factor"] is None
    assert result["payoff_ratio"] is None
    assert result["expectancy_per_trade"] is None
    assert result["max_consecutive_losses"] == 0


def test_all_winning_trades_have_undefined_profit_factor_without_losses():
    trades = [{"exit_date": "2026-01-02", "realized_pnl": 5}]
    result = calculate_backtest_metrics(equity(100, 105), trades, 100)
    assert result["profit_factor"] is None
    assert result["average_loss"] is None
    assert result["payoff_ratio"] is None


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "100"])
def test_invalid_equity_values_fail_closed(bad):
    with pytest.raises(ValueError):
        calculate_backtest_metrics([{"date": "2026-01-01", "equity": bad}], [], 100)


def test_negative_equity_and_non_monotonic_dates_are_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_backtest_metrics(equity(-1), [], 100)
    with pytest.raises(ValueError, match="strictly increasing"):
        calculate_backtest_metrics(
            [{"date": "2026-01-02", "equity": 100}, {"date": "2026-01-01", "equity": 90}],
            [], 100,
        )


def test_invalid_trade_pnl_is_rejected():
    with pytest.raises(ValueError, match="realized_pnl"):
        calculate_backtest_metrics(
            equity(100), [{"exit_date": "2026-01-02", "realized_pnl": float("nan")}], 100
        )
