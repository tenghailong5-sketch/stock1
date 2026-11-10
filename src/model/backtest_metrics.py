"""Validated performance statistics for the FHM-100 portfolio simulator."""
from __future__ import annotations

import math
from numbers import Real
from typing import Any

import pandas as pd


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _timestamp(value: Any, field: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a valid timestamp") from exc
    if pd.isna(result):
        raise ValueError(f"{field} must be a valid timestamp")
    return result


def calculate_backtest_metrics(
    daily_equity: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    initial_cash: float,
) -> dict[str, Any]:
    """Calculate equity-curve drawdown and completed-trade statistics.

    Win rate uses all completed trade records as its denominator, including
    break-even records. Profit factor and payoff ratio are undefined (None)
    when there are no losing trades; missing statistics are never fabricated.
    """
    starting_capital = _finite_number(initial_cash, "initial_cash")
    if starting_capital <= 0:
        raise ValueError("initial_cash must be positive")
    if not isinstance(daily_equity, list) or not daily_equity:
        raise ValueError("daily_equity must be a non-empty list")
    if not isinstance(trades, list):
        raise ValueError("trades must be a list")

    points: list[tuple[pd.Timestamp, float]] = []
    for index, item in enumerate(daily_equity):
        if not isinstance(item, dict) or "date" not in item or "equity" not in item:
            raise ValueError(f"daily_equity[{index}] requires date and equity")
        stamp = _timestamp(item["date"], f"daily_equity[{index}].date")
        equity = _finite_number(item["equity"], f"daily_equity[{index}].equity")
        if equity < 0:
            raise ValueError("daily equity cannot be negative")
        if points and stamp <= points[-1][0]:
            raise ValueError("daily_equity dates must be strictly increasing")
        points.append((stamp, equity))

    # Initial capital is the equity high-water mark before the first session.
    # If the first observed equity is already below it, the peak predates the
    # dataset and therefore has no fabricated calendar date.
    peak = starting_capital
    peak_date = points[0][0] if points[0][1] >= starting_capital else None
    max_drawdown = 0.0
    max_drawdown_peak_date = peak_date
    max_drawdown_trough_date = points[0][0]
    for stamp, equity in points:
        if equity > peak:
            peak = equity
            peak_date = stamp
        drawdown = 0.0 if peak == 0 else (peak - equity) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_peak_date = peak_date
            max_drawdown_trough_date = stamp

    # A position may be liquidated through multiple partial fills. Aggregate
    # those fills by symbol + entry date so win rate and streaks count round trips,
    # not individual execution fragments. Generic trade rows without those keys
    # remain independent completed trades.
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    independent: list[tuple[pd.Timestamp, int, float]] = []
    for index, trade in enumerate(trades):
        if not isinstance(trade, dict) or "realized_pnl" not in trade or "exit_date" not in trade:
            raise ValueError(f"trades[{index}] requires exit_date and realized_pnl")
        stamp = _timestamp(trade["exit_date"], f"trades[{index}].exit_date")
        pnl = _finite_number(trade["realized_pnl"], f"trades[{index}].realized_pnl")
        if trade.get("symbol") is not None and trade.get("entry_date") is not None:
            key = (str(trade["symbol"]), str(_timestamp(trade["entry_date"], f"trades[{index}].entry_date")))
            if key not in grouped:
                grouped[key] = {"exit_date": stamp, "pnl": pnl, "first_index": index}
            else:
                grouped[key]["exit_date"] = max(grouped[key]["exit_date"], stamp)
                grouped[key]["pnl"] += pnl
        else:
            independent.append((stamp, index, pnl))
    completed = independent + [
        (item["exit_date"], item["first_index"], item["pnl"]) for item in grouped.values()
    ]
    completed.sort(key=lambda item: (item[0], item[1]))

    pnls = [item[2] for item in completed]
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value < 0]
    breakeven_count = sum(value == 0 for value in pnls)
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    trade_count = len(pnls)

    max_loss_streak = 0
    current_loss_streak = 0
    for pnl in pnls:
        if pnl < 0:
            current_loss_streak += 1
            max_loss_streak = max(max_loss_streak, current_loss_streak)
        else:
            current_loss_streak = 0

    ending_equity = points[-1][1]
    return {
        "initial_cash": round(starting_capital, 2),
        "ending_equity": round(ending_equity, 2),
        "total_return_pct": round((ending_equity / starting_capital - 1.0) * 100.0, 4),
        "max_drawdown_pct": round(max_drawdown * 100.0, 4),
        "max_drawdown_peak_date": max_drawdown_peak_date.date().isoformat(),
        "max_drawdown_trough_date": max_drawdown_trough_date.date().isoformat(),
        "trade_count": trade_count,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven_trades": breakeven_count,
        "win_rate_pct": round(len(wins) / trade_count * 100.0, 4) if trade_count else None,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
        "average_win": round(sum(wins) / len(wins), 2) if wins else None,
        "average_loss": round(sum(losses) / len(losses), 2) if losses else None,
        "payoff_ratio": round((sum(wins) / len(wins)) / abs(sum(losses) / len(losses)), 4)
        if wins and losses else None,
        "expectancy_per_trade": round(sum(pnls) / trade_count, 2) if trade_count else None,
        "max_consecutive_losses": max_loss_streak,
        "realized_trade_pnl": round(sum(pnls), 2),
    }
