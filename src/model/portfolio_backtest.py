"""Event-driven, long-only A-share portfolio simulation.

This is a research engine, not a broker or a claim of historical validity.
It refuses data that fails the point-in-time admission audit.
"""
from __future__ import annotations

from datetime import date
import math
from numbers import Real
from typing import Any

import pandas as pd

from .backtest_audit import require_backtest_ready
from .backtest_metrics import calculate_backtest_metrics
from .execution import ExecutionCosts, simulate_daily_fill
from .position_sizing import size_long_position


def _as_date(value: Any) -> date:
    return pd.Timestamp(value).date()


def _volume(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("bar volume must be a finite non-negative integer")
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError("bar volume must be a finite non-negative integer")
    return int(number)


def _risk(position: dict[str, Any]) -> float:
    shares = position["shares"]
    entry = position["entry_price"]
    stop = position["stop_price"]
    return max(0.0, shares * (entry - stop) + shares * entry * 30.0 / 10000.0)


def run_portfolio_backtest(
    candidates: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    initial_cash: float = 100000.0,
    per_trade_risk_limit: float = 2000.0,
    portfolio_risk_limit: float = 3000.0,
    max_positions: int = 2,
    max_participation: float = 0.05,
    costs: ExecutionCosts = ExecutionCosts(),
) -> dict[str, Any]:
    """Run close-signal/next-session-open entries with T+1 stop exits.

    Candidate rows require an explicit stop_price. The audit must certify all
    candidate and bar fields before any simulation is attempted. Candidates
    are ranked by score (descending), then symbol and strategy for stable ties.
    """
    if isinstance(initial_cash, bool) or not isinstance(initial_cash, Real) or not math.isfinite(float(initial_cash)) or initial_cash <= 0:
        raise ValueError("initial_cash must be finite and positive")
    if "stop_price" not in candidates.columns:
        raise ValueError("candidate_missing_stop_price")
    require_backtest_ready(candidates, bars)

    work = candidates.copy()
    work["_decision_date"] = work["decision_time"].map(_as_date)
    work["_score"] = pd.to_numeric(work["score"], errors="coerce")
    bars_work = bars.copy()
    bars_work["_date"] = bars_work["trade_date"].map(_as_date)
    bars_work["_volume"] = bars_work["volume"].map(_volume)
    dates = sorted(bars_work["_date"].unique())
    bars_by_day = {
        day: {str(row["symbol"]): row for row in group.to_dict(orient="records")}
        for day, group in bars_work.groupby("_date", sort=True)
    }
    candidates_by_entry_day: dict[date, list[dict[str, Any]]] = {}
    skipped: list[dict[str, Any]] = []
    for row in work.to_dict(orient="records"):
        if row["eligible"] is not True:
            continue
        decision_day = row["_decision_date"]
        next_index = next((i for i, day in enumerate(dates) if day > decision_day), None)
        if next_index is None:
            skipped.append({"symbol": row["symbol"], "reason": "no_next_trading_session"})
            continue
        candidates_by_entry_day.setdefault(dates[next_index], []).append(row)

    cash = float(initial_cash)
    positions: dict[str, dict[str, Any]] = {}
    fills: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    daily_equity: list[dict[str, Any]] = []
    realized_pnl = 0.0

    for day in dates:
        day_bars = bars_by_day[day]

        # 1) Process protective stops on positions carried into this session.
        for symbol in sorted(list(positions)):
            position = positions[symbol]
            bar = day_bars.get(symbol)
            if bar is None or float(bar["low"]) > position["stop_price"]:
                continue
            reference = min(float(bar["open"]), position["stop_price"])
            result = simulate_daily_fill(
                side="sell", quantity=position["shares"], reference_price=reference,
                trade_date=day, entry_date=position["entry_date"],
                suspended=bar["suspended"], limit_up_locked=bar["limit_up_locked"],
                limit_down_locked=bar["limit_down_locked"],
                available_volume=bar["_volume"], max_participation=max_participation,
                costs=costs,
            )
            fills.append({"date": day.isoformat(), "symbol": symbol, **result.to_dict()})
            if not result.filled:
                continue
            sold = result.quantity
            allocated_buy_cost = position["buy_cost"] * sold / position["shares"]
            proceeds = sold * float(result.fill_price) - result.total_cost
            cash += proceeds
            pnl = (float(result.fill_price) - position["entry_price"]) * sold - result.total_cost - allocated_buy_cost
            realized_pnl += pnl
            trades.append({
                "symbol": symbol, "entry_date": position["entry_date"].isoformat(),
                "exit_date": day.isoformat(), "shares": sold,
                "entry_price": position["entry_price"], "exit_price": result.fill_price,
                "realized_pnl": round(pnl, 2), "exit_reason": "protective_stop",
            })
            position["shares"] -= sold
            position["buy_cost"] -= allocated_buy_cost
            if position["shares"] == 0:
                del positions[symbol]

        # 2) Enter eligible close signals at the next session's open.
        rows = candidates_by_entry_day.get(day, [])
        rows = sorted(rows, key=lambda r: (-float(r["_score"]), str(r["symbol"]), str(r["strategy"])))
        seen_symbols: set[str] = set()
        for candidate in rows:
            symbol = str(candidate["symbol"])
            if symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            if symbol in positions:
                skipped.append({"symbol": symbol, "date": day.isoformat(), "reason": "already_held"})
                continue
            bar = day_bars.get(symbol)
            if bar is None:
                skipped.append({"symbol": symbol, "date": day.isoformat(), "reason": "missing_entry_bar"})
                continue
            stop = candidate["stop_price"]
            if isinstance(stop, bool) or not isinstance(stop, Real) or not math.isfinite(float(stop)) or float(stop) <= 0:
                skipped.append({"symbol": symbol, "date": day.isoformat(), "reason": "invalid_stop_price"})
                continue
            entry_reference = float(bar["open"])
            expected_entry = entry_reference * (1.0 + costs.slippage_bps / 10000.0)
            if float(stop) >= expected_entry:
                skipped.append({"symbol": symbol, "date": day.isoformat(), "reason": "stop_not_below_entry"})
                continue
            portfolio_risk = sum(_risk(p) for p in positions.values())
            sizing = size_long_position(
                entry_price=expected_entry, stop_price=float(stop), available_cash=cash,
                open_position_count=len(positions), open_portfolio_risk=portfolio_risk,
                per_trade_risk_limit=per_trade_risk_limit,
                portfolio_risk_limit=portfolio_risk_limit, max_positions=max_positions,
            )
            if not sizing.approved:
                skipped.append({"symbol": symbol, "date": day.isoformat(), "reason": "position_sizing_rejected", "details": list(sizing.reasons)})
                continue
            result = simulate_daily_fill(
                side="buy", quantity=sizing.shares, reference_price=entry_reference,
                trade_date=day, suspended=bar["suspended"],
                limit_up_locked=bar["limit_up_locked"], limit_down_locked=bar["limit_down_locked"],
                available_volume=bar["_volume"], max_participation=max_participation,
                costs=costs,
            )
            fills.append({"date": day.isoformat(), "symbol": symbol, **result.to_dict()})
            if not result.filled:
                continue
            shares = result.quantity
            required = shares * float(result.fill_price) + result.total_cost
            if required > cash:
                skipped.append({"symbol": symbol, "date": day.isoformat(), "reason": "cash_shortfall_after_costs"})
                continue
            cash -= required
            positions[symbol] = {
                "shares": shares, "entry_price": float(result.fill_price),
                "entry_date": day, "stop_price": float(stop),
                "buy_cost": result.total_cost, "last_close": float(bar["close"]),
            }

        # 3) Mark remaining positions to the latest available close.
        market_value = 0.0
        for symbol, position in positions.items():
            bar = day_bars.get(symbol)
            if bar is not None:
                position["last_close"] = float(bar["close"])
            market_value += position["shares"] * position["last_close"]
        daily_equity.append({
            "date": day.isoformat(), "cash": round(cash, 2),
            "market_value": round(market_value, 2),
            "equity": round(cash + market_value, 2),
            "open_positions": len(positions),
        })

    ending_equity = daily_equity[-1]["equity"] if daily_equity else cash
    metrics = calculate_backtest_metrics(daily_equity, trades, float(initial_cash)) if daily_equity else None
    return {
        "audit_passed": True,
        "initial_cash": round(float(initial_cash), 2),
        "ending_cash": round(cash, 2),
        "ending_equity": round(ending_equity, 2),
        "total_return_pct": round((ending_equity / float(initial_cash) - 1) * 100, 4),
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(sum(
            p["shares"] * (p["last_close"] - p["entry_price"]) - p["buy_cost"]
            for p in positions.values()
        ), 2),
        "open_positions": [
            {"symbol": symbol, "shares": p["shares"], "entry_date": p["entry_date"].isoformat(),
             "entry_price": p["entry_price"], "stop_price": p["stop_price"],
             "last_close": p["last_close"]}
            for symbol, p in sorted(positions.items())
        ],
        "fills": fills,
        "trades": trades,
        "skipped": skipped,
        "daily_equity": daily_equity,
        "metrics": metrics,
    }
