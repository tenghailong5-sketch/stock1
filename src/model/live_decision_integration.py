"""Live runner adapter that executes M2 -> M3 -> M5 -> M8 gates per candidate."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from src.model.four_layer_decision import evaluate_trade_decision
from src.providers.akshare_15m import fetch_live_15m_confirmation


def _truth(value) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _finite(value) -> bool:
    try:
        return not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def build_live_decision_gate_audit(
    watchlist: pd.DataFrame,
    m5: pd.DataFrame,
    market_context: dict,
    mainline: pd.DataFrame,
    industry_membership: pd.DataFrame,
    *,
    decision_time,
    risk_context_by_symbol: dict | None = None,
    minute_confirmation_fetcher=None,
    minute_scan_limit: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Run all four decision gates without inventing missing factor/risk/PIT inputs.

    The current runtime mainline and stock-history collectors are forward-only
    diagnostics. They are not promoted to historical PIT unless upstream rows
    explicitly carry valid PIT lineage.
    """
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    if minute_scan_limit < 0:
        raise ValueError("minute_scan_limit cannot be negative")
    risk_context_by_symbol = risk_context_by_symbol or {}
    minute_confirmation_fetcher = minute_confirmation_fetcher or fetch_live_15m_confirmation

    if watchlist is None or watchlist.empty:
        return pd.DataFrame(), {
            "status": "NO_CANDIDATES", "candidate_count": 0,
            "minute_scanned_count": 0, "allowed_count": 0, "blocked_count": 0,
        }
    if "代码" not in watchlist.columns:
        raise ValueError("watchlist requires 代码")
    symbols = watchlist["代码"].astype(str).str.strip().str.zfill(6).tolist()
    if len(symbols) != len(set(symbols)):
        raise ValueError("watchlist contains duplicate symbols")

    # Select a bounded minute-confirmation set by the existing exploratory rank.
    rank_col = "快速初筛分" if "快速初筛分" in watchlist else "当日成交额"
    ranked = watchlist.copy()
    if rank_col in ranked:
        ranked[rank_col] = pd.to_numeric(ranked[rank_col], errors="coerce")
        ranked = ranked.sort_values(rank_col, ascending=False, na_position="last")
    minute_symbols = set(ranked["代码"].astype(str).str.zfill(6).head(minute_scan_limit))
    minute_results = {}
    for symbol in minute_symbols:
        try:
            minute_results[symbol] = minute_confirmation_fetcher(symbol)
        except Exception as exc:
            minute_results[symbol] = {
                "status": "UNAVAILABLE_15M", "entry_ready": False,
                "reason": f"{type(exc).__name__}:{str(exc)[:180]}",
                "historical_pit_eligible": False,
            }

    membership_by_symbol = {}
    if (
        industry_membership is not None and not industry_membership.empty
        and {"symbol", "industry_code"}.issubset(industry_membership.columns)
    ):
        membership = industry_membership.copy()
        membership["symbol"] = membership["symbol"].astype(str).str.strip().str.zfill(6)
        for symbol, group in membership.groupby("symbol"):
            if len(group) == 1:
                membership_by_symbol[symbol] = group.iloc[0].to_dict()

    latest_mainline = {}
    if mainline is not None and not mainline.empty and "industry_code" in mainline:
        panel = mainline.copy()
        if "trade_date" in panel:
            panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
            panel = panel.sort_values("trade_date")
        for code, group in panel.groupby("industry_code"):
            if len(group):
                latest_mainline[str(code)] = group.iloc[-1].to_dict()

    m5_lookup = {}
    if m5 is not None and not m5.empty:
        for row in m5.to_dict("records"):
            m5_lookup[(str(row.get("symbol", "")).zfill(6), str(row.get("strategy", "")))] = row

    market_ready = market_context.get("status") == "READY"
    market_state = market_context.get("market_state")
    market_scale = market_context.get("position_scale")
    records = []
    for stock in watchlist.to_dict("records"):
        symbol = str(stock["代码"]).strip().zfill(6)
        membership_row = membership_by_symbol.get(symbol, {})
        industry_code = str(membership_row.get("industry_code", ""))
        industry = latest_mainline.get(industry_code, {})
        phase = industry.get("mainline_phase", "UNKNOWN")
        industry_ready = _truth(industry.get("persistence_ready"))
        industry_pit = _truth(industry.get("historical_pit_eligible"))
        concentration_status = str(industry.get("stock_turnover_pit_status", "not_ready"))
        minute = minute_results.get(symbol, {
            "status": "NOT_SCANNED_15M", "entry_ready": False,
            "reason": "outside_bounded_15m_scan_pool", "historical_pit_eligible": False,
        })
        risk = risk_context_by_symbol.get(symbol, {})
        entry_price = risk.get("entry_price", stock.get("最新价", np.nan))
        args = {
            "market_state": market_state or "UNKNOWN",
            "market_position_scale": market_scale,
            "market_data_ready": market_ready,
            "mainline_phase": phase,
            "industry_persistence_ready": industry_ready,
            "industry_historical_pit_eligible": industry_pit,
            "stock_turnover_pit_status": concentration_status,
            "strategy_eligible": False,
            "strategy_score": None,
            "entry_ready": _truth(minute.get("entry_ready")),
            "entry_price": entry_price,
            "stop_price": risk.get("stop_price", np.nan),
            "quantity": risk.get("quantity", 0),
            "planned_risk": risk.get("planned_risk", np.nan),
            "reward_risk": risk.get("reward_risk", np.nan),
            "minimum_reward_risk": risk.get("minimum_reward_risk", np.nan),
            "existing_portfolio_risk": risk.get("existing_portfolio_risk", np.nan),
            "existing_position_count": risk.get("existing_position_count", -1),
            "same_industry_or_corr_gt_06": risk.get("same_industry_or_corr_gt_06"),
        }
        for strategy in ("A", "B", "C"):
            candidate = m5_lookup.get((symbol, strategy), {})
            args["strategy_eligible"] = _truth(candidate.get("eligible"))
            args["strategy_score"] = candidate.get("score")
            decision = evaluate_trade_decision(**args)
            records.append({
                "symbol": symbol, "strategy": strategy,
                "market_mode": decision.market_mode,
                "industry_mode": decision.industry_mode,
                "m5_score": candidate.get("score"),
                "m5_eligible": _truth(candidate.get("eligible")),
                "minute_status": minute.get("status"),
                "minute_entry_ready": _truth(minute.get("entry_ready")),
                "minute_reason": minute.get("reason"),
                "stock_industry_code": industry_code or None,
                "mainline_phase": phase,
                "stock_turnover_pit_status": concentration_status,
                "allowed": decision.allowed,
                "max_allowed_risk": decision.max_allowed_risk,
                "planned_risk": decision.planned_risk,
                "reasons": list(decision.reasons),
            })
    result = pd.DataFrame.from_records(records)
    allowed = int(result["allowed"].sum()) if not result.empty else 0
    audit = {
        "status": "EXECUTED_FAIL_CLOSED",
        "candidate_count": int(len(result)),
        "unique_stock_count": int(len(symbols)),
        "minute_scanned_count": int(len(minute_symbols)),
        "allowed_count": allowed,
        "blocked_count": int(len(result) - allowed),
        "market_context_status": market_context.get("status"),
        "mainline_pit_rows": int(
            mainline.get("historical_pit_eligible", pd.Series(dtype=bool)).eq(True).sum()
        ) if mainline is not None and not mainline.empty else 0,
        "formal_signal_emitted": False,
        "decision_time": cutoff.isoformat(),
    }
    return result, audit


def summarize_gate_blockers(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["strategy", "blocker", "count"])
    rows = []
    for row in frame.to_dict("records"):
        for reason in row.get("reasons", []):
            rows.append({"strategy": row.get("strategy", "UNKNOWN"), "blocker": reason})
    if not rows:
        return pd.DataFrame(columns=["strategy", "blocker", "count"])
    return (pd.DataFrame(rows).groupby(["strategy", "blocker"], as_index=False)
            .size().rename(columns={"size": "count"})
            .sort_values(["strategy", "count", "blocker"], ascending=[True, False, True])
            .reset_index(drop=True))

def aggregate_archived_stock_turnover(
    archive_root, expected_symbols, *, trade_date, decision_time,
) -> tuple[pd.DataFrame, dict]:
    """Build industry turnover concentration only from archived, complete PIT rows."""
    from pathlib import Path
    from src.model.m4_industry_membership import aggregate_pit_industry_turnover

    root = Path(archive_root)
    stock_rows, membership_rows = [], []
    stock_dir = root / "stock_daily"
    member_dir = root / "industry_membership"
    for path in sorted(stock_dir.glob("*.jsonl")) if stock_dir.exists() else []:
        try:
            frame = pd.read_json(path, lines=True, dtype={"symbol": str})
        except (ValueError, OSError) as exc:
            return pd.DataFrame(), {
                "status": "invalid_stock_archive",
                "reason": f"{path.name}:{type(exc).__name__}:{str(exc)[:160]}",
                "historical_pit_eligible": False,
            }
        if not frame.empty:
            if "symbol" not in frame:
                frame["symbol"] = path.stem
            stock_rows.append(frame)
    for path in sorted(member_dir.glob("*.jsonl")) if member_dir.exists() else []:
        try:
            frame = pd.read_json(path, lines=True, dtype={"symbol": str})
        except (ValueError, OSError) as exc:
            return pd.DataFrame(), {
                "status": "invalid_membership_archive",
                "reason": f"{path.name}:{type(exc).__name__}:{str(exc)[:160]}",
                "historical_pit_eligible": False,
            }
        if not frame.empty:
            membership_rows.append(frame)
    empty = pd.DataFrame(columns=[
        "trade_date", "industry_code", "turnover", "available_time",
        "decision_time", "availability_basis", "historical_pit_eligible",
    ])
    if not stock_rows:
        return empty, {
            "status": "stock_archive_missing", "expected_universe_size": len(expected_symbols),
            "stock_bar_eligible_symbols": 0, "historical_pit_eligible": False,
        }
    if not membership_rows:
        return empty, {
            "status": "membership_archive_missing", "expected_universe_size": len(expected_symbols),
            "stock_bar_eligible_symbols": 0, "historical_pit_eligible": False,
        }
    stock_bars = pd.concat(stock_rows, ignore_index=True, sort=False)
    membership = pd.concat(membership_rows, ignore_index=True, sort=False)
    try:
        result, audit = aggregate_pit_industry_turnover(
            stock_bars, membership, expected_symbols,
            trade_date=trade_date, decision_time=decision_time,
            scope_id="live_full_market_universe",
        )
    except Exception as exc:
        return empty, {
            "status": "aggregation_error",
            "reason": f"{type(exc).__name__}:{str(exc)[:180]}",
            "historical_pit_eligible": False,
        }
    return result, audit
