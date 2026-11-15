"""Live runner adapter that executes M2 -> M3 -> M5 -> M8 gates per candidate."""
from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

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


def _live_snapshot_ready(row: dict, cutoff: pd.Timestamp, *, max_age_hours: float = 240.0) -> bool:
    """Validate a retrieval-time snapshot for forward use only; never grant PIT status."""
    required = {"available_time", "source_id", "source_revision", "availability_basis", "snapshot_scope"}
    if not required.issubset(row):
        return False
    if str(row.get("availability_basis", "")).strip() != "retrieval_time_only":
        return False
    if _truth(row.get("historical_pit_eligible")):
        return False
    if any(not str(row.get(field, "")).strip() for field in ("source_id", "source_revision", "snapshot_scope")):
        return False
    try:
        available = pd.Timestamp(row["available_time"])
        if available.tzinfo is None:
            return False
        age = cutoff.tz_convert("UTC") - available.tz_convert("UTC")
        if not pd.Timedelta(0) <= age <= pd.Timedelta(hours=max_age_hours):
            return False
        # If the provider supplies an effective membership interval, honor it;
        # retrieval-time evidence does not override an explicitly expired mapping.
        valid_from_raw = row.get("valid_from")
        valid_to_raw = row.get("valid_to")
        if valid_from_raw is not None and not pd.isna(valid_from_raw):
            valid_from = pd.Timestamp(valid_from_raw).normalize()
            decision_day = cutoff.tz_convert("Asia/Shanghai").tz_localize(None).normalize()
            if valid_from > decision_day:
                return False
        if valid_to_raw is not None and not pd.isna(valid_to_raw):
            valid_to = pd.Timestamp(valid_to_raw).normalize()
            decision_day = cutoff.tz_convert("Asia/Shanghai").tz_localize(None).normalize()
            if decision_day >= valid_to:
                return False
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def load_risk_context_file(path: str | None) -> tuple[dict, dict]:
    """Load operator-supplied, per-symbol M8 plans without inventing account data.

    JSON shape: {"600000": {"entry_price": ..., "stop_price": ...,
    "quantity": ..., "planned_risk": ..., "reward_risk": ...,
    "minimum_reward_risk": ..., "existing_portfolio_risk": ...,
    "existing_position_count": ..., "available_risk_budget": ...}}
    """
    if not path:
        return {}, {"status": "NOT_CONFIGURED", "symbol_count": 0}
    from pathlib import Path
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("risk context JSON root must be an object keyed by symbol")
        normalized = {}
        for raw_symbol, context in payload.items():
            symbol = str(raw_symbol).strip().zfill(6)
            if len(symbol) != 6 or not symbol.isdigit() or not isinstance(context, dict):
                raise ValueError("risk context must map six-digit symbols to objects")
            if symbol in normalized:
                raise ValueError(f"duplicate normalized symbol: {symbol}")
            normalized[symbol] = context
        return normalized, {
            "status": "LOADED", "symbol_count": len(normalized),
            "source": "operator_configured_file",
        }
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {}, {
            "status": "INVALID_OR_UNAVAILABLE",
            "reason": f"{type(exc).__name__}:{str(exc)[:200]}",
            "symbol_count": 0,
        }


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
    minute_scan_limit: int | None = None,
    minute_scan_workers: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run all four decision gates without inventing missing factor/risk/PIT inputs.

    The current runtime mainline and stock-history collectors are forward-only
    diagnostics. They are not promoted to historical PIT unless upstream rows
    explicitly carry valid PIT lineage.
    """
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    if minute_scan_limit is not None and minute_scan_limit < 0:
        raise ValueError("minute_scan_limit must be non-negative or None")
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

    # M5 eligibility is the prerequisite for an expensive 15-minute check.
    # Keep all candidate rows for diagnostics, but only scan symbols with at
    # least one eligible A/B/C strategy. This changes request ordering, not
    # candidate coverage or the decision rules.
    m5_lookup = {}
    if m5 is not None and not m5.empty:
        for row in m5.to_dict("records"):
            m5_lookup[(str(row.get("symbol", "")).zfill(6), str(row.get("strategy", "")))] = row
    m5_eligible_symbols = {
        symbol for (symbol, strategy), candidate in m5_lookup.items()
        if strategy in {"A", "B", "C"} and _truth(candidate.get("eligible"))
    }
    # Preserve the existing exploratory rank among M5-qualified symbols.
    rank_col = "快速初筛分" if "快速初筛分" in watchlist else "当日成交额"
    ranked = watchlist.copy()
    if rank_col in ranked:
        ranked[rank_col] = pd.to_numeric(ranked[rank_col], errors="coerce")
        ranked = ranked.sort_values(rank_col, ascending=False, na_position="last")
    # Bind every 15-minute observation to the runner's single frozen cutoff.
    minute_ranked_symbols = ranked["代码"].astype(str).str.zfill(6)
    minute_ranked_symbols = minute_ranked_symbols[minute_ranked_symbols.isin(m5_eligible_symbols)]
    minute_symbols = (
        minute_ranked_symbols.tolist() if minute_scan_limit is None
        else minute_ranked_symbols.head(minute_scan_limit).tolist()
    )
    if minute_scan_workers is None:
        minute_scan_workers = int(os.getenv("FHM_15M_SCAN_WORKERS", "4"))
    minute_scan_workers = max(1, min(int(minute_scan_workers), 8))

    def _fetch_minute(symbol):
        try:
            return symbol, minute_confirmation_fetcher(symbol, captured_at=cutoff)
        except Exception as exc:
            return symbol, {
                "status": "UNAVAILABLE_15M", "entry_ready": False,
                "reason": f"{type(exc).__name__}:{str(exc)[:180]}",
                "historical_pit_eligible": False,
            }

    minute_results = {}
    with ThreadPoolExecutor(max_workers=minute_scan_workers) as executor:
        for symbol, result in executor.map(_fetch_minute, minute_symbols):
            minute_results[symbol] = result

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

    market_ready = market_context.get("status") == "READY"
    market_state = market_context.get("market_state")
    market_scale = market_context.get("position_scale")
    records = []
    for stock in watchlist.to_dict("records"):
        symbol = str(stock["代码"]).strip().zfill(6)
        membership_row = membership_by_symbol.get(symbol, {})
        industry_code = str(membership_row.get("industry_code", ""))
        industry = latest_mainline.get(industry_code, {})
        # Respect the M3 ignition evidence gate. Raw candidate phase alone
        # must never authorize a live entry when startup evidence is unconfirmed.
        phase = industry.get("mainline_phase_gated", industry.get("mainline_phase", "UNKNOWN"))
        industry_ready = _truth(industry.get("persistence_ready"))
        # The current runner passes a forward-only observed membership snapshot.
        # Never let that mapping authorize a PIT mainline entry, even if a
        # separate industry row happens to carry PIT metadata.
        membership_pit = False
        required_membership_pit = {
            "historical_pit_eligible", "availability_basis",
            "available_time", "valid_from", "valid_to",
        }
        if required_membership_pit.issubset(membership_row):
            try:
                member_available = pd.Timestamp(membership_row["available_time"])
                target_trade_date = pd.Timestamp(
                    stock.get("trade_date", industry.get("trade_date"))
                ).normalize()
                valid_from = pd.Timestamp(membership_row["valid_from"]).normalize()
                valid_to_raw = membership_row.get("valid_to")
                valid_to = pd.Timestamp(valid_to_raw).normalize() if pd.notna(valid_to_raw) else None
                membership_pit = (
                    _truth(membership_row.get("historical_pit_eligible"))
                    and str(membership_row.get("availability_basis", "")).strip() == "source_published_at"
                    and member_available.tzinfo is not None
                    and member_available.tz_convert("UTC") <= cutoff.tz_convert("UTC")
                    and not pd.isna(target_trade_date)
                    and valid_from <= target_trade_date
                    and (valid_to is None or target_trade_date < valid_to)
                )
            except (TypeError, ValueError, OverflowError):
                membership_pit = False
        industry_pit = (
            membership_pit and _truth(industry.get("historical_pit_eligible"))
        )
        # Forward mode is a separate permission path. Both the current
        # membership observation and industry observation must carry fresh,
        # timestamped source lineage; neither is promoted to historical PIT.
        membership_live = _live_snapshot_ready(membership_row, cutoff)
        # scan_runtime_mainlines already produces the current forward-only
        # industry panel at this run's cutoff. It does not publish per-row
        # lineage columns, so use its actual readiness/phase instead of
        # demanding a second synthetic metadata envelope.
        industry_live = (
            membership_live
            and bool(industry)
            and str(phase) in {"IGNITION", "PEAK", "DECAY"}
        )
        concentration_status = str(industry.get("stock_turnover_pit_status", "not_ready"))
        live_concentration_status = str(industry.get("stock_turnover_live_status", "not_ready"))
        minute = minute_results.get(symbol, {
            "status": "NOT_SCANNED_15M", "entry_ready": False,
            "reason": ("outside_configured_15m_scan_pool" if symbol in m5_eligible_symbols
                       else "m5_strategy_not_eligible"), "historical_pit_eligible": False,
        })
        risk = risk_context_by_symbol.get(symbol, {})
        # A market quote is not an approved execution plan. M8 requires an
        # explicit entry, stop, size, and portfolio-risk context.
        entry_price = risk.get("entry_price", np.nan)
        required_risk_fields = (
            "entry_price", "stop_price", "quantity", "planned_risk",
            "reward_risk", "minimum_reward_risk", "existing_portfolio_risk",
            "existing_position_count", "available_risk_budget",
        )
        risk_context_complete = (
            all(field in risk for field in required_risk_fields)
            and all(_finite(risk.get(field)) for field in (
                "entry_price", "stop_price", "planned_risk", "reward_risk",
                "minimum_reward_risk", "existing_portfolio_risk",
                "existing_position_count", "available_risk_budget",
            ))
            and isinstance(risk.get("quantity"), int)
            and not isinstance(risk.get("quantity"), bool)
        )
        args = {
            "market_state": market_state or "UNKNOWN",
            "market_position_scale": market_scale,
            "market_data_ready": market_ready,
            "mainline_phase": phase,
            "industry_persistence_ready": industry_ready,
            "industry_historical_pit_eligible": industry_pit,
            "stock_turnover_pit_status": concentration_status,
            "decision_mode": "live_forward",
            "industry_live_data_eligible": industry_live,
            "stock_turnover_live_status": live_concentration_status,
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
            "available_risk_budget": risk.get("available_risk_budget", np.nan),
            "same_industry_or_corr_gt_06": risk.get("same_industry_or_corr_gt_06", False),
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
                "m8_risk_context_complete": bool(risk_context_complete),
                "stock_industry_code": industry_code or None,
                "mainline_phase": phase,
                "stock_turnover_pit_status": concentration_status,
                "industry_live_data_eligible": industry_live,
                "stock_turnover_live_status": live_concentration_status,
                "historical_pit_eligible": industry_pit,
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
        "m5_eligible_symbol_count": int(len(m5_eligible_symbols)),
        "minute_scanned_count": int(len(minute_symbols)),
        "minute_scan_limit": "ALL" if minute_scan_limit is None else int(minute_scan_limit),
        "allowed_count": allowed,
        "blocked_count": int(len(result) - allowed),
        "m8_risk_context_complete_symbol_count": int(
            result.loc[result["m8_risk_context_complete"], "symbol"].nunique()
        ) if not result.empty else 0,
        "m8_risk_context_missing_symbol_count": int(
            result.loc[~result["m8_risk_context_complete"], "symbol"].nunique()
        ) if not result.empty else 0,
        "market_context_status": market_context.get("status"),
        "mainline_pit_rows": int(
            mainline.get("historical_pit_eligible", pd.Series(dtype=bool)).eq(True).sum()
        ) if mainline is not None and not mainline.empty else 0,
        "live_forward_ready_rows": int(result.get("industry_live_data_eligible", pd.Series(dtype=bool)).eq(True).sum()) if not result.empty else 0,
        "decision_mode": "live_forward",
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
    scopes = membership.get("snapshot_scope", pd.Series(dtype=object)).dropna().astype(str).str.strip().unique()
    if len(scopes) != 1 or not scopes[0]:
        return empty, {
            "status": "ambiguous_membership_scope",
            "snapshot_scopes": sorted(scopes.tolist()),
            "historical_pit_eligible": False,
        }
    try:
        result, audit = aggregate_pit_industry_turnover(
            stock_bars, membership, expected_symbols,
            trade_date=trade_date, decision_time=decision_time,
            scope_id=str(scopes[0]),
        )
    except Exception as exc:
        return empty, {
            "status": "aggregation_error",
            "reason": f"{type(exc).__name__}:{str(exc)[:180]}",
            "historical_pit_eligible": False,
        }
    return result, audit
