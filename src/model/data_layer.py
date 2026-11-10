"""Point-in-time A-share data primitives for FHM-100 R2.1.

Provider adapters return raw observations with explicit availability metadata.
This module deliberately does not invent historical industry membership,
corporate actions, exchange-rule history, or official FHM scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_DAILY = {"symbol", "trade_date", "open", "high", "low", "close",
                  "volume", "amount", "available_time", "source_id", "source_revision"}


@dataclass(frozen=True)
class DataQuality:
    accepted: bool
    reason: str
    rows: int


def _utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError("timestamps must include timezone")
    return ts.tz_convert("UTC")


def validate_daily_bars(frame: pd.DataFrame, decision_time) -> tuple[pd.DataFrame, DataQuality]:
    """Validate raw (unadjusted) bars and retain only data available by decision time."""
    missing = REQUIRED_DAILY - set(frame.columns)
    if missing:
        return frame.iloc[0:0].copy(), DataQuality(False, "missing:" + ",".join(sorted(missing)), 0)
    cutoff = _utc(decision_time)
    x = frame.copy()
    raw_available = pd.to_datetime(x["available_time"], errors="coerce")
    if raw_available.isna().any() or raw_available.dt.tz is None:
        return frame.iloc[0:0].copy(), DataQuality(False, "available_time_must_be_timezone_aware", 0)
    x["available_time"] = raw_available.dt.tz_convert("UTC")
    x["trade_date"] = pd.to_datetime(x["trade_date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "amount"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x[x.available_time.notna() & (x.available_time <= cutoff)]
    x = x.dropna(subset=["trade_date", "open", "high", "low", "close", "volume", "amount"])
    logical = ((x.open > 0) & (x.high > 0) & (x.low > 0) & (x.close > 0) &
               (x.high >= x[["open", "low", "close"]].max(axis=1)) &
               (x.low <= x[["open", "high", "close"]].min(axis=1)) &
               (x.volume >= 0) & (x.amount >= 0))
    x = x[logical].copy()
    # Preserve source revisions; choose the latest available revision for the same source/date.
    x = x.sort_values("available_time").drop_duplicates(
        ["symbol", "trade_date", "source_id"], keep="last")
    x = x.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    return x, DataQuality(True, "ok", len(x))


def build_total_return_index(prices: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Build a total-return index only where every corporate action is verified.

    Expected action columns: symbol, ex_date, action_type, cash_per_share,
    share_factor, verified_status. cash_per_share is pre-tax per old share;
    share_factor is new shares per old share. Rights issues and complex actions
    are intentionally rejected until a fully specified cashflow ledger exists.
    """
    action_required = {"symbol", "ex_date", "action_type", "cash_per_share",
                       "share_factor", "verified_status"}
    if action_required - set(actions.columns):
        raise ValueError("corporate action ledger schema incomplete")
    out = []
    for symbol, g in prices.groupby("symbol", sort=False):
        g = g.sort_values("trade_date").copy()
        a = actions[actions.symbol == symbol].copy()
        if not a.empty:
            if not (a.verified_status == "VERIFIED").all():
                g["total_return_index"] = np.nan
                g["corporate_action_status"] = "UNVERIFIED"
                out.append(g)
                continue
            if not set(a.action_type).issubset({"cash_dividend", "split", "bonus"}):
                g["total_return_index"] = np.nan
                g["corporate_action_status"] = "UNSUPPORTED_ACTION"
                out.append(g)
                continue
        a["ex_date"] = pd.to_datetime(a.ex_date)
        action_map = {pd.Timestamp(r.ex_date).normalize(): r
                      for r in a.itertuples(index=False)}
        idx = [100.0]
        status = ["VERIFIED" if a.empty else "VERIFIED"]
        prev_close = float(g.iloc[0].close)
        for row in g.iloc[1:].itertuples(index=False):
            action = action_map.get(pd.Timestamp(row.trade_date).normalize())
            cash, shares = (0.0, 1.0) if action is None else (
                float(action.cash_per_share), float(action.share_factor))
            if shares <= 0 or prev_close <= 0:
                idx.append(np.nan); status.append("INVALID_ACTION"); prev_close = float(row.close); continue
            # Wealth per old share: cash distribution plus value of new shares.
            gross_factor = (cash + shares * float(row.close)) / prev_close
            idx.append(idx[-1] * gross_factor if np.isfinite(idx[-1]) else np.nan)
            status.append("VERIFIED")
            prev_close = float(row.close)
        g["total_return_index"] = idx
        g["corporate_action_status"] = status
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else prices.assign(
        total_return_index=np.nan, corporate_action_status="NO_DATA")


def industry_relative_strength(industry_returns: pd.DataFrame,
                               market_returns: pd.DataFrame,
                               periods=(5, 20)) -> pd.DataFrame:
    """Returns are decimal cumulative returns keyed by trade_date."""
    x = industry_returns.merge(market_returns, on="trade_date", suffixes=("_industry", "_market"))
    for h in periods:
        x[f"rs_industry_{h}"] = x[f"return_{h}_industry"] - x[f"return_{h}_market"]
    return x


def rolling_residual_strength(stock_returns: pd.Series, market_returns: pd.Series,
                              industry_returns: pd.Series, window: int = 60,
                              min_periods: int = 40) -> pd.Series:
    """Rolling OLS residual return proxy using only trailing observations.

    Estimates betas from returns strictly before the return being explained.
    Requires aligned, point-in-time return series.
    """
    frame = pd.concat([stock_returns, market_returns, industry_returns], axis=1).astype(float)
    frame.columns = ["stock", "market", "industry"]
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for i in range(len(frame)):
        hist = frame.iloc[max(0, i-window):i].dropna()
        current = frame.iloc[i]
        if len(hist) < min_periods or current.isna().any():
            continue
        X = np.column_stack([np.ones(len(hist)), hist.market.values, hist.industry.values])
        beta, *_ = np.linalg.lstsq(X, hist.stock.values, rcond=None)
        result.iloc[i] = current.stock - beta[1]*current.market - beta[2]*current.industry
    return result


def cross_sectional_percentile(values: pd.Series, higher_is_better=True) -> pd.Series:
    """Percentile ranks; missing inputs remain missing and are never imputed."""
    ranked = values.rank(method="average", pct=True)
    return ranked * 100 if higher_is_better else (1-ranked) * 100


def assert_historical_industry_membership(membership: pd.DataFrame) -> None:
    required = {"symbol", "industry_code", "valid_from", "valid_to", "available_time", "source_id"}
    missing = required - set(membership.columns)
    if missing:
        raise ValueError("historical industry membership required; missing " + ",".join(sorted(missing)))
    if membership[["valid_from", "available_time"]].isna().any().any():
        raise ValueError("industry membership dates cannot be missing")


def decision_snapshot_id(parts: Iterable[str]) -> str:
    """Stable deterministic identifier; caller supplies source/version hashes."""
    import hashlib
    payload = "|".join(sorted(str(p) for p in parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
