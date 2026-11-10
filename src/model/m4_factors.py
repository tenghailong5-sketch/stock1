"""M4 point-in-time factor primitives from FHM-100 baseline.

This module computes raw factor observations. Cross-sectional percentile ranks
must be calculated within the correct contemporaneous peer universe. It does
not fabricate catalyst, capital-flow, turnover, or limit-event data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_BARS = {"symbol", "trade_date", "open", "high", "low", "close",
                 "volume", "amount", "available_time"}


def _numeric(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def price_structure(bars: pd.DataFrame, decision_time=None) -> pd.DataFrame:
    """Compute price factors using only bars available by decision_time.

    If decision_time is omitted, bars must still carry timezone-aware
    available_time values; the caller is responsible for upstream PIT filtering.
    """
    missing = REQUIRED_BARS - set(bars.columns)
    if missing:
        raise ValueError("bars missing: " + ",".join(sorted(missing)))
    x = _numeric(bars, ["open", "high", "low", "close", "volume", "amount"])
    x["trade_date"] = pd.to_datetime(x["trade_date"], errors="coerce")
    raw_available = pd.to_datetime(x["available_time"], errors="coerce")
    if x["trade_date"].isna().any() or raw_available.isna().any():
        raise ValueError("invalid trade_date or available_time")
    if raw_available.dt.tz is None:
        raise ValueError("available_time must be timezone-aware")
    x["available_time"] = raw_available.dt.tz_convert("UTC")
    if decision_time is not None:
        cutoff = pd.Timestamp(decision_time)
        if cutoff.tzinfo is None:
            raise ValueError("decision_time must be timezone-aware")
        x = x.loc[x["available_time"] <= cutoff.tz_convert("UTC")].copy()
    x = x.sort_values(["symbol", "trade_date"]).copy()
    if x[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("invalid OHLC values")
    span = (x["high"] - x["low"]).where(lambda v: v > 0)
    x["CLV"] = (x["close"] - x["low"]) / span
    x["BodyStrength"] = (x["close"] - x["open"]) / span
    grouped = x.groupby("symbol", group_keys=False)
    for n in (5, 10, 20):
        x[f"return_{n}"] = grouped["close"].pct_change(n)
    x["high20_prev"] = grouped["high"].transform(lambda s: s.shift(1).rolling(20, min_periods=20).max())
    x["breakout20"] = x["close"] / x["high20_prev"] - 1
    x["upper_shadow_ratio"] = (x["high"] - x[["open", "close"]].max(axis=1)) / span
    x["recent_high20"] = grouped["high"].transform(lambda s: s.rolling(20, min_periods=20).max())
    x["Pullback"] = x["close"] / x["recent_high20"] - 1
    # Sessions elapsed since the most recent rolling-20-session high-water
    # close. Zero means the current close is at a new rolling peak.
    def _duration_since_peak(s: pd.Series) -> pd.Series:
        values = s.to_numpy(dtype=float)
        out = np.full(len(values), np.nan)
        for i in range(len(values)):
            start = max(0, i - 19)
            window = values[start:i + 1]
            if not np.isfinite(window).all():
                continue
            peak_value = np.max(window)
            peak_positions = np.flatnonzero(window == peak_value)
            last_peak = start + int(peak_positions[-1])
            out[i] = i - last_peak
        return pd.Series(out, index=s.index)
    x["pullback_duration"] = grouped["close"].transform(_duration_since_peak)
    x["amount_ma20"] = grouped["amount"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    x["amount_ma5"] = grouped["amount"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    x["LQ"] = x["amount"] / x["amount_ma20"]
    x["amount_ma5_to_ma20"] = x["amount_ma5"] / x["amount_ma20"]
    x["volume_ma5"] = grouped["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    x["volume_ma20"] = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    x["volume_ma5_to_ma20"] = x["volume_ma5"] / x["volume_ma20"]
    x["price_up_volume_confirmation"] = (
        (x["close"] > grouped["close"].shift(1)) &
        (x["amount"] > x["amount_ma20"])
    )
    x["high_volume_stall"] = (
        (x["amount"] > 1.5 * x["amount_ma20"]) &
        (x["close"] <= grouped["close"].shift(1))
    )
    x["low_volume_pullback"] = (
        (x["close"] < grouped["close"].shift(1)) &
        (x["volume"] < x["volume_ma5"])
    )
    return x


def _pit_frame(frame: pd.DataFrame, decision_time, name: str) -> pd.DataFrame:
    if "available_time" not in frame.columns:
        raise ValueError(f"{name} requires available_time")
    out = frame.copy()
    available = pd.to_datetime(out["available_time"], errors="coerce")
    if available.isna().any() or available.dt.tz is None:
        raise ValueError(f"{name}.available_time must be valid and timezone-aware")
    out["available_time"] = available.dt.tz_convert("UTC")
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    return out.loc[out["available_time"] <= cutoff.tz_convert("UTC")].copy()


def industry_relative_strength(stock_industry_returns: pd.DataFrame,
                               market_returns: pd.DataFrame,
                               decision_time=None) -> pd.DataFrame:
    """Return 5d/20d industry excess returns with PIT-validated source rows."""
    required = {"trade_date", "industry_code", "return_5", "return_20"}
    if required - set(stock_industry_returns.columns):
        raise ValueError("industry returns missing required columns")
    if {"trade_date", "market_return_5", "market_return_20"} - set(market_returns.columns):
        raise ValueError("market returns missing required columns")
    if decision_time is None:
        raise ValueError("decision_time is required for PIT industry strength")
    ind = _pit_frame(stock_industry_returns, decision_time, "industry returns")
    market = _pit_frame(market_returns, decision_time, "market returns")
    if ind.duplicated(["trade_date", "industry_code"]).any():
        raise ValueError("duplicate point-in-time industry rows")
    if market.duplicated(["trade_date"]).any():
        raise ValueError("duplicate point-in-time market rows")
    x = ind.merge(market, on="trade_date", how="left", validate="many_to_one")
    x["industry_excess_5"] = x["return_5"] - x["market_return_5"]
    x["industry_excess_20"] = x["return_20"] - x["market_return_20"]
    return x


def residual_strength(stock_returns: pd.DataFrame, market_returns: pd.DataFrame,
                      industry_returns: pd.DataFrame, window: int = 60,
                      min_periods: int = 40, decision_time=None) -> pd.DataFrame:
    """Compute 5d residual return using beta fitted on prior daily returns.

    Beta regression uses non-overlapping one-session observations from dates
    strictly before the target date; target residual applies those betas to
    the target's 5-session stock, market and industry returns.
    """
    required = {"symbol", "trade_date", "industry_code", "return_5", "daily_return"}
    if required - set(stock_returns.columns):
        raise ValueError("stock_returns missing required columns")
    for frame, cols in ((market_returns, {"trade_date", "market_return", "market_return_5"}),
                        (industry_returns, {"trade_date", "industry_return", "industry_return_5", "industry_code"})):
        if cols - set(frame.columns):
            raise ValueError("benchmark return frame missing required columns")
    if decision_time is None:
        raise ValueError("decision_time is required for PIT residual strength")
    stocks = _pit_frame(stock_returns, decision_time, "stock returns")
    markets = _pit_frame(market_returns, decision_time, "market returns")
    industries = _pit_frame(industry_returns, decision_time, "industry returns")
    if markets.duplicated(["trade_date"]).any():
        raise ValueError("duplicate point-in-time market rows")
    if industries.duplicated(["trade_date", "industry_code"]).any():
        raise ValueError("duplicate point-in-time industry rows")
    x = stocks.merge(markets, on="trade_date", how="left", validate="many_to_one")
    x = x.merge(industries, on=["trade_date", "industry_code"], how="left", validate="many_to_one")
    x = x.sort_values(["symbol", "trade_date"])
    output = []
    for symbol, group in x.groupby("symbol"):
        g = group.copy()
        residual = []
        for i, row in enumerate(g.itertuples(index=False)):
            hist = g.iloc[max(0, i-window):i][["market_return", "industry_return", "daily_return"]].dropna()
            current = [row.return_5, row.market_return_5, row.industry_return_5]
            if len(hist) < min_periods or not np.isfinite(current).all():
                residual.append(np.nan)
                continue
            design = np.column_stack([np.ones(len(hist)), hist.market_return, hist.industry_return])
            beta = np.linalg.lstsq(design, hist.daily_return.to_numpy(), rcond=None)[0]
            residual.append(float(row.return_5 - beta[0]*5 - beta[1]*row.market_return_5 - beta[2]*row.industry_return_5))
        g["residual_return_5"] = residual
        output.append(g)
    return pd.concat(output, ignore_index=True) if output else x.assign(residual_return_5=np.nan)

def rank_factor(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Cross-sectional percentile rank; NaN remains missing and is never imputed."""
    return values.rank(pct=True, ascending=not higher_is_better, method="average") * 100
