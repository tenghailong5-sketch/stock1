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


def price_structure(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute CLV, BodyStrength, 5/10/20d returns, breakout, wick and drawdown."""
    missing = REQUIRED_BARS - set(bars.columns)
    if missing:
        raise ValueError("bars missing: " + ",".join(sorted(missing)))
    x = _numeric(bars, ["open", "high", "low", "close", "volume", "amount"])
    x["trade_date"] = pd.to_datetime(x["trade_date"], errors="coerce")
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
    x["pullback_duration"] = grouped["close"].transform(
        lambda s: s.groupby((s == s.cummax()).cumsum()).cumcount()
    )
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


def industry_relative_strength(stock_industry_returns: pd.DataFrame,
                               market_returns: pd.DataFrame) -> pd.DataFrame:
    """Return 5d/20d industry excess returns; caller supplies PIT industry index returns."""
    required = {"trade_date", "industry_code", "return_5", "return_20"}
    if required - set(stock_industry_returns.columns):
        raise ValueError("industry returns missing required columns")
    if {"trade_date", "market_return_5", "market_return_20"} - set(market_returns.columns):
        raise ValueError("market returns missing required columns")
    x = stock_industry_returns.merge(market_returns, on="trade_date", how="left", validate="many_to_one")
    x["industry_excess_5"] = x["return_5"] - x["market_return_5"]
    x["industry_excess_20"] = x["return_20"] - x["market_return_20"]
    return x


def residual_strength(stock_returns: pd.DataFrame, market_returns: pd.DataFrame,
                      industry_returns: pd.DataFrame, window: int = 60,
                      min_periods: int = 40) -> pd.DataFrame:
    """Rolling two-factor OLS residuals; beta estimation excludes the target session."""
    required = {"symbol", "trade_date", "industry_code", "return_5"}
    if required - set(stock_returns.columns):
        raise ValueError("stock_returns missing required columns")
    for frame, cols in ((market_returns, {"trade_date", "market_return"}),
                        (industry_returns, {"trade_date", "industry_return", "industry_code"})):
        if cols - set(frame.columns):
            raise ValueError("benchmark return frame missing required columns")
    x = stock_returns.merge(market_returns, on="trade_date", how="left", validate="many_to_one")
    x = x.merge(industry_returns, on=["trade_date", "industry_code"], how="left", validate="many_to_one")
    x = x.sort_values(["symbol", "trade_date"])
    output = []
    for symbol, group in x.groupby("symbol"):
        g = group.copy()
        residual = []
        for i, row in enumerate(g.itertuples(index=False)):
            hist = g.iloc[max(0, i-window):i][["market_return", "industry_return", "return_5"]].dropna()
            if len(hist) < min_periods or pd.isna(row.return_5) or pd.isna(row.market_return) or pd.isna(row.industry_return):
                residual.append(np.nan)
                continue
            design = np.column_stack([np.ones(len(hist)), hist.market_return, hist.industry_return])
            beta = np.linalg.lstsq(design, hist.return_5.to_numpy(), rcond=None)[0]
            residual.append(float(row.return_5 - beta[0] - beta[1]*row.market_return - beta[2]*row.industry_return))
        g["residual_return_5"] = residual
        output.append(g)
    return pd.concat(output, ignore_index=True) if output else x.assign(residual_return_5=np.nan)


def rank_factor(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Cross-sectional percentile rank; NaN remains missing and is never imputed."""
    return values.rank(pct=True, ascending=higher_is_better, method="average") * 100
