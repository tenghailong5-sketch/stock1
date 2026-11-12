"""M4 point-in-time factor primitives from FHM-100 baseline.

This module computes raw factor observations. Cross-sectional percentile ranks
must be calculated within the correct contemporaneous peer universe. It does
not fabricate catalyst, capital-flow, turnover, or limit-event data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_BARS = {"symbol", "trade_date", "open", "high", "low", "close",
                 "volume", "amount", "available_time", "source_id", "source_revision"}


def _numeric(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def price_structure(bars: pd.DataFrame, decision_time=None) -> pd.DataFrame:
    """Compute price factors using only bars available by decision_time.

    A timezone-aware decision_time is mandatory; rows not yet available at
    that instant are excluded before any rolling feature is calculated.
    """
    missing = REQUIRED_BARS - set(bars.columns)
    if missing:
        raise ValueError("bars missing: " + ",".join(sorted(missing)))
    if decision_time is None:
        raise ValueError("decision_time is required for point-in-time price factors")
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    x = _numeric(bars, ["open", "high", "low", "close", "volume", "amount"])
    x["trade_date"] = pd.to_datetime(x["trade_date"], errors="coerce")
    raw_available = pd.to_datetime(x["available_time"], errors="coerce")
    if x["trade_date"].isna().any() or raw_available.isna().any():
        raise ValueError("invalid trade_date or available_time")
    if raw_available.dt.tz is None:
        raise ValueError("available_time must be timezone-aware")
    x["available_time"] = raw_available.dt.tz_convert("UTC")
    cutoff_utc = cutoff.tz_convert("UTC")
    x = x.loc[x["available_time"] <= cutoff_utc].copy()
    for column in ("source_id", "source_revision"):
        values = x[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"invalid {column}")
        x[column] = values.astype(str).str.strip()
    x = x.sort_values(["symbol", "trade_date"]).copy()
    if x["symbol"].isna().any() or x["symbol"].astype(str).str.strip().eq("").any():
        raise ValueError("invalid symbol values")
    x["symbol"] = x["symbol"].astype(str).str.strip()
    if x.duplicated(["symbol", "trade_date"]).any():
        raise ValueError("duplicate symbol/trade_date bars")
    # A row dated after the exchange-local decision date is a future observation,
    # even if a malformed provider timestamp claims it was already available.
    decision_date = cutoff.tz_convert("Asia/Shanghai").date()
    if (x["trade_date"].dt.date > decision_date).any():
        raise ValueError("future trade_date relative to decision_time")
    if x[["open", "high", "low", "close", "volume", "amount"]].isna().any().any():
        raise ValueError("invalid OHLCV/amount values")
    if not np.isfinite(x[["open", "high", "low", "close", "volume", "amount"]].to_numpy(dtype=float)).all():
        raise ValueError("non-finite OHLCV/amount values")
    if (x[["volume", "amount"]] < 0).any().any():
        raise ValueError("negative volume or amount")
    if ((x[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (x["high"] < x[["open", "low", "close"]].max(axis=1))
        | (x["low"] > x[["open", "high", "close"]].min(axis=1))).any():
        raise ValueError("inconsistent OHLC values")
    span = (x["high"] - x["low"]).where(lambda v: v > 0)
    x["CLV"] = (x["close"] - x["low"]) / span
    x["BodyStrength"] = (x["close"] - x["open"]) / span
    x["upper_shadow_ratio"] = (x["high"] - x[["open", "close"]].max(axis=1)) / span

    # R2.1-02: raw OHLC is for execution and same-session candle geometry only.
    # Cross-session momentum/breakout/pullback must use a verified PIT total-return
    # price series built from the point-in-time corporate-action ledger. Never
    # silently substitute today's raw close history across ex-dates.
    pit_columns = {"pit_total_return_close", "pit_total_return_high",
                   "pit_total_return_available_time", "corporate_action_status"}
    if not pit_columns.issubset(x.columns):
        x["cross_session_price_status"] = "MISSING_PIT_TOTAL_RETURN_FIELDS"
        x["return_5"] = np.nan
        x["return_10"] = np.nan
        x["return_20"] = np.nan
        x["high20_prev"] = np.nan
        x["breakout20"] = np.nan
        x["recent_high20"] = np.nan
        x["Pullback"] = np.nan
        x["pullback_duration"] = np.nan
    else:
        x["pit_total_return_close"] = pd.to_numeric(x["pit_total_return_close"], errors="coerce")
        x["pit_total_return_high"] = pd.to_numeric(x["pit_total_return_high"], errors="coerce")
        try:
            pit_available = pd.to_datetime(x["pit_total_return_available_time"], errors="coerce")
            pit_time_valid = bool(
                pit_available.notna().all() and pit_available.dt.tz is not None
            )
            if pit_time_valid:
                x["pit_total_return_available_time"] = pit_available.dt.tz_convert("UTC")
        except (AttributeError, TypeError, ValueError):
            pit_time_valid = False
        grouped = x.groupby("symbol", group_keys=False)
        x["cross_session_price_status"] = "READY"
        for symbol, indexes in grouped.groups.items():
            group = x.loc[indexes]
            close_values = group["pit_total_return_close"].to_numpy(dtype=float)
            high_values = group["pit_total_return_high"].to_numpy(dtype=float)
            verified = group["corporate_action_status"].eq("VERIFIED").all()
            pit_available_valid = (
                pit_time_valid
                and group["pit_total_return_available_time"].notna().all()
                and (group["pit_total_return_available_time"] <= cutoff_utc).all()
            )
            valid_prices = (
                np.isfinite(close_values).all() and np.isfinite(high_values).all()
                and (close_values > 0).all() and (high_values > 0).all()
            )
            if not (verified and valid_prices and pit_available_valid):
                x.loc[indexes, "cross_session_price_status"] = (
                    "UNVERIFIED_CORPORATE_ACTIONS" if not verified
                    else "PIT_TOTAL_RETURN_NOT_AVAILABLE_AT_DECISION" if not pit_available_valid
                    else "INVALID_PIT_TOTAL_RETURN_SERIES"
                )
                for column in ("return_5", "return_10", "return_20", "high20_prev",
                               "breakout20", "recent_high20", "Pullback", "pullback_duration"):
                    x.loc[indexes, column] = np.nan
                continue
            symbol_rows = x.loc[indexes].sort_values("trade_date")
            close = symbol_rows["pit_total_return_close"]
            high = symbol_rows["pit_total_return_high"]
            for n in (5, 10, 20):
                x.loc[symbol_rows.index, f"return_{n}"] = close.pct_change(n)
            high20_prev = high.shift(1).rolling(20, min_periods=20).max()
            recent_high20 = high.rolling(20, min_periods=20).max()
            x.loc[symbol_rows.index, "high20_prev"] = high20_prev
            x.loc[symbol_rows.index, "breakout20"] = close / high20_prev - 1
            x.loc[symbol_rows.index, "recent_high20"] = recent_high20
            x.loc[symbol_rows.index, "Pullback"] = close / recent_high20 - 1

            # Sessions elapsed since the most recent 20-session total-return
            # closing high. Zero means the current close is at a new peak.
            values = close.to_numpy(dtype=float)
            duration = np.full(len(values), np.nan)
            for i in range(len(values)):
                start = max(0, i - 19)
                window = values[start:i + 1]
                if not np.isfinite(window).all():
                    continue
                peak_positions = np.flatnonzero(window == np.max(window))
                duration[i] = i - (start + int(peak_positions[-1]))
            x.loc[symbol_rows.index, "pullback_duration"] = duration

    grouped = x.groupby("symbol", group_keys=False)
    x["amount_ma20"] = grouped["amount"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    x["amount_ma5"] = grouped["amount"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    # Zero rolling denominators occur in all-suspended/zero-activity windows.
    # They are undefined ratios, not infinity and not evidence of extreme liquidity.
    x["LQ"] = (x["amount"] / x["amount_ma20"]).where(x["amount_ma20"] > 0)
    x["amount_ma5_to_ma20"] = (
        x["amount_ma5"] / x["amount_ma20"]
    ).where(x["amount_ma20"] > 0)
    x["volume_ma5"] = grouped["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    x["volume_ma20"] = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    x["volume_ma5_to_ma20"] = (
        x["volume_ma5"] / x["volume_ma20"]
    ).where(x["volume_ma20"] > 0)

    # These price/volume confirmations compare different trading sessions.
    # Raw OHLC can jump mechanically across ex-dates, so use only the verified
    # point-in-time total-return close. If it is unavailable, preserve NA rather
    # than silently turning an unknown condition into False.
    comparison_close = pd.to_numeric(
        x.get("pit_total_return_close", pd.Series(np.nan, index=x.index)),
        errors="coerce",
    ).where(x["cross_session_price_status"].eq("READY"))
    previous_comparison_close = comparison_close.groupby(x["symbol"]).shift(1)
    price_comparable = comparison_close.notna() & previous_comparison_close.notna()
    amount_baseline_valid = x["amount_ma20"].notna() & (x["amount_ma20"] > 0)
    volume_baseline_valid = x["volume_ma5"].notna()

    def _nullable_flag(condition: pd.Series, valid: pd.Series) -> pd.Series:
        result = pd.Series(pd.NA, index=x.index, dtype="boolean")
        valid = valid.fillna(False)
        result.loc[valid] = condition.loc[valid].astype(bool)
        return result

    x["price_up_volume_confirmation"] = _nullable_flag(
        (comparison_close > previous_comparison_close)
        & (x["amount"] > x["amount_ma20"]),
        price_comparable & amount_baseline_valid,
    )
    x["high_volume_stall"] = _nullable_flag(
        (x["amount"] > 1.5 * x["amount_ma20"])
        & (comparison_close <= previous_comparison_close),
        price_comparable & amount_baseline_valid,
    )
    x["low_volume_pullback"] = _nullable_flag(
        (comparison_close < previous_comparison_close)
        & (x["volume"] < x["volume_ma5"]),
        price_comparable & volume_baseline_valid,
    )
    return x


def _pit_frame(frame: pd.DataFrame, decision_time, name: str,
               identity_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    required = {"available_time", "trade_date", "source_id", "source_revision", *identity_cols}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: " + ",".join(sorted(missing)))
    out = frame.copy()
    available = pd.to_datetime(out["available_time"], errors="coerce")
    if available.isna().any() or available.dt.tz is None:
        raise ValueError(f"{name}.available_time must be valid and timezone-aware")
    out["available_time"] = available.dt.tz_convert("UTC")
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    # First discard rows not yet known. Then validate the dates and identifiers
    # of every row that could actually enter this decision snapshot.
    out = out.loc[out["available_time"] <= cutoff.tz_convert("UTC")].copy()
    for column in ("source_id", "source_revision"):
        values = out[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} contains invalid {column}")
        out[column] = values.astype(str).str.strip()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    if out["trade_date"].isna().any():
        raise ValueError(f"{name} contains invalid trade_date")
    decision_date = cutoff.tz_convert("Asia/Shanghai").date()
    if (out["trade_date"].dt.date > decision_date).any():
        raise ValueError(f"{name} contains future trade_date relative to decision_time")
    for column in identity_cols:
        values = out[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} contains invalid {column}")
        out[column] = values.astype(str).str.strip()
    return out


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
    ind = _pit_frame(stock_industry_returns, decision_time, "industry returns", ("industry_code",))
    market = _pit_frame(market_returns, decision_time, "market returns")
    if ind.duplicated(["trade_date", "industry_code"]).any():
        raise ValueError("duplicate point-in-time industry rows")
    if market.duplicated(["trade_date"]).any():
        raise ValueError("duplicate point-in-time market rows")
    x = ind.merge(market, on="trade_date", how="left", validate="many_to_one")
    # Provider adapters may emit numeric columns as strings. Normalize at the
    # factor boundary and keep malformed/non-finite observations missing so the
    # downstream coverage audit, rather than an arithmetic exception or an
    # invented value, determines whether the cross-section is usable.
    numeric_columns = [
        "return_5", "return_20", "market_return_5", "market_return_20",
    ]
    for column in numeric_columns:
        values = pd.to_numeric(x[column], errors="coerce")
        x[column] = values.where(np.isfinite(values))
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
    stocks = _pit_frame(stock_returns, decision_time, "stock returns", ("symbol", "industry_code"))
    markets = _pit_frame(market_returns, decision_time, "market returns")
    industries = _pit_frame(industry_returns, decision_time, "industry returns", ("industry_code",))
    if stocks.duplicated(["symbol", "trade_date"]).any():
        raise ValueError("duplicate point-in-time stock rows")
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
            hist = g.iloc[max(0, i-window):i][["market_return", "industry_return", "daily_return"]].copy()
            hist = hist.apply(pd.to_numeric, errors="coerce")
            hist_values = hist.to_numpy(dtype=float)
            hist = hist.loc[np.isfinite(hist_values).all(axis=1)]
            current = pd.to_numeric(
                pd.Series([row.return_5, row.market_return_5, row.industry_return_5]),
                errors="coerce",
            ).to_numpy(dtype=float)
            if len(hist) < min_periods or not np.isfinite(current).all():
                residual.append(np.nan)
                continue
            design = np.column_stack([np.ones(len(hist)), hist.market_return, hist.industry_return])
            # Market and industry returns must independently identify both
            # exposures. A rank-deficient OLS fit has non-unique betas; using
            # its arbitrary least-squares solution would fabricate residual alpha.
            if np.linalg.matrix_rank(design) < design.shape[1]:
                residual.append(np.nan)
                continue
            beta = np.linalg.lstsq(design, hist.daily_return.to_numpy(dtype=float), rcond=None)[0]
            if not np.isfinite(beta).all():
                residual.append(np.nan)
                continue
            residual.append(float(current[0] - beta[1]*current[1] - beta[2]*current[2]))
        g["residual_return_5"] = residual
        output.append(g)
    return pd.concat(output, ignore_index=True) if output else x.assign(residual_return_5=np.nan)

def rank_factor(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Cross-sectional percentile rank; NaN remains missing and is never imputed."""
    return values.rank(pct=True, ascending=higher_is_better, method="average") * 100



def rank_factor_by_date(frame: pd.DataFrame, value_col: str,
                        date_col: str = "trade_date",
                        higher_is_better: bool = True) -> pd.Series:
    """Rank a factor only among peers sharing the same decision date."""
    if date_col not in frame or value_col not in frame:
        raise ValueError("date and factor columns are required")
    return frame.groupby(date_col, sort=False)[value_col].transform(
        lambda values: rank_factor(values, higher_is_better=higher_is_better)
    )
