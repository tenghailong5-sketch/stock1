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
    """Return the latest 5d/20d industry excess-return snapshot available at decision_time.

    Historical replay must call this function once per historical decision_time;
    this function intentionally does not emit a historical factor time series
    under one current availability cutoff.
    """
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
    # Do not expose older factor rows under one present-day availability cutoff.
    # Historical replay must call this function separately for each decision.
    if not x.empty:
        x = x.loc[x["trade_date"].eq(x["trade_date"].max())].copy()
    return x


def residual_strength(stock_returns: pd.DataFrame, market_returns: pd.DataFrame,
                      industry_returns: pd.DataFrame, window: int = 60,
                      min_periods: int = 40, decision_time=None) -> pd.DataFrame:
    """Compute the latest 5d residual return using trailing daily beta estimates.

    Beta regression uses non-overlapping one-session observations strictly
    before the latest target date. Only that latest decision snapshot is
    returned. Historical replay must call this function separately at each
    historical decision_time; older residuals under one present-day cutoff
    would leak later-retrieved observations into backtests.
    """
    required = {
        "symbol", "trade_date", "industry_code",
        "pit_total_return_return_5", "pit_total_return_daily_return",
        "pit_total_return_available_time", "corporate_action_status",
    }
    if required - set(stock_returns.columns):
        raise ValueError(
            "stock_returns missing verified PIT total-return fields: "
            + ",".join(sorted(required - set(stock_returns.columns)))
        )
    for frame, cols in ((market_returns, {"trade_date", "market_return", "market_return_5"}),
                        (industry_returns, {"trade_date", "industry_return", "industry_return_5", "industry_code"})):
        if cols - set(frame.columns):
            raise ValueError("benchmark return frame missing required columns")
    if decision_time is None:
        raise ValueError("decision_time is required for PIT residual strength")
    stocks = _pit_frame(stock_returns, decision_time, "stock returns", ("symbol", "industry_code"))
    if stocks.empty:
        stocks["residual_return_5"] = pd.Series(dtype=float)
        return stocks
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    pit_available = pd.to_datetime(stocks["pit_total_return_available_time"], errors="coerce")
    if pit_available.isna().any() or pit_available.dt.tz is None:
        raise ValueError("pit_total_return_available_time must be valid and timezone-aware")
    stocks["pit_total_return_available_time"] = pit_available.dt.tz_convert("UTC")
    # Raw close-to-close returns can contain mechanical ex-rights jumps. A
    # residual is usable only when the total-return series and corporate-action
    # ledger are explicitly verified and available by this decision cutoff.
    stocks = stocks.loc[
        stocks["corporate_action_status"].astype(str).str.strip().eq("VERIFIED")
        & (stocks["pit_total_return_available_time"] <= cutoff.tz_convert("UTC"))
    ].copy()
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
            hist = g.iloc[max(0, i-window):i][["market_return", "industry_return", "pit_total_return_daily_return"]].copy()
            hist = hist.apply(pd.to_numeric, errors="coerce")
            hist_values = hist.to_numpy(dtype=float)
            hist = hist.loc[np.isfinite(hist_values).all(axis=1)]
            current = pd.to_numeric(
                pd.Series([row.pit_total_return_return_5, row.market_return_5, row.industry_return_5]),
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
            beta = np.linalg.lstsq(design, hist.pit_total_return_daily_return.to_numpy(dtype=float), rcond=None)[0]
            if not np.isfinite(beta).all():
                residual.append(np.nan)
                continue
            residual.append(float(current[0] - beta[1]*current[1] - beta[2]*current[2]))
        g["residual_return_5"] = residual
        # Do not expose older target dates under the same present-day cutoff.
        output.append(g.tail(1))
    return pd.concat(output, ignore_index=True) if output else x.assign(residual_return_5=np.nan)


def _require_historical_pit_eligible(frame: pd.DataFrame, name: str) -> None:
    """Reject retrieval-time backfills masquerading as historical PIT observations."""
    required = {"historical_pit_eligible", "availability_basis", "available_time",
                "source_id", "source_revision"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing PIT eligibility fields: " + ",".join(sorted(missing)))
    if frame.empty:
        raise ValueError(f"{name} has no observations")
    eligible = frame["historical_pit_eligible"].map(
        lambda value: value is True or isinstance(value, np.bool_) and bool(value)
    )
    if not eligible.all():
        raise ValueError(f"{name} contains observations not certified historical PIT eligible")
    basis = frame["availability_basis"].astype(str).str.strip()
    if basis.eq("").any() or basis.eq("retrieval_time_only").any():
        raise ValueError(f"{name} has retrieval-time-only availability")
    for col in ("source_id", "source_revision"):
        values = frame[col]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} has invalid {col}")
    available = pd.to_datetime(frame["available_time"], errors="coerce", utc=True)
    if available.isna().any():
        raise ValueError(f"{name}.available_time contains invalid timestamps")


def build_m4_pit_factor_snapshot(
    stock_bars: pd.DataFrame,
    stock_returns: pd.DataFrame,
    industry_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
    *,
    decision_time,
    expected_universe_size: int,
    coverage_floor: float = 0.95,
    hard_gate_snapshot: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one audited M4 cross-section for one historical decision instant.

    Historical replay must call this once per decision_time. Percentiles are
    emitted only when the supplied symbol universe exactly matches the declared
    expected size and each factor meets its coverage floor. Gate flags are
    copied only from an independently archived, lineage-validated gate snapshot;
    this builder never infers event, liquidity, breakout or risk approvals.
    """
    if isinstance(expected_universe_size, bool) or not isinstance(expected_universe_size, int) or expected_universe_size < 1:
        raise ValueError("expected_universe_size must be a positive integer")
    if not 0 < float(coverage_floor) <= 1:
        raise ValueError("coverage_floor must be within (0,1]")
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")

    # All four factor inputs must carry original, auditable point-in-time
    # availability. Retrieval-time downloads of old history are not PIT data.
    for frame, name in (
        (stock_bars, "stock_bars"),
        (stock_returns, "stock_returns"),
        (industry_returns, "industry_returns"),
        (market_returns, "market_returns"),
    ):
        _require_historical_pit_eligible(frame, name)

    price = price_structure(stock_bars, decision_time=cutoff)
    price = price.sort_values("trade_date").groupby("symbol", as_index=False).tail(1).copy()
    if price["symbol"].duplicated().any():
        raise ValueError("duplicate latest price rows")
    stock_pit = _pit_frame(stock_returns, cutoff, "stock returns", ("symbol", "industry_code"))
    if stock_pit.empty:
        raise ValueError("no PIT stock return rows available")
    stock_pit = stock_pit.sort_values("trade_date").groupby("symbol", as_index=False).tail(1)
    industry = industry_relative_strength(industry_returns, market_returns, cutoff)
    residual = residual_strength(stock_returns, market_returns, industry_returns, decision_time=cutoff)
    if not industry.empty:
        industry = industry[["industry_code", "industry_excess_5", "industry_excess_20"]]
    else:
        industry = pd.DataFrame(columns=["industry_code", "industry_excess_5", "industry_excess_20"])
    residual = residual[["symbol", "industry_code", "trade_date", "residual_return_5"]]
    base = price.merge(
        stock_pit[["symbol", "industry_code", "trade_date", "available_time", "source_id", "source_revision"]],
        on="symbol", how="inner", suffixes=("", "_returns"), validate="one_to_one",
    )
    base = base.merge(industry, on="industry_code", how="left", validate="many_to_one")
    base = base.merge(
        residual[["symbol", "residual_return_5"]],
        on="symbol", how="left", validate="one_to_one",
    )
    base["trade_date"] = pd.to_datetime(base["trade_date"], errors="coerce")
    if base.empty:
        raise ValueError("no symbols have both PIT price and return observations")
    if base["trade_date"].nunique() != 1:
        raise ValueError("M4 snapshot inputs do not share one decision trade_date")
    base["industry_relative_strength_raw"] = base["industry_excess_5"]
    base["stock_residual_strength_raw"] = base["residual_return_5"]
    base["price_structure_raw"] = base["CLV"]
    # A transparent, observable activity/price-quality proxy; this is a factor,
    # not a hard liquidity approval. The actual risk/liquidity gate stays external.
    base["trade_quality_raw"] = base["amount_ma5_to_ma20"]
    base["pre_pullback_strength_raw"] = base["residual_return_5"]
    # Pullback structure is unavailable until verified PIT total-return geometry
    # and the required reversal confirmation are both present.
    base["pullback_reversal_structure_raw"] = np.nan

    mappings = {
        "industry_relative_strength_pct": ("industry_relative_strength_raw", True),
        "stock_residual_strength_pct": ("stock_residual_strength_raw", True),
        "price_structure_pct": ("price_structure_raw", True),
        "trade_quality_pct": ("trade_quality_raw", True),
        "pre_pullback_strength_pct": ("pre_pullback_strength_raw", True),
        "pullback_reversal_structure_pct": ("pullback_reversal_structure_raw", True),
    }
    audits = []
    for output, (raw, higher) in mappings.items():
        values = pd.to_numeric(base[raw], errors="coerce")
        values = values.where(np.isfinite(values))
        coverage = int(values.notna().sum()) / expected_universe_size
        universe_ok = len(base) == expected_universe_size
        status = "ranked" if universe_ok and coverage >= coverage_floor else (
            "universe_size_mismatch" if not universe_ok else "insufficient_factor_coverage"
        )
        base[output] = rank_factor(values, higher_is_better=higher) if status == "ranked" else np.nan
        audits.append({
            "factor_name": output, "trade_date": base["trade_date"].iloc[0],
            "expected_universe_size": expected_universe_size,
            "observed_universe_size": int(len(base)),
            "finite_factor_count": int(values.notna().sum()),
            "factor_coverage": coverage, "coverage_floor": float(coverage_floor),
            "status": status,
        })

    gate_fields = (
        "industry_top_30pct", "stock_residual_top_20pct", "amount_ma20_ge_100m",
        "breakout_or_confirmed_retest",
        "no_high_volume_stall_or_repeated_rejection",
        "pre_pullback_residual_positive", "pullback_between_0_5_and_4pct",
        "pullback_volume_le_80pct_prior5d", "stop_or_reversal_confirmed",
        "industry_not_weak", "no_material_negative_event",
        "liquidity_and_risk_pass",
    )
    for field in gate_fields:
        base[field] = pd.NA
    # A's explicit 20-session average traded-value floor is directly
    # derivable from the PIT OHLCV archive; unknown rolling windows stay NA.
    amount_ma20 = pd.to_numeric(base.get("amount_ma20"), errors="coerce")
    base["amount_ma20_ge_100m"] = pd.Series(
        pd.NA, index=base.index, dtype="boolean"
    )
    valid_amount = amount_ma20.notna() & np.isfinite(amount_ma20)
    base.loc[valid_amount, "amount_ma20_ge_100m"] = (
        amount_ma20.loc[valid_amount] >= 100_000_000
    )
    if hard_gate_snapshot is not None:
        required_gate = {"symbol", "available_time", "source_id", "source_revision", *gate_fields}
        missing = required_gate - set(hard_gate_snapshot.columns)
        if missing:
            raise ValueError("hard_gate_snapshot missing: " + ",".join(sorted(missing)))
        gates = hard_gate_snapshot.copy()
        available = pd.to_datetime(gates["available_time"], errors="coerce")
        if available.isna().any() or available.dt.tz is None:
            raise ValueError("hard_gate_snapshot.available_time must be timezone-aware")
        gates["available_time"] = available.dt.tz_convert("UTC")
        gates = gates.loc[gates["available_time"] <= cutoff.tz_convert("UTC")].copy()
        for col in ("source_id", "source_revision"):
            if gates[col].isna().any() or gates[col].astype(str).str.strip().eq("").any():
                raise ValueError("hard_gate_snapshot has invalid lineage")
        gates["symbol"] = gates["symbol"].astype(str).str.strip()
        if gates["symbol"].duplicated().any():
            raise ValueError("duplicate hard_gate_snapshot symbol")
        for field in gate_fields:
            if not gates[field].map(lambda v: isinstance(v, (bool, np.bool_))).all():
                raise ValueError("hard gate values must be explicit booleans: " + field)
        base = base.drop(columns=list(gate_fields)).merge(
            gates[["symbol", *gate_fields]], on="symbol", how="left", validate="one_to_one"
        )
    base["historical_pit_eligible"] = True
    base["available_time"] = pd.to_datetime(base["available_time"], utc=True)
    base["source_id"] = "derived:m4_pit_factor_builder"
    base["source_revision"] = "derived:v1"
    base["m4_decision_time"] = cutoff.isoformat()
    base["m4_factor_status"] = "ready"
    # Incomplete factor ranks remain NA and are visible in the audit; the M5
    # adapter will not turn them into scores.
    return base, pd.DataFrame(audits)

def rank_factor(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Rank finite numeric observations; malformed/non-finite values stay missing."""
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.where(np.isfinite(numeric))
    return numeric.rank(pct=True, ascending=higher_is_better, method="average") * 100



def rank_factor_by_date(frame: pd.DataFrame, value_col: str,
                        date_col: str = "trade_date",
                        higher_is_better: bool = True) -> pd.Series:
    """Rank a factor only among peers sharing the same decision date."""
    if date_col not in frame or value_col not in frame:
        raise ValueError("date and factor columns are required")
    return frame.groupby(date_col, sort=False)[value_col].transform(
        lambda values: rank_factor(values, higher_is_better=higher_is_better)
    )
