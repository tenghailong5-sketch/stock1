"""Optional CNEquity adapter for historical SW mainline research.

The resulting history is explicitly research-only: historical membership and
industry bars reconstructed from a later local lake are not promoted to
historical PIT trading evidence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def _frame(value, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    if hasattr(value, "to_dicts"):
        return pd.DataFrame(value.to_dicts())
    raise TypeError(f"{name} loader result must be a DataFrame or Polars frame")


def _aware_utc(values: pd.Series, name: str) -> pd.Series:
    aware = values.map(
        lambda value: (
            pd.notna(value) and pd.Timestamp(value).tzinfo is not None
        )
    )
    if not bool(aware.all()):
        raise ValueError(f"{name} must contain timezone-aware timestamps")
    return pd.to_datetime(values, errors="coerce", utc=True)


def _normalize_returns(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        raise ValueError("industry index has no finite returns")
    # CNEquity sources may expose percentage points or decimal returns.
    # Daily index returns above 50 percentage points are not plausible here.
    scale = 100.0 if finite.abs().quantile(0.99) > 0.5 else 1.0
    result = numeric / scale
    if result.dropna().abs().gt(0.5).any():
        raise ValueError("industry daily return exceeds the sanity bound")
    return result


def build_cnequity_mainline_inputs(
    industry_index,
    market_index,
    *,
    expected_industry_count: int = 31,
    min_priced_coverage: float = 0.80,
    benchmark_symbol: str = "000001.SH",
) -> tuple[pd.DataFrame, dict]:
    """Build the daily input panel required by classify_mainlines().

    Uses SW level-1, amount-weighted industry-index returns and turnover.
    Market turnover is the sum of those same first-level industry amounts,
    matching the existing model's declared turnover-share denominator.
    """
    industries = _frame(industry_index, "industry_index")
    market = _frame(market_index, "index_bars")
    required_industry = {
        "trade_date", "industry_code", "level", "weighting", "ret",
        "amount", "n_members", "n_priced", "n_excluded",
        "source", "data_version", "fetched_at",
    }
    required_market = {
        "symbol", "trade_date", "close", "frequency",
        "source", "data_version", "fetched_at",
    }
    missing = required_industry - set(industries.columns)
    if missing:
        raise ValueError("industry_index missing: " + ",".join(sorted(missing)))
    missing = required_market - set(market.columns)
    if missing:
        raise ValueError("index_bars missing: " + ",".join(sorted(missing)))
    if expected_industry_count < 1:
        raise ValueError("expected_industry_count must be positive")
    if not 0 < float(min_priced_coverage) <= 1:
        raise ValueError("min_priced_coverage must be in (0, 1]")

    industries["trade_date"] = pd.to_datetime(industries["trade_date"], errors="coerce").dt.normalize()
    industries["industry_code"] = industries["industry_code"].astype(str).str.strip()
    industries["level"] = industries["level"].astype(str).str.strip().str.upper()
    industries["weighting"] = industries["weighting"].astype(str).str.strip().str.lower()
    industries = industries.loc[
        industries["level"].isin({"1", "L1"})
        & industries["weighting"].eq("amount")
    ].copy()
    if industries.empty:
        raise ValueError("no SW level-1 amount-weighted industry rows")
    if industries["trade_date"].isna().any() or industries["industry_code"].eq("").any():
        raise ValueError("industry_index contains invalid date or industry code")
    industries["ret"] = _normalize_returns(industries["ret"])
    industries["amount"] = pd.to_numeric(industries["amount"], errors="coerce")
    if industries["amount"].isna().any() or not np.isfinite(industries["amount"]).all():
        raise ValueError("industry amount contains missing or non-finite values")
    if industries["amount"].lt(0).any():
        raise ValueError("industry amount cannot be negative")
    for column in ("n_members", "n_priced", "n_excluded"):
        industries[column] = pd.to_numeric(industries[column], errors="coerce")
        if industries[column].isna().any() or not np.isfinite(industries[column]).all():
            raise ValueError(f"industry {column} contains missing or non-finite values")
        if industries[column].lt(0).any():
            raise ValueError(f"industry {column} cannot be negative")
    if industries["n_members"].le(0).any():
        raise ValueError("industry n_members must be positive")
    if (
        industries["n_priced"].gt(industries["n_members"]).any()
        or industries["n_priced"].add(industries["n_excluded"]).ne(industries["n_members"]).any()
    ):
        raise ValueError("industry member/priced/excluded counts are inconsistent")
    industries["member_coverage"] = industries["n_priced"] / industries["n_members"]
    low_coverage_rows = int(industries["member_coverage"].lt(min_priced_coverage).sum())
    all_industry_dates = set(industries["trade_date"].unique())
    industries = industries.loc[industries["member_coverage"] >= min_priced_coverage].copy()
    if industries.empty:
        raise ValueError("all SW level-1 industry rows are below priced-member coverage")
    # Rolling phase transitions assume consecutive market sessions. Remove an
    # entire industry's series if any session is missing; never bridge a gap.
    dates_per_industry = industries.groupby("industry_code")["trade_date"].nunique()
    complete_codes = set(dates_per_industry[dates_per_industry.eq(len(all_industry_dates))].index)
    incomplete_industries_excluded = int(industries["industry_code"].nunique() - len(complete_codes))
    industries = industries.loc[industries["industry_code"].isin(complete_codes)].copy()
    minimum_complete_industries = int(np.ceil(expected_industry_count * 0.90))
    if len(complete_codes) < minimum_complete_industries:
        raise ValueError(
            "complete industry histories below 90% peer coverage: "
            f"{len(complete_codes)} < {minimum_complete_industries}"
        )
    industries["fetched_at"] = _aware_utc(industries["fetched_at"], "industry_index.fetched_at")
    if industries["source"].isna().any() or industries["data_version"].isna().any():
        raise ValueError("industry_index source and data_version are required")
    if industries.duplicated(["trade_date", "industry_code"]).any():
        raise ValueError("duplicate SW level-1 industry/date rows")
    codes = industries["industry_code"].nunique()
    if codes > expected_industry_count:
        raise ValueError("observed SW level-1 universe exceeds expected industry count")

    market["symbol"] = market["symbol"].astype(str).str.strip().str.upper()
    market["frequency"] = market["frequency"].astype(str).str.strip().str.lower()
    market = market.loc[market["frequency"].eq("1d")].copy()
    allowed_symbols = {benchmark_symbol.upper(), benchmark_symbol.split(".")[0]}
    market = market.loc[market["symbol"].isin(allowed_symbols)].copy()
    if market.empty:
        raise ValueError("requested market benchmark is absent")
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce").dt.normalize()
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    if market["trade_date"].isna().any() or market["close"].isna().any():
        raise ValueError("market index contains invalid date or close")
    if not np.isfinite(market["close"]).all() or market["close"].le(0).any():
        raise ValueError("market index close must be finite and positive")
    market["fetched_at"] = _aware_utc(market["fetched_at"], "index_bars.fetched_at")
    if market["source"].isna().any() or market["data_version"].isna().any():
        raise ValueError("market index source and data_version are required")
    if market.duplicated(["trade_date"]).any():
        raise ValueError("duplicate market benchmark/date rows")
    market = market.sort_values("trade_date").copy()
    missing_market_dates = set(industries["trade_date"].unique()) - set(market["trade_date"].unique())
    if missing_market_dates:
        raise ValueError("market benchmark missing industry trading dates")
    market["market_return_1d"] = market["close"].pct_change()

    industries = industries.sort_values(["industry_code", "trade_date"]).copy()
    industries["industry_return_3d"] = industries.groupby("industry_code")["ret"].transform(
        lambda values: (1.0 + values).rolling(3, min_periods=3).apply(np.prod, raw=True) - 1.0
    )
    industries["industry_return_5d"] = industries.groupby("industry_code")["ret"].transform(
        lambda values: (1.0 + values).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1.0
    )
    market["market_return_3d"] = (
        (1.0 + market["market_return_1d"]).rolling(3, min_periods=3).apply(np.prod, raw=True) - 1.0
    )
    market["market_return_5d"] = (
        (1.0 + market["market_return_1d"]).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1.0
    )
    market_returns = market[["trade_date", "market_return_3d", "market_return_5d"]]
    panel = industries.merge(market_returns, on="trade_date", how="inner", validate="many_to_one")
    panel["turnover"] = panel["amount"]
    panel["market_turnover"] = panel.groupby("trade_date")["amount"].transform("sum")
    panel["expected_industry_count"] = int(expected_industry_count)
    panel = panel.dropna(subset=[
        "industry_return_3d", "industry_return_5d",
        "market_return_3d", "market_return_5d",
    ]).copy()
    panel["historical_pit_eligible"] = False
    panel["research_only"] = True
    panel["trading_authorization"] = False
    panel["availability_basis"] = "historical_reconstruction_from_cnequity_lake"
    panel["source_provenance"] = panel["source"].astype(str)
    panel["source_version"] = panel["data_version"].astype(str)
    panel["source_fetched_at"] = panel["fetched_at"]
    output_columns = [
        "trade_date", "industry_code", "industry_return_3d", "market_return_3d",
        "industry_return_5d", "market_return_5d", "turnover", "market_turnover",
        "expected_industry_count", "n_members", "n_priced", "n_excluded",
        "member_coverage", "historical_pit_eligible", "research_only",
        "trading_authorization", "availability_basis", "source_provenance",
        "source_version", "source_fetched_at",
    ]
    panel = panel[output_columns].sort_values(["industry_code", "trade_date"]).reset_index(drop=True)
    audit = {
        "status": "READY_RESEARCH_ONLY" if not panel.empty else "NO_ALIGNED_HISTORY",
        "row_count": int(len(panel)),
        "industry_count": int(panel["industry_code"].nunique()) if not panel.empty else 0,
        "expected_industry_count": int(expected_industry_count),
        "trade_date_count": int(panel["trade_date"].nunique()) if not panel.empty else 0,
        "start_date": panel["trade_date"].min().date().isoformat() if not panel.empty else None,
        "end_date": panel["trade_date"].max().date().isoformat() if not panel.empty else None,
        "benchmark_symbol": benchmark_symbol,
        "market_source": sorted(market["source"].astype(str).unique().tolist()),
        "market_data_versions": sorted(market["data_version"].astype(str).unique().tolist()),
        "market_fetched_at_max": market["fetched_at"].max().isoformat(),
        "weighting": "amount",
        "historical_pit_eligible": False,
        "trading_authorization": False,
        "min_priced_coverage": float(min_priced_coverage),
        "low_coverage_rows_excluded": low_coverage_rows,
        "incomplete_industries_excluded": incomplete_industries_excluded,
        "note": "Historical reconstruction supports lifecycle research only; it is not PIT trading evidence.",
    }
    return panel, audit


def load_cnequity_mainline_inputs(
    *,
    start: str,
    end: str,
    data_root: str | Path | None = None,
    loader: Callable | None = None,
    benchmark_symbol: str = "000001.SH",
    expected_industry_count: int = 31,
    min_priced_coverage: float = 0.80,
) -> tuple[pd.DataFrame, dict]:
    """Read CNEquity's local industry_index and benchmark index_bars datasets."""
    if loader is None:
        try:
            from cnequity.query import load as loader
        except ImportError as exc:
            raise RuntimeError(
                "CNEquity is not installed; install it and initialize its local data lake"
            ) from exc
    kwargs = {"start": start, "end": end}
    if data_root is not None:
        kwargs["data_root"] = str(data_root)
    industry = loader("industry_index", **kwargs)
    market_kwargs = dict(kwargs)
    market_kwargs["symbols"] = [benchmark_symbol]
    market = loader("index_bars", **market_kwargs)
    panel, audit = build_cnequity_mainline_inputs(
        industry, market,
        expected_industry_count=expected_industry_count,
        min_priced_coverage=min_priced_coverage,
        benchmark_symbol=benchmark_symbol,
    )
    audit.update({
        "provider": "CNEquity",
        "data_root_configured": data_root is not None,
        "requested_start": start,
        "requested_end": end,
    })
    return panel, audit
