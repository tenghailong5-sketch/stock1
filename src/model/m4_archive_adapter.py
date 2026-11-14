"""Forward-only adapters for retrieval-time index archives.

Historical index bars fetched in one batch become usable only after that
batch's real retrieval timestamp. These adapters never backdate availability
or mark backfilled rows historical-PIT eligible.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


_REQUIRED = {
    "trade_date", "close", "available_time", "source_id",
    "source_revision", "availability_basis", "historical_pit_eligible",
}


def _index_returns(
    bars: pd.DataFrame,
    *,
    identity_col: str,
    decision_time,
    output_5: str,
    output_20: str,
) -> pd.DataFrame:
    if decision_time is None:
        raise ValueError("decision_time is required")
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    if bars is None or not isinstance(bars, pd.DataFrame):
        raise ValueError("index bars must be a DataFrame")
    missing = (_REQUIRED | {identity_col}) - set(bars.columns)
    if missing:
        raise ValueError("index bars missing required columns: " + ",".join(sorted(missing)))

    x = bars.copy()
    available = pd.to_datetime(x["available_time"], errors="coerce")
    if available.isna().any() or available.dt.tz is None:
        raise ValueError("index available_time must be valid and timezone-aware")
    x["available_time"] = available.dt.tz_convert("UTC")
    x = x.loc[x["available_time"] <= cutoff.tz_convert("UTC")].copy()
    if x.empty:
        return pd.DataFrame(columns=[
            "trade_date", identity_col, "daily_return", output_5, output_20, "available_time",
            "source_id", "source_revision", "availability_basis",
            "historical_pit_eligible",
        ])

    identity = x[identity_col]
    if identity.isna().any() or identity.astype(str).str.strip().eq("").any():
        raise ValueError(f"invalid {identity_col}")
    x[identity_col] = identity.astype(str).str.strip()
    x["trade_date"] = pd.to_datetime(x["trade_date"], errors="coerce")
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    if x["trade_date"].isna().any() or x["close"].isna().any():
        raise ValueError("invalid index trade_date or close")
    if not np.isfinite(x["close"].to_numpy(dtype=float)).all() or (x["close"] <= 0).any():
        raise ValueError("index close must be finite and positive")
    for col in ("source_id", "source_revision", "availability_basis"):
        if x[col].isna().any() or x[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"invalid index {col}")
        x[col] = x[col].astype(str).str.strip()
    if not x["availability_basis"].eq("retrieval_time_only").all():
        raise ValueError("backfilled index bars require retrieval_time_only availability")
    pit = x["historical_pit_eligible"].map(
        lambda value: value is True or str(value).strip().lower() in {"true", "1", "yes"}
    )
    if x["historical_pit_eligible"].isna().any() or pit.any():
        raise ValueError("retrieved historical index bars must remain PIT-ineligible")
    decision_date = cutoff.tz_convert("Asia/Shanghai").date()
    if (x["trade_date"].dt.date > decision_date).any():
        raise ValueError("future index trade_date relative to decision_time")
    if x.duplicated([identity_col, "trade_date"]).any():
        raise ValueError("duplicate index identity/trade_date rows")

    results = []
    for identity_value, group in x.sort_values([identity_col, "trade_date"]).groupby(identity_col):
        if group["source_id"].nunique() != 1:
            raise ValueError("mixed source_id within an index series")
        g = group.reset_index(drop=True).copy()
        g["daily_return"] = g["close"].pct_change(1)
        g[output_5] = g["close"].pct_change(5)
        g[output_20] = g["close"].pct_change(20)
        # The 20-session return depends on the full 21-bar path. Stamp the
        # maximum actual availability time and hash all revisions in that path.
        avail = []
        revisions = []
        for i in range(len(g)):
            start = max(0, i - 20)
            window = g.iloc[start:i + 1]
            avail.append(window["available_time"].max())
            revisions.append(hashlib.sha256(
                "|".join(window["source_revision"].astype(str)).encode("utf-8")
            ).hexdigest())
        g["available_time"] = avail
        g["source_revision"] = revisions
        g[identity_col] = identity_value
        g["historical_pit_eligible"] = False
        g = g.loc[g[output_5].notna() & g[output_20].notna()].copy()
        results.append(g[[
            "trade_date", identity_col, output_5, output_20, "available_time",
            "source_id", "source_revision", "availability_basis",
            "historical_pit_eligible",
        ]])
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def build_industry_index_returns(index_bars: pd.DataFrame, *, decision_time) -> pd.DataFrame:
    """Build industry return_5/return_20 rows visible at a forward decision."""
    return _index_returns(
        index_bars, identity_col="industry_code", decision_time=decision_time,
        output_5="return_5", output_20="return_20",
    )


def build_market_index_returns(index_bars: pd.DataFrame, *, decision_time) -> pd.DataFrame:
    """Build market_return_5/market_return_20 rows from one market index series."""
    if index_bars is None or "index_symbol" not in index_bars.columns:
        raise ValueError("market index bars require index_symbol")
    if index_bars["index_symbol"].astype(str).nunique() != 1:
        raise ValueError("market return adapter requires exactly one benchmark index")
    out = _index_returns(
        index_bars, identity_col="index_symbol", decision_time=decision_time,
        output_5="market_return_5", output_20="market_return_20",
    )
    return out.rename(columns={"daily_return": "market_return"})


def build_stock_daily_returns(stock_bars: pd.DataFrame, *, decision_time) -> pd.DataFrame:
    """Build stock daily/5d/20d returns from retrieval-time-only stock archives."""
    return _index_returns(
        stock_bars, identity_col="symbol", decision_time=decision_time,
        output_5="return_5", output_20="return_20",
    )
