"""Industry mainline strength and lifecycle observations."""
from __future__ import annotations

import pandas as pd


REQUIRED = {"trade_date", "industry_code", "industry_return_5d",
            "market_return_5d", "turnover", "market_turnover",
            "expected_industry_count"}


def classify_mainlines(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank relative strength, turnover share expansion and breadth by date.

    Output lifecycle is a descriptive cross-sectional heuristic. It requires
    genuine daily industry observations; current membership snapshots alone
    are not historical constituents and cannot establish historical PIT.
    """
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError("missing mainline fields: " + ",".join(sorted(missing)))
    out = frame.copy()
    for col in ("industry_return_5d", "market_return_5d", "turnover", "market_turnover"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[list(REQUIRED)].isna().any().any():
        raise ValueError("mainline input contains missing required observations")
    if (out[["turnover", "market_turnover"]] < 0).any().any():
        raise ValueError("turnover cannot be negative")
    if (out.expected_industry_count.isna().any()
            or (pd.to_numeric(out.expected_industry_count, errors="coerce") <= 0).any():
        raise ValueError("expected_industry_count must be positive")
    out["expected_industry_count"] = pd.to_numeric(
        out["expected_industry_count"], errors="raise"
    ).astype(int)
    if out.duplicated(["trade_date", "industry_code"]).any():
        raise ValueError("duplicate industry/date rows are not allowed")
    observed_counts = out.groupby("trade_date")["industry_code"].nunique()
    expected_counts = out.groupby("trade_date")["expected_industry_count"].nunique()
    if expected_counts.ne(1).any() or (
        observed_counts != out.groupby("trade_date")["expected_industry_count"].first()
    ).any():
        raise ValueError("incomplete or inconsistent industry peer universe")
    if (out.market_turnover <= 0).any():
        raise ValueError("market_turnover must be positive")
    out["relative_strength_5d"] = out.industry_return_5d - out.market_return_5d
    out["turnover_share"] = out.turnover / out.market_turnover
    out = out.sort_values(["industry_code", "trade_date"]).reset_index(drop=True)
    out["turnover_share_change_3d"] = out.groupby("industry_code").turnover_share.diff(3)
    out["relative_strength_change_3d"] = out.groupby("industry_code").relative_strength_5d.diff(3)
    out["strength_percentile"] = out.groupby("trade_date").relative_strength_5d.rank(pct=True)
    out["mainline_phase"] = "TRANSITION"
    ready = out.turnover_share_change_3d.notna() & out.relative_strength_change_3d.notna()
    ignition = ready & (out.strength_percentile >= 0.8) & (out.turnover_share_change_3d > 0) & (out.relative_strength_change_3d > 0)
    decay = ready & (out.strength_percentile < 0.5) & (out.turnover_share_change_3d < 0) & (out.relative_strength_change_3d < 0)
    peak = ready & (out.strength_percentile >= 0.8) & (out.turnover_share_change_3d <= 0) & (out.relative_strength_change_3d <= 0)
    out.loc[~ready, "mainline_phase"] = "UNKNOWN"
    out.loc[ignition, "mainline_phase"] = "IGNITION"
    out.loc[peak, "mainline_phase"] = "PEAK"
    out.loc[decay, "mainline_phase"] = "DECAY"
    out["historical_pit_eligible"] = False
    return out
