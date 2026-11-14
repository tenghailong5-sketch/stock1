"""Industry mainline strength and lifecycle observations."""
from __future__ import annotations

import pandas as pd


DEFAULT_IGNITION_STRENGTH_MIN = 0.65


REQUIRED = {"trade_date", "industry_code", "industry_return_3d",
            "market_return_3d", "industry_return_5d", "market_return_5d",
            "turnover", "market_turnover", "expected_industry_count"}


def classify_mainlines(
    frame: pd.DataFrame, *,
    ignition_strength_min: float = DEFAULT_IGNITION_STRENGTH_MIN,
    ignition_turnover_min: float = 0.50,
) -> pd.DataFrame:
    """Rank relative strength, turnover share expansion and breadth by date.

    Output lifecycle is a descriptive cross-sectional heuristic. It requires
    genuine daily industry observations; current membership snapshots alone
    are not historical constituents and cannot establish historical PIT.
    """
    if not 0 <= ignition_strength_min <= 1:
        raise ValueError("ignition_strength_min must be within [0,1]")
    if not 0 <= ignition_turnover_min <= 1:
        raise ValueError("ignition_turnover_min must be within [0,1]")
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError("missing mainline fields: " + ",".join(sorted(missing)))
    out = frame.copy()
    for col in (
        "industry_return_3d", "market_return_3d",
        "industry_return_5d", "market_return_5d", "turnover", "market_turnover",
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[list(REQUIRED)].isna().any().any():
        raise ValueError("mainline input contains missing required observations")
    if (out[["turnover", "market_turnover"]] < 0).any().any():
        raise ValueError("turnover cannot be negative")
    if out.expected_industry_count.isna().any() or (
        pd.to_numeric(out.expected_industry_count, errors="coerce") <= 0
    ).any():
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
    if out.groupby("trade_date")["market_turnover"].nunique().ne(1).any():
        raise ValueError("market_turnover denominator must be consistent within each date")
    if (out.market_turnover <= 0).any():
        raise ValueError("market_turnover must be positive")
    out["relative_strength_3d"] = out.industry_return_3d - out.market_return_3d
    out["relative_strength_5d"] = out.industry_return_5d - out.market_return_5d
    out["turnover_share"] = out.turnover / out.market_turnover
    out["turnover_share_basis"] = "relative_share_of_sw_first_level_industry_index_turnover"
    out["turnover_share_basis"] = "relative_share_of_sw_first_level_industry_index_turnover"
    out = out.sort_values(["industry_code", "trade_date"]).reset_index(drop=True)
    out["turnover_share_change_1d"] = out.groupby("industry_code").turnover_share.diff(1)
    out["turnover_share_change_3d"] = out.groupby("industry_code").turnover_share.diff(3)
    out["relative_strength_change_2d"] = out.groupby("industry_code").relative_strength_3d.diff(2)
    out["relative_strength_change_3d"] = out.groupby("industry_code").relative_strength_5d.diff(3)
    out["strength_3d_percentile"] = out.groupby("trade_date").relative_strength_3d.rank(pct=True)
    out["strength_percentile"] = out.groupby("trade_date").relative_strength_5d.rank(pct=True)
    out["turnover_share_percentile"] = out.groupby("trade_date").turnover_share.rank(pct=True)
    ready = out.turnover_share_change_3d.notna() & out.relative_strength_change_3d.notna()
    out["mainline_phase"] = "UNKNOWN"
    out["phase_transition"] = False
    out["phase_reason"] = "three_session_warmup"
    ignition_ready = (
        out.turnover_share_change_1d.notna()
        & out.relative_strength_change_2d.notna()
    )
    ignition_candidate = (
        ignition_ready & (out.strength_3d_percentile >= ignition_strength_min)
        & (out.turnover_share_percentile >= ignition_turnover_min)
        & (out.relative_strength_3d > 0)
        & (out.turnover_share_change_1d > 0)
        & (out.relative_strength_change_2d > 0)
    )
    peak_candidate = (
        ready & (out.strength_percentile >= 0.8)
        & (out.turnover_share_percentile >= 0.65)
        & (out.relative_strength_5d > 0)
        & (out.turnover_share_change_3d <= 0)
        & (out.relative_strength_change_3d <= 0)
    )
    decay_candidate = (
        ready & (
            (out.strength_percentile < 0.5)
            | (out.turnover_share_percentile < 0.5)
            | (out.relative_strength_5d < 0)
        )
        & (out.turnover_share_change_3d < 0)
        & (out.relative_strength_change_3d < 0)
    )

    # Two consecutive observations confirm each phase transition. The phase
    # begins on the confirmation date; no retrospective backdating is allowed.
    for _, indexes in out.groupby("industry_code", sort=False).groups.items():
        state = "TRANSITION"
        ignition_streak = peak_streak = decay_streak = 0
        for index in indexes:
            if not bool(ready.loc[index]):
                out.at[index, "mainline_phase"] = "UNKNOWN"
                out.at[index, "phase_reason"] = "required_history_warmup"
                continue
            ignition_streak = ignition_streak + 1 if bool(ignition_candidate.loc[index]) else 0
            peak_streak = peak_streak + 1 if bool(peak_candidate.loc[index]) else 0
            decay_streak = decay_streak + 1 if bool(decay_candidate.loc[index]) else 0
            previous = state
            reason = "phase_state_persisted"
            if ignition_streak >= 2 and state in {"TRANSITION", "PEAK", "DECAY"}:
                state = "IGNITION"
                reason = "two_session_3d_strength_and_one_day_turnover_expansion"
            elif peak_streak >= 2 and state == "IGNITION":
                state = "PEAK"
                reason = "two_session_strength_plateau_with_turnover_contraction"
            elif decay_streak >= 2 and state in {"IGNITION", "PEAK"}:
                state = "DECAY"
                reason = "two_session_relative_strength_and_turnover_contraction"
            out.at[index, "mainline_phase"] = state
            out.at[index, "phase_transition"] = state != previous
            out.at[index, "phase_reason"] = reason
    out["historical_pit_eligible"] = False
    return out
