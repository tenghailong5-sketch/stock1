"""M3 market-state engine implementing the FHM-100 R1 addendum.

Inputs must be point-in-time index_daily observations. Percentiles use only the
trailing 60 observations including the current session. A state change is
confirmed only after two consecutive sessions; the confirmed state is usable
from the following session. Thresholds are research parameters, not validated
profitability claims.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = {
    "trade_date", "close", "ma20", "ma60", "breadth", "amount",
    "amount_ma20", "limit_up_cnt", "limit_down_cnt", "vol20",
}


def _rolling_percentile(values: pd.Series, window: int = 60) -> pd.Series:
    def rank_last(x):
        current = x[-1]
        return float(np.mean(x <= current)) if np.isfinite(current) else np.nan
    return values.astype(float).rolling(window, min_periods=window).apply(rank_last, raw=True)


def calculate_state_scores(index_daily: pd.DataFrame, window: int = 60,
                           decision_time=None) -> pd.DataFrame:
    """Calculate R1 StateScore from observations actually available at decision_time."""
    missing = (REQUIRED | {"available_time"}) - set(index_daily.columns)
    if missing:
        raise ValueError("index_daily missing: " + ",".join(sorted(missing)))
    if window != 60:
        raise ValueError("FHM-100 R1 specifies a 60-session percentile window")
    if decision_time is None:
        raise ValueError("decision_time is required for point-in-time market state")
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    x = index_daily.copy()
    available = pd.to_datetime(x["available_time"], errors="coerce")
    if available.isna().any() or available.dt.tz is None:
        raise ValueError("available_time must be valid and timezone-aware")
    x["available_time"] = available.dt.tz_convert("UTC")
    x = x.loc[x["available_time"] <= cutoff.tz_convert("UTC")].copy()
    x["trade_date"] = pd.to_datetime(x["trade_date"], errors="coerce")
    x = x.sort_values("trade_date").reset_index(drop=True)
    if x["trade_date"].isna().any() or x["trade_date"].duplicated().any():
        raise ValueError("trade_date must be valid and unique")
    numeric = ["close", "ma20", "ma60", "breadth", "amount", "amount_ma20",
               "limit_up_cnt", "limit_down_cnt", "vol20"]
    for col in numeric:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    if x[numeric].isna().any().any():
        # Keep malformed observations visible as missing; they cannot create a state.
        x.loc[x[numeric].isna().any(axis=1), numeric] = np.nan
    if ((x[["close", "ma20", "ma60", "amount_ma20"]] <= 0) & x[["close", "ma20", "ma60", "amount_ma20"]].notna()).any().any():
        raise ValueError("prices and amount_ma20 must be positive")
    if ((x[["breadth"]] < 0) | (x[["breadth"]] > 1)).any().any():
        raise ValueError("breadth must be in [0,1]")
    if (x[["limit_up_cnt", "limit_down_cnt", "amount", "vol20"]] < 0).any().any():
        raise ValueError("counts, amount and volatility must be non-negative")

    x["S_trend"] = (
        0.5 * (x["close"] > x["ma20"]).astype(float)
        + 0.5 * (x["ma20"] > x["ma60"]).astype(float)
    )
    trend_missing = x[["close", "ma20", "ma60"]].isna().any(axis=1)
    x.loc[trend_missing, "S_trend"] = np.nan
    x["limit_balance"] = x["limit_up_cnt"] - x["limit_down_cnt"]
    x["activity_ratio"] = x["amount"] / x["amount_ma20"]
    x["S_breadth"] = _rolling_percentile(x["breadth"], window)
    x["S_activity"] = _rolling_percentile(x["activity_ratio"], window)
    x["S_limit"] = _rolling_percentile(x["limit_balance"], window)
    x["S_vol"] = _rolling_percentile(x["vol20"], window)
    components = ["S_trend", "S_breadth", "S_activity", "S_limit", "S_vol"]
    x["StateScore"] = (
        0.35*x["S_trend"] + 0.25*x["S_breadth"]
        + 0.20*x["S_activity"] + 0.20*x["S_limit"] - 0.10*x["S_vol"]
    )
    x.loc[x[components].isna().any(axis=1), "StateScore"] = np.nan
    return x


def classify_raw_state(row) -> str:
    """Apply R1 thresholds; RISK_OFF takes precedence over RISK_ON."""
    fields = ["StateScore", "S_trend", "S_limit", "limit_up_cnt", "limit_down_cnt"]
    if any(pd.isna(row.get(k)) for k in fields):
        return "INSUFFICIENT_DATA"
    if (row["StateScore"] <= 0.35 or row["S_limit"] <= 0.20
            or row["limit_down_cnt"] > 0.5 * row["limit_up_cnt"]):
        return "RISK_OFF"
    if row["StateScore"] >= 0.60 and row["S_trend"] >= 0.5:
        return "RISK_ON"
    return "NEUTRAL"


def apply_state_cooldown(scored: pd.DataFrame) -> pd.DataFrame:
    """Two consecutive raw sessions confirm a switch; confirmed state starts T+1."""
    x = scored.copy().sort_values("trade_date").reset_index(drop=True)
    if "raw_market_state" not in x:
        x["raw_market_state"] = x.apply(classify_raw_state, axis=1)
    confirmed = []
    current = "NEUTRAL"
    pending = None
    streak = 0
    for raw in x["raw_market_state"]:
        if raw == "INSUFFICIENT_DATA":
            pending, streak = None, 0
            confirmed.append("INSUFFICIENT_DATA")
            continue
        if raw == current:
            pending, streak = None, 0
        elif raw == pending:
            streak += 1
        else:
            pending, streak = raw, 1
        confirmed.append(current)
        if streak >= 2:
            current = pending
            pending, streak = None, 0
    x["market_state"] = confirmed
    x["position_scale"] = x.market_state.map(
        {"RISK_ON": 1.0, "NEUTRAL": 0.7, "RISK_OFF": 0.0}
    )
    x["candidate_score_threshold"] = x.market_state.map(
        {"RISK_ON": 80, "NEUTRAL": 85, "RISK_OFF": np.nan}
    )
    return x


def market_environment_score(row) -> float:
    """R1 proposed 0-100 mapping; returns NaN when the state inputs are incomplete."""
    score = row.get("StateScore")
    state = row.get("market_state")
    if pd.isna(score) or state not in {"RISK_ON", "NEUTRAL", "RISK_OFF"}:
        return float("nan")
    if state == "RISK_ON":
        z = (score - 0.60) / (0.90 - 0.60)
        value = 75 + 25*z
    elif state == "NEUTRAL":
        z = (score - 0.35) / (0.60 - 0.35)
        value = 40 + 35*z
    else:
        z = score / 0.35
        value = 10 + 30*z
    return float(np.clip(value, 0, 100))
