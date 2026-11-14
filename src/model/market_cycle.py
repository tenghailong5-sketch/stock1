"""Observable A-share market emotion cycle; descriptive states, not trade signals."""
from __future__ import annotations

import pandas as pd


REQUIRED = {"trade_date", "breadth", "limit_up_count", "limit_down_count"}


def classify_market_cycle(frame: pd.DataFrame) -> pd.DataFrame:
    """Classify sessions as ICE/RECOVERY/HEAT/EXHAUSTION/UNKNOWN.

    Uses only same-session observable aggregates. Optional board-failure and
    promotion rates improve discrimination but remain missing when unavailable.
    Thresholds are transparent research heuristics, not fitted parameters.
    """
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError("missing market-cycle fields: " + ",".join(sorted(missing)))
    out = frame.copy().sort_values("trade_date").reset_index(drop=True)
    for col in ("breadth", "limit_up_count", "limit_down_count"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[list(REQUIRED - {"trade_date"})].isna().any().any():
        out["market_cycle"] = "UNKNOWN"
        out["cycle_reason"] = "required_market_observation_missing"
        return out
    if ((out.breadth < 0) | (out.breadth > 1)).any():
        raise ValueError("breadth must be within [0,1]")
    for col in ("limit_up_count", "limit_down_count"):
        if (out[col] < 0).any():
            raise ValueError(col + " cannot be negative")

    out["limit_balance"] = out.limit_up_count - out.limit_down_count
    out["breadth_change_3d"] = out.breadth.diff(3)
    out["limit_up_change_3d"] = out.limit_up_count.diff(3)
    states, reasons = [], []
    for row in out.itertuples():
        if pd.isna(row.breadth_change_3d) or pd.isna(row.limit_up_change_3d):
            states.append("UNKNOWN")
            reasons.append("three_session_warmup")
            continue
        board_fail = getattr(row, "board_failure_rate", float("nan"))
        promotion = getattr(row, "promotion_rate", float("nan"))
        if row.breadth <= 0.25 and row.limit_down_count >= row.limit_up_count:
            state, why = "ICE", "weak_breadth_and_limit_down_pressure"
        elif (row.breadth_change_3d > 0.12 and row.limit_up_change_3d > 0
              and row.limit_balance > 0):
            state, why = "RECOVERY", "breadth_and_limit_up_rebound"
        elif ((pd.notna(board_fail) and board_fail >= 0.45)
              or (pd.notna(promotion) and promotion < 0.25)
              or (row.breadth < 0.45 and row.limit_down_count > row.limit_up_count)):
            state, why = "EXHAUSTION", "board_failure_or_promotion_collapse"
        elif row.breadth >= 0.68 and row.limit_up_count >= 30:
            state, why = "HEAT", "broad_strength_and_many_limit_ups"
        else:
            state, why = "TRANSITION", "mixed_market_signals"
        states.append(state)
        reasons.append(why)
    out["market_cycle"] = states
    out["cycle_reason"] = reasons
    out["historical_pit_eligible"] = False
    return out
