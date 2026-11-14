"""Outcome-only retrospective validation of detected mainline phase transitions.

Forward returns are labels for evaluation only. They must never be passed into
classify_mainlines or any live candidate/scoring path. All outputs remain
non-PIT because the source industry history lacks original availability times.
"""
from __future__ import annotations

import pandas as pd


REQUIRED = {
    "trade_date", "industry_code", "mainline_phase", "phase_transition",
    "industry_close", "market_close",
}


def evaluate_mainline_transitions(frame: pd.DataFrame, *, horizons=(5, 10, 20)) -> tuple[pd.DataFrame, dict]:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError("missing validation fields: " + ",".join(sorted(missing)))
    if not horizons or any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in horizons):
        raise ValueError("horizons must contain positive integer trading-session counts")
    out = frame.copy()
    out["trade_date"] = pd.to_datetime(out.trade_date, errors="coerce")
    for col in ("industry_close", "market_close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[list(REQUIRED)].isna().any().any():
        raise ValueError("validation inputs contain missing required observations")
    if out.duplicated(["industry_code", "trade_date"]).any():
        raise ValueError("duplicate industry/date rows in validation input")
    if (out[["industry_close", "market_close"]] <= 0).any().any():
        raise ValueError("validation closes must be positive")
    out = out.sort_values(["industry_code", "trade_date"]).reset_index(drop=True)
    for horizon in horizons:
        out[f"industry_forward_return_{horizon}d"] = (
            out.groupby("industry_code").industry_close.shift(-horizon)
            .div(out.industry_close).sub(1)
        )
        out[f"market_forward_return_{horizon}d"] = (
            out.groupby("industry_code").market_close.shift(-horizon)
            .div(out.market_close).sub(1)
        )
        out[f"excess_forward_return_{horizon}d"] = (
            out[f"industry_forward_return_{horizon}d"]
            - out[f"market_forward_return_{horizon}d"]
        )
    events = out[out.phase_transition & out.mainline_phase.isin(
        ["IGNITION", "PEAK", "DECAY"]
    )].copy()
    events["historical_pit_eligible"] = False
    summary = {}
    for phase in ("IGNITION", "PEAK", "DECAY"):
        phase_rows = events[events.mainline_phase.eq(phase)]
        phase_metrics = {"detected_transitions": int(len(phase_rows))}
        for horizon in horizons:
            values = phase_rows[f"excess_forward_return_{horizon}d"].dropna()
            phase_metrics[f"evaluated_{horizon}d"] = int(len(values))
            phase_metrics[f"mean_excess_{horizon}d"] = (
                float(values.mean()) if len(values) else None
            )
            favorable = values.gt(0) if phase == "IGNITION" else values.lt(0)
            phase_metrics[f"directional_hit_rate_{horizon}d"] = (
                float(favorable.mean()) if len(values) else None
            )
        summary[phase] = phase_metrics
    summary["interpretation"] = (
        "forward-outcome validation only; not causal, not point-in-time, "
        "and not a substitute for manually adjudicated historical episode labels"
    )
    summary["historical_pit_eligible"] = False
    return events.reset_index(drop=True), summary


def compare_ignition_thresholds(
    frame: pd.DataFrame, *,
    candidates=((0.55, 0.50), (0.60, 0.50), (0.65, 0.50),
                (0.65, 0.60), (0.70, 0.50), (0.70, 0.65),
                (0.75, 0.60), (0.80, 0.65)),
    horizon: int = 10,
    train_fraction: float = 0.60,
) -> tuple[pd.DataFrame, dict]:
    """Compare ignition thresholds with a chronological train/test split.

    The training labels stop one full forward horizon before the split. The
    test period is never used to select thresholds. Forward returns are used
    only as outcome labels and never enter the classifier inputs.
    """
    from src.model.mainline_flows import classify_mainlines

    required = REQUIRED | {
        "industry_return_3d", "market_return_3d",
        "industry_return_3d", "market_return_3d",
        "industry_return_5d", "market_return_5d", "turnover",
        "market_turnover", "expected_industry_count",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("missing threshold research fields: " + ",".join(sorted(missing)))
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not 0.5 <= train_fraction <= 0.8:
        raise ValueError("train_fraction must be within [0.5,0.8]")
    base = frame.copy()
    base["trade_date"] = pd.to_datetime(base.trade_date, errors="coerce")
    if base.trade_date.isna().any():
        raise ValueError("threshold research contains invalid trade dates")
    dates = sorted(base.trade_date.unique())
    if len(dates) < max(40, horizon * 3):
        raise ValueError("insufficient dates for chronological threshold research")
    split_pos = int(len(dates) * train_fraction)
    if split_pos <= horizon:
        raise ValueError("training window is too short for the forward horizon")
    split_date = pd.Timestamp(dates[split_pos])
    train_label_end = pd.Timestamp(dates[split_pos - horizon - 1])
    rows = []
    for strength_min, turnover_min in candidates:
        classified = classify_mainlines(
            base, ignition_strength_min=float(strength_min),
            ignition_turnover_min=float(turnover_min),
        )
        classified = classified.sort_values(
            ["industry_code", "trade_date"]
        ).reset_index(drop=True)
        classified["excess_forward_return"] = (
            classified.groupby("industry_code").industry_close.shift(-horizon)
            .div(classified.industry_close).sub(1)
            - (
                classified.groupby("industry_code").market_close.shift(-horizon)
                .div(classified.market_close).sub(1)
            )
        )
        events = classified[
            classified.mainline_phase.eq("IGNITION") & classified.phase_transition
        ].copy()
        train = events[
            (events.trade_date <= train_label_end)
            & events.excess_forward_return.notna()
        ]
        test = events[
            (events.trade_date >= split_date)
            & events.excess_forward_return.notna()
        ]
        result = {
            "ignition_strength_min": float(strength_min),
            "ignition_turnover_min": float(turnover_min),
            "split_date": split_date.date().isoformat(),
            "train_label_end": train_label_end.date().isoformat(),
            "train_events": int(len(train)),
            "train_mean_excess": (
                float(train.excess_forward_return.mean()) if len(train) else None
            ),
            "train_hit_rate": (
                float(train.excess_forward_return.gt(0).mean()) if len(train) else None
            ),
            "test_events": int(len(test)),
            "test_mean_excess": (
                float(test.excess_forward_return.mean()) if len(test) else None
            ),
            "test_hit_rate": (
                float(test.excess_forward_return.gt(0).mean()) if len(test) else None
            ),
        }
        rows.append(result)
    results = pd.DataFrame(rows)
    eligible = results[results.train_events.ge(5)].copy()
    if eligible.empty:
        selected = None
    else:
        selected_row = eligible.sort_values(
            ["train_mean_excess", "train_hit_rate", "train_events"],
            ascending=[False, False, False],
        ).iloc[0]
        selected = {
            "ignition_strength_min": float(selected_row.ignition_strength_min),
            "ignition_turnover_min": float(selected_row.ignition_turnover_min),
        }
    summary = {
        "horizon_sessions": horizon,
        "train_fraction": train_fraction,
        "split_date": split_date.date().isoformat(),
        "train_label_end": train_label_end.date().isoformat(),
        "selected_on_training_only": selected,
        "selection_rule": "highest training mean excess return among configurations with at least five training events",
        "test_set_used_for_selection": False,
        "historical_pit_eligible": False,
    }
    return results, summary


def evaluate_frozen_chronological_holdout(
    frame: pd.DataFrame, *, horizons=(5, 10, 20), train_fraction: float = 0.60,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate frozen phase labels on a later chronological holdout.

    The split is based on all observed dates, not event dates. Training outcome
    labels are purged by the maximum forward horizon before the holdout begins.
    No parameters are selected or changed by this function.
    """
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError("missing validation fields: " + ",".join(sorted(missing)))
    if not horizons or any(not isinstance(h, int) or h < 1 for h in horizons):
        raise ValueError("horizons must contain positive integer trading-session counts")
    if not 0.5 <= train_fraction <= 0.8:
        raise ValueError("train_fraction must be within [0.5,0.8]")
    dates = sorted(pd.to_datetime(frame["trade_date"], errors="coerce").dropna().unique())
    if len(dates) < max(40, max(horizons) * 3):
        raise ValueError("insufficient dates for chronological holdout")
    split_pos = int(len(dates) * train_fraction)
    split_date = pd.Timestamp(dates[split_pos])
    train_label_end = pd.Timestamp(dates[split_pos - max(horizons) - 1])
    events, _ = evaluate_mainline_transitions(frame, horizons=horizons)
    events["trade_date"] = pd.to_datetime(events.trade_date)
    train = events[
        (events.trade_date <= train_label_end)
        & events[f"excess_forward_return_{max(horizons)}d"].notna()
    ].copy()
    test = events[
        (events.trade_date >= split_date)
        & events[f"excess_forward_return_{max(horizons)}d"].notna()
    ].copy()

    def metrics(rows: pd.DataFrame) -> dict:
        result = {}
        for phase in ("IGNITION", "PEAK", "DECAY"):
            subset = rows[rows.mainline_phase.eq(phase)]
            phase_result = {"events": int(len(subset))}
            for horizon in horizons:
                values = subset[f"excess_forward_return_{horizon}d"].dropna()
                favorable = values.gt(0) if phase == "IGNITION" else values.lt(0)
                phase_result[f"evaluated_{horizon}d"] = int(len(values))
                phase_result[f"mean_excess_{horizon}d"] = (
                    float(values.mean()) if len(values) else None
                )
                phase_result[f"directional_hit_rate_{horizon}d"] = (
                    float(favorable.mean()) if len(values) else None
                )
            result[phase] = phase_result
        return result

    summary = {
        "split_date": split_date.date().isoformat(),
        "train_label_end": train_label_end.date().isoformat(),
        "train": metrics(train),
        "holdout": metrics(test),
        "classifier_frozen": True,
        "threshold_search_used": False,
        "holdout_used_for_selection": False,
        "historical_pit_eligible": False,
    }
    train["validation_partition"] = "TRAIN"
    test["validation_partition"] = "HOLDOUT"
    return pd.concat([train, test], ignore_index=True), summary
