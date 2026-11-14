"""Runtime industry-mainline scan invoked by each FHM-100 model run.

This is a bounded, forward-only observation pass. It does not start a scheduler,
perform a historical replay, or promote diagnostics into formal stock scores.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import akshare as ak
import pandas as pd
import requests

from src.model.mainline_flows import classify_mainlines
from src.model.mainline_signal_gates import (
    attach_startup_evidence,
    audit_emotion_history,
    evaluate_peak_and_decay_independently,
    gate_ignition_labels,
)
from src.model.market_cycle import classify_market_cycle
from src.model.m3_assembler import assemble_market_emotion_history
from src.model.policy_catalysts import visible_policy_events
from src.providers.eastmoney_market_emotion import fetch_market_emotion_pool_history
from src.providers.sw_api_industry import (
    fetch_sw_api_first_level_index_list,
    fetch_sw_api_industry_index_history,
)


def _read_index_archive(path: Path, *, decision_time: pd.Timestamp) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError("Shanghai Composite index archive is missing")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(rows)
    required = {"trade_date", "close", "available_time"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError("market index archive missing: " + ",".join(sorted(missing)))
    frame["trade_date"] = pd.to_datetime(frame.trade_date, errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame.close, errors="coerce")
    frame["available_time"] = pd.to_datetime(frame.available_time, errors="coerce", utc=True)
    cutoff = decision_time.tz_convert("UTC")
    frame = frame.dropna(subset=["trade_date", "close", "available_time"])
    frame = frame.loc[(frame.available_time <= cutoff) & (frame.close > 0)]
    frame = frame.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    frame["market_return_3d"] = frame.close.div(frame.close.shift(3)).sub(1)
    frame["market_return_5d"] = frame.close.div(frame.close.shift(5)).sub(1)
    return frame[["trade_date", "market_return_3d", "market_return_5d", "close"]].rename(
        columns={"close": "market_close"}
    )


def _assemble_industry_panel(*, decision_time: pd.Timestamp, archive_root: Path) -> tuple[pd.DataFrame, dict]:
    session = requests.Session()
    groups = fetch_sw_api_first_level_index_list(session=session)
    expected = int(len(groups))
    if expected < 20:
        raise RuntimeError(f"SW first-level industry universe incomplete: {expected}")
    end_date = decision_time.tz_convert("Asia/Shanghai").strftime("%Y%m%d")
    start_date = (decision_time.tz_convert("Asia/Shanghai") - pd.Timedelta(days=150)).strftime("%Y%m%d")
    frames = []
    failures = []
    for row in groups.itertuples(index=False):
        try:
            history = fetch_sw_api_industry_index_history(
                str(row.industry_code), start_date=start_date, end_date=end_date,
                session=session,
            )
            history["industry_name"] = str(row.industry_name)
            frames.append(history)
        except Exception as exc:
            failures.append(f"{row.industry_code}:{type(exc).__name__}:{str(exc)[:140]}")
        time.sleep(0.15)
    if failures:
        raise RuntimeError(
            f"SW industry series incomplete {len(failures)}/{expected}: "
            + " | ".join(failures[:8])
        )
    panel = pd.concat(frames, ignore_index=True)
    counts = panel.groupby("trade_date").industry_code.nunique()
    complete_dates = counts[counts == expected].index
    panel = panel.loc[panel.trade_date.isin(complete_dates)].copy()
    market = _read_index_archive(
        archive_root / "index_daily" / "000001.jsonl",
        decision_time=decision_time,
    )
    panel["trade_date"] = pd.to_datetime(panel.trade_date, errors="coerce").dt.normalize()
    panel["industry_return_3d"] = pd.to_numeric(panel.return_3, errors="coerce")
    panel["industry_return_5d"] = pd.to_numeric(panel.return_5, errors="coerce")
    panel["turnover"] = pd.to_numeric(panel.amount, errors="coerce")
    daily_turnover = panel.groupby("trade_date").turnover.sum().rename("market_turnover")
    panel = panel.merge(daily_turnover, on="trade_date", validate="many_to_one")
    panel = panel.merge(market, on="trade_date", how="inner", validate="many_to_one")
    panel["expected_industry_count"] = expected
    required = [
        "industry_return_3d", "industry_return_5d", "market_return_3d",
        "market_return_5d", "turnover", "market_turnover",
    ]
    panel = panel.dropna(subset=required)
    if panel.empty:
        raise RuntimeError("no complete aligned SW industry/market observations")
    audit = {
        "industry_count": expected,
        "complete_trade_dates": int(panel.trade_date.nunique()),
        "latest_trade_date": panel.trade_date.max().strftime("%Y-%m-%d"),
        "source": "official_SW_index_api_plus_archived_SSE_index",
        "historical_pit_eligible": False,
        "failed_industries": [],
    }
    return panel, audit


def _assemble_emotion(*, decision_time: pd.Timestamp, archive_root: Path) -> tuple[pd.DataFrame, dict]:
    result = assemble_market_emotion_history(
        archive_root, decision_time=decision_time.isoformat(), min_sessions=60
    )
    if result.get("status") != "READY":
        raise RuntimeError(
            "market emotion archive not ready: "
            + str(result.get("status")) + "/" + str(result.get("valid_sessions"))
        )
    history = result["history"].copy()
    dates = history.trade_date.tail(15).tolist()
    pools = fetch_market_emotion_pool_history(
        dates,
        limit_up_fetcher=ak.stock_zt_pool_em,
        broken_board_fetcher=ak.stock_zt_pool_zbgc_em,
        strong_pool_fetcher=ak.stock_zt_pool_strong_em,
    )
    history["trade_date"] = pd.to_datetime(history.trade_date).dt.normalize()
    pools["trade_date"] = pd.to_datetime(pools.trade_date).dt.normalize()
    pools = pools.rename(columns={"available_time": "pool_available_time"})
    history = history.merge(pools, on="trade_date", how="left", validate="one_to_one", suffixes=("", "_pool"))
    for field in (
        "highest_observed_limit_up_streak", "strong_pool_count",
        "board_failure_rate", "promotion_rate",
    ):
        pool_field = field + "_pool"
        if pool_field in history:
            history[field] = history[pool_field].combine_first(history[field])
    if "pool_available_time" in history:
        history["available_time"] = history.apply(
            lambda row: max(row.available_time, row.pool_available_time)
            if pd.notna(row.pool_available_time) else row.available_time,
            axis=1,
        )
    cycles = classify_market_cycle(history)
    audit = audit_emotion_history(cycles)
    audit["latest_trade_date"] = cycles.trade_date.max().strftime("%Y-%m-%d")
    audit["pool_rows_refreshed"] = int(len(pools))
    audit["pool_ready_rows"] = int(pools.pool_history_status.eq("READY").sum())
    audit["historical_pit_eligible"] = False
    return cycles, audit


def scan_runtime_mainlines(
    *, decision_time, archive_root: str | Path = Path("data/forward_archive"),
    policy_path: str | Path = Path("data/policy_catalysts/official_policy_events.jsonl"),
) -> tuple[pd.DataFrame, dict]:
    """Run the mainline/emotion/policy evidence chain as part of model execution."""
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    root = Path(archive_root)
    try:
        panel, industry_audit = _assemble_industry_panel(
            decision_time=cutoff, archive_root=root
        )
        phases = classify_mainlines(panel[[
            "trade_date", "industry_code", "industry_return_3d",
            "market_return_3d", "industry_return_5d", "market_return_5d",
            "turnover", "market_turnover", "expected_industry_count",
        ]])
        names = panel[["industry_code", "industry_name"]].drop_duplicates("industry_code")
        phases = phases.merge(names, on="industry_code", how="left", validate="many_to_one")
        emotion, emotion_audit = _assemble_emotion(
            decision_time=cutoff, archive_root=root
        )
        if not policy_path.exists():
            raise RuntimeError("official policy catalyst archive is missing")
        policy = pd.read_json(policy_path, lines=True)
        visible, policy_audit = visible_policy_events(policy, decision_time=cutoff)
        # The local seed archive is explicitly incomplete. A missing event is
        # therefore UNKNOWN, never evidence that no catalyst exists.
        evidence = attach_startup_evidence(
            phases, emotion, visible, policy_archive_complete=False
        )
        evidence = evaluate_peak_and_decay_independently(evidence)
        evidence = gate_ignition_labels(evidence)
        latest = evidence.trade_date.max()
        latest_rows = evidence.loc[evidence.trade_date.eq(latest)].copy()
        audit = {
            "status": "READY_DIAGNOSTIC_ONLY",
            "industry": industry_audit,
            "emotion": emotion_audit,
            "policy": {
                **policy_audit,
                "archive_complete": False,
                "visible_events": int(len(visible)),
                "absence_means_no_catalyst": False,
            },
            "latest_trade_date": latest.strftime("%Y-%m-%d"),
            "latest_industry_rows": int(len(latest_rows)),
            "phase_counts": latest_rows.mainline_phase.value_counts().to_dict(),
            "gated_phase_counts": latest_rows.mainline_phase_gated.value_counts().to_dict(),
            "startup_evidence_counts": latest_rows.startup_evidence_status.value_counts().to_dict(),
            "confirmed_startups": int(latest_rows.startup_evidence_status.eq("CONFIRMED").sum()),
            "peak_evidence_true": int(latest_rows.peak_evidence.fillna(False).sum()),
            "decay_evidence_true": int(latest_rows.decay_evidence.fillna(False).sum()),
            "historical_pit_eligible": False,
            "formal_stock_score_eligible": False,
        }
        return latest_rows.sort_values(
            ["mainline_phase", "strength_percentile", "turnover_share"],
            ascending=[True, False, False],
        ), audit
    except Exception as exc:
        return pd.DataFrame(), {
            "status": "UNAVAILABLE",
            "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
            "historical_pit_eligible": False,
            "formal_stock_score_eligible": False,
        }
