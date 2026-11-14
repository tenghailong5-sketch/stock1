"""Retrospective SW first-level industry mainline replay from official index history."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

from src.model.mainline_flows import classify_mainlines
from src.model.market_cycle import classify_market_cycle
from src.model.m3_assembler import (
    assemble_m3_history, assemble_market_emotion_history,
)
from src.model.policy_catalysts import visible_policy_events
from src.providers.sw_api_industry import (
    fetch_sw_api_first_level_index_list,
    fetch_sw_api_industry_index_history,
)


def replay(start_date: str, end_date: str, *, output_dir: Path = Path("output")) -> dict:
    session = requests.Session()
    groups = fetch_sw_api_first_level_index_list(session=session)
    expected = len(groups)
    if expected < 20:
        raise RuntimeError(f"SW first-level industry list unexpectedly small: {expected}")

    frames, failures = [], []
    for row in groups.itertuples(index=False):
        code, name = row.industry_code, row.industry_name
        try:
            frame = fetch_sw_api_industry_index_history(
                code, start_date=start_date, end_date=end_date,
                session=session,
            )
            frame["industry_name"] = str(name)
            frames.append(frame)
            time.sleep(0.2)
        except Exception as exc:
            failures.append(f"{code}:{type(exc).__name__}:{str(exc)[:180]}")
    if failures:
        raise RuntimeError(
            f"industry history incomplete: {len(failures)}/{expected}; "
            + " | ".join(failures[:12])
        )
    panel = pd.concat(frames, ignore_index=True)
    counts = panel.groupby("trade_date")["industry_code"].nunique()
    complete_days = counts[counts == expected].index
    panel = panel[panel.trade_date.isin(complete_days)].copy()
    if panel.empty:
        raise RuntimeError("no dates have complete first-level industry index coverage")

    market_path = Path("data/forward_archive/index_daily/000001.jsonl")
    if not market_path.exists():
        raise RuntimeError("Shanghai Composite daily archive is missing")
    market_rows = [
        json.loads(line) for line in market_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    market = pd.DataFrame(market_rows)
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce").dt.normalize()
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    market = market.dropna(subset=["trade_date", "close"]).sort_values("trade_date")
    market["market_return_5d"] = market["close"].div(market["close"].shift(5)).sub(1)

    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize()
    panel["industry_return_5d"] = pd.to_numeric(panel["return_5"], errors="coerce")
    panel["turnover"] = pd.to_numeric(panel["amount"], errors="coerce")
    daily_total = panel.groupby("trade_date")["turnover"].sum().rename("market_turnover")
    panel = panel.merge(daily_total, on="trade_date", how="left", validate="many_to_one")
    panel = panel.merge(
        market[["trade_date", "market_return_5d"]],
        on="trade_date", how="left", validate="many_to_one",
    )
    panel = panel.dropna(subset=["industry_return_5d", "market_return_5d", "turnover", "market_turnover"])
    panel["expected_industry_count"] = expected
    panel["historical_pit_eligible"] = False
    if panel.empty:
        raise RuntimeError("no aligned industry/market history remains after date joins")

    phases = classify_mainlines(panel[[
        "trade_date", "industry_code", "industry_return_5d", "market_return_5d",
        "turnover", "market_turnover", "expected_industry_count",
        "industry_name", "available_time", "source_id", "source_revision",
    ]])
    market_decision_time = pd.Timestamp.now(tz="Asia/Shanghai")
    m3_result = assemble_m3_history(
        Path("data/forward_archive"), index_symbol="000001",
        decision_time=market_decision_time.isoformat(),
    )
    if m3_result.get("status") != "READY":
        raise RuntimeError("M3 historical market archive is not ready: " + str(m3_result.get("status")))
    emotion_result = assemble_market_emotion_history(
        Path("data/forward_archive"), decision_time=market_decision_time.isoformat(),
        min_sessions=60,
    )
    if emotion_result.get("status") != "READY":
        raise RuntimeError(
            "raw market emotion archive is not ready: "
            + str(emotion_result.get("valid_sessions"))
        )
    market_cycles = classify_market_cycle(emotion_result["history"])
    market_cycles["trade_date"] = pd.to_datetime(market_cycles.trade_date).dt.normalize()
    market_cycle_map = market_cycles.set_index("trade_date")["market_cycle"].to_dict()
    phases["trade_date"] = pd.to_datetime(phases.trade_date).dt.normalize()
    phases["market_cycle"] = phases.trade_date.map(market_cycle_map).fillna("UNKNOWN")
    phases["market_cycle_pit_eligible"] = False
    phases["historical_pit_eligible"] = False
    phases["availability_basis"] = "retrieval_time_only"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sw_mainline_phase_replay.csv"
    phases.to_csv(csv_path, index=False)
    counts = phases["mainline_phase"].value_counts().to_dict()
    ordered = phases.sort_values(["industry_code", "trade_date"]).copy()
    ordered["phase_segment"] = ordered.groupby("industry_code")["mainline_phase"].transform(
        lambda values: values.ne(values.shift()).cumsum()
    )
    segments = ordered.groupby(
        ["industry_code", "industry_name", "mainline_phase", "phase_segment"],
        dropna=False,
    ).agg(
        segment_start=("trade_date", "min"),
        segment_end=("trade_date", "max"),
        sessions=("trade_date", "size"),
        max_strength_percentile=("strength_percentile", "max"),
        max_turnover_share=("turnover_share", "max"),
        max_turnover_share_percentile=("turnover_share_percentile", "max"),
    ).reset_index()
    segments = segments[segments.mainline_phase.isin(["IGNITION", "PEAK", "DECAY"])]
    segments["historical_pit_eligible"] = False
    segments_path = output_dir / "sw_mainline_phase_segments.csv"
    segments.to_csv(segments_path, index=False)
    complete_lifecycle_industries = []
    for (code, name), group in segments.sort_values("segment_start").groupby(
        ["industry_code", "industry_name"]
    ):
        phases_for_industry = group.mainline_phase.tolist()
        if any(
            phases_for_industry[i:i + 3] == ["IGNITION", "PEAK", "DECAY"]
            for i in range(len(phases_for_industry) - 2)
        ):
            complete_lifecycle_industries.append({
                "industry_code": str(code), "industry_name": str(name)
            })

    policy_path = Path("data/policy_catalysts/official_policy_events.jsonl")
    if not policy_path.exists():
        raise RuntimeError("official policy catalyst seed archive is missing")
    policy_events = pd.read_json(policy_path, lines=True)
    visible_events, policy_audit = visible_policy_events(
        policy_events, decision_time=f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}T23:59:59+08:00"
    )
    trading_days = sorted(pd.to_datetime(phases.trade_date).dt.normalize().unique())
    day_positions = {pd.Timestamp(day): i for i, day in enumerate(trading_days)}
    first_replay_day = pd.Timestamp(trading_days[0])
    industry_names = (
        phases[["industry_code", "industry_name"]]
        .drop_duplicates("industry_code").set_index("industry_code")["industry_name"].to_dict()
    )
    ignition_transitions = phases[
        phases.mainline_phase.eq("IGNITION") & phases.phase_transition
    ].copy()
    alignment_rows = []
    for event in visible_events.itertuples(index=False):
        available_day = pd.Timestamp(event.available_at).tz_convert("Asia/Shanghai").normalize().tz_localize(None)
        outside_window = available_day < first_replay_day
        next_sessions = [day for day in trading_days if pd.Timestamp(day) > available_day]
        first_possible_day = pd.Timestamp(next_sessions[0]) if next_sessions else pd.NaT
        for code in event.industry_codes:
            first_ignition = pd.NaT
            lag_sessions = None
            status = "OUTSIDE_REPLAY_WINDOW" if outside_window else "NO_POST_EVENT_IGNITION"
            if not outside_window and pd.notna(first_possible_day):
                candidates = ignition_transitions[
                    ignition_transitions.industry_code.eq(str(code))
                    & (pd.to_datetime(ignition_transitions.trade_date).dt.normalize() > available_day)
                ].sort_values("trade_date")
                if not candidates.empty:
                    first_ignition = pd.Timestamp(candidates.iloc[0].trade_date).normalize()
                    lag_sessions = day_positions[first_ignition] - day_positions[first_possible_day] + 1
                    status = (
                        "MATCHED_WITHIN_10_SESSIONS"
                        if lag_sessions <= 10 else "IGNITION_AFTER_10_SESSIONS"
                    )
            alignment_rows.append({
                "event_id": event.event_id,
                "title": event.title,
                "available_at": event.available_at,
                "availability_basis": event.availability_basis,
                "source_url": event.source_url,
                "industry_code": str(code),
                "industry_name": industry_names.get(str(code), "UNMAPPED"),
                "first_possible_trade_date": first_possible_day,
                "first_ignition_date": first_ignition,
                "market_cycle_on_first_ignition": (
                    market_cycle_map.get(first_ignition, "UNKNOWN")
                    if pd.notna(first_ignition) else "UNKNOWN"
                ),
                "trade_session_lag": lag_sessions,
                "alignment_status": status,
                "historical_pit_eligible": False,
                "interpretation": "temporal_coincidence_only_not_causal_validation",
            })
    alignment = pd.DataFrame(alignment_rows)
    alignment_path = output_dir / "sw_mainline_policy_alignment.csv"
    alignment.to_csv(alignment_path, index=False)
    alignment_counts = alignment.alignment_status.value_counts().to_dict()
    summary = {
        "status": "RETROSPECTIVE_REPLAY_COMPLETE",
        "industry_count": expected,
        "complete_trade_dates": int(phases.trade_date.nunique()),
        "rows": int(len(phases)),
        "phase_counts": {str(k): int(v) for k, v in counts.items()},
        "phase_segments": int(len(segments)),
        "industries_with_ignition_peak_decay_sequence": complete_lifecycle_industries,
        "visible_policy_events": int(len(visible_events)),
        "policy_event_industry_pairs": int(len(alignment)),
        "policy_ignition_alignment_counts": {str(k): int(v) for k, v in alignment_counts.items()},
        "market_m3_status": m3_result.get("status"),
        "market_emotion_archive_status": emotion_result.get("status"),
        "market_emotion_archive_sessions": int(emotion_result.get("valid_sessions", 0)),
        "market_emotion_quality_warnings": emotion_result.get("quality_warnings", []),
        "market_cycle_sessions": int(len(market_cycles)),
        "breadth_observations": int(market_cycles.breadth.notna().sum()),
        "limit_pool_observations": int(
            market_cycles[["limit_up_count", "limit_down_count"]].notna().all(axis=1).sum()
        ),
        "board_failure_rate_observations": int(market_cycles.board_failure_rate.notna().sum()),
        "promotion_rate_observations": int(market_cycles.promotion_rate.notna().sum()),
        "market_cycle_counts": {
            str(k): int(v) for k, v in market_cycles.market_cycle.value_counts().to_dict().items()
        },
        "market_cycle_decision_time": market_decision_time.isoformat(),
        "market_cycle_historical_pit_eligible": False,
        "policy_availability_audit": policy_audit,
        "policy_alignment_interpretation": "temporal_coincidence_only_not_causal_validation",
        "phase_first_dates": {
            phase: str(phases.loc[phases.mainline_phase.eq(phase), "trade_date"].min().date())
            for phase in ("IGNITION", "PEAK", "DECAY")
            if phases.mainline_phase.eq(phase).any()
        },
        "start_date": str(phases.trade_date.min().date()),
        "end_date": str(phases.trade_date.max().date()),
        "historical_pit_eligible": False,
        "output": str(csv_path),
        "phase_segments_output": str(segments_path),
        "policy_alignment_output": str(alignment_path),
    }
    (output_dir / "sw_mainline_phase_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="20260522")
    parser.add_argument("--end-date", default="20261113")
    args = parser.parse_args()
    replay(args.start_date, args.end_date)
