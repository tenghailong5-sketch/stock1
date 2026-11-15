"""Runtime industry-mainline scan invoked by each FHM-100 model run.

This is a bounded, forward-only observation pass. It does not start a scheduler,
perform a historical replay, or promote diagnostics into formal stock scores.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import akshare as ak
import numpy as np
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
from src.model.catalyst_mapping import map_candidate_sectors
from src.model.event_price_in import evaluate_event_price_in
from src.providers.eastmoney_market_emotion import fetch_market_emotion_pool_history
from src.providers.gov_policy import discover_gov_policy_candidates, append_policy_candidates
from src.providers.global_news import (
    discover_global_news_candidates, append_global_news_candidates,
)
from src.providers.sw_api_industry import (
    fetch_sw_api_first_level_index_list,
    fetch_sw_api_industry_index_history,
)
from src.providers.sw_industry import fetch_sw_industry_index_history as fetch_sw_industry_index_history_native


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
    frame["market_return_1d"] = frame.close.pct_change()
    frame["market_return_3d"] = frame.close.div(frame.close.shift(3)).sub(1)
    frame["market_return_5d"] = frame.close.div(frame.close.shift(5)).sub(1)
    return frame[["trade_date", "market_return_1d", "market_return_3d", "market_return_5d", "close"]].rename(
        columns={"close": "market_close"}
    )


def _read_cached_sw_universe(archive_root: Path, decision_time: pd.Timestamp) -> pd.DataFrame:
    """Read the latest verified SW first-level universe captured before cutoff."""
    directory = Path(archive_root) / "sw_industry_universe"
    if not directory.exists():
        return pd.DataFrame(columns=["industry_code", "industry_name"])
    for path in sorted(directory.glob("*.jsonl"), reverse=True):
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            frame = pd.DataFrame(rows)
            if not {"industry_code", "industry_name", "observed_at"}.issubset(frame.columns):
                continue
            frame["observed_at"] = pd.to_datetime(frame["observed_at"], errors="coerce", utc=True)
            frame = frame.loc[frame.observed_at.le(decision_time.tz_convert("UTC"))].dropna(subset=["observed_at"])
            frame["industry_code"] = frame.industry_code.astype(str).str.strip()
            frame["industry_name"] = frame.industry_name.astype(str).str.strip()
            frame = frame.loc[frame.industry_code.str.fullmatch(r"\d{6}", na=False) & frame.industry_name.ne("")]
            frame = frame.drop_duplicates("industry_code")
            if len(frame) >= 20:
                return frame[["industry_code", "industry_name"]].sort_values("industry_code").reset_index(drop=True)
        except Exception:
            continue
    return pd.DataFrame(columns=["industry_code", "industry_name"])


def _read_cached_sw_index_history(path: Path, decision_time: pd.Timestamp) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_json(path, lines=True)
        required = {"trade_date", "close", "amount", "available_time"}
        if frame.empty or not required.issubset(frame.columns):
            return pd.DataFrame()
        frame["trade_date"] = pd.to_datetime(frame.trade_date, errors="coerce").dt.normalize()
        frame["available_time"] = pd.to_datetime(frame.available_time, errors="coerce", utc=True)
        frame = frame.loc[frame.available_time.le(decision_time.tz_convert("UTC"))].dropna(subset=["trade_date", "close", "amount"])
        frame = frame.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
        if len(frame) < 25:
            return pd.DataFrame()
        latest = frame.trade_date.max()
        cutoff_day = decision_time.tz_convert("Asia/Shanghai").normalize().tz_localize(None)
        if (cutoff_day - latest).days > 7:
            return pd.DataFrame()
        return frame
    except Exception:
        return pd.DataFrame()


def _assemble_industry_panel(*, decision_time: pd.Timestamp, archive_root: Path) -> tuple[pd.DataFrame, dict]:
    groups = _read_cached_sw_universe(archive_root, decision_time)
    universe_source = "archived_SW_universe"
    if groups.empty:
        try:
            groups = fetch_sw_api_first_level_index_list()
            universe_source = "official_SW_index_api"
        except Exception as exc:
            raise RuntimeError(
                "SW industry universe unavailable from live API and archive: "
                + f"{type(exc).__name__}: {str(exc)[:180]}"
            ) from exc
    expected = int(len(groups))
    if expected < 20:
        raise RuntimeError(f"SW first-level industry universe incomplete: {expected}")
    end_date = decision_time.tz_convert("Asia/Shanghai").strftime("%Y%m%d")
    start_date = (decision_time.tz_convert("Asia/Shanghai") - pd.Timedelta(days=150)).strftime("%Y%m%d")
    cache_dir = Path(archive_root) / "industry_index_history"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_one(row):
        code = str(row.industry_code)
        cache_path = cache_dir / f"{code}.jsonl"
        try:
            # Prefer AkShare's separate market-data route; use the SW publisher
            # endpoint only as a fallback, not as the sole runtime dependency.
            try:
                history = fetch_sw_industry_index_history_native(code)
                history = history.loc[pd.to_datetime(history.trade_date, errors="coerce").between(
                    pd.Timestamp(start_date), pd.Timestamp(end_date)
                )].copy()
                if "return_3" not in history:
                    close = pd.to_numeric(history.close, errors="coerce")
                    history["return_3"] = close.div(close.shift(3)).sub(1)
                if "amount" not in history or pd.to_numeric(history.amount, errors="coerce").notna().sum() == 0:
                    raise RuntimeError("AkShare SW index response has no usable turnover")
            except Exception as alternate_exc:
                try:
                    history = fetch_sw_api_industry_index_history(
                        code, start_date=start_date, end_date=end_date
                    )
                except Exception as official_exc:
                    history = _read_cached_sw_index_history(cache_path, decision_time)
                    if history.empty:
                        raise RuntimeError(
                            f"AkShare={type(alternate_exc).__name__}:{str(alternate_exc)[:80]}; "
                            f"SW_API={type(official_exc).__name__}:{str(official_exc)[:80]}; cache unavailable"
                        ) from official_exc
            history["trade_date"] = pd.to_datetime(history.trade_date, errors="coerce").dt.normalize()
            history = history.loc[history.trade_date.between(pd.Timestamp(start_date), pd.Timestamp(end_date))].copy()
            history["close"] = pd.to_numeric(history.close, errors="coerce")
            history["amount"] = pd.to_numeric(history.get("amount"), errors="coerce")
            if "return_3" not in history:
                history["return_3"] = history.close.div(history.close.shift(3)).sub(1)
            if "return_5" not in history:
                history["return_5"] = history.close.div(history.close.shift(5)).sub(1)
            history = history.dropna(subset=["trade_date", "close", "amount"])
            if history.empty:
                raise RuntimeError("no valid index observations with turnover")
            cache = history.copy()
            if "available_time" not in cache:
                cache["available_time"] = decision_time.tz_convert("UTC")
            cache["available_time"] = pd.to_datetime(cache.available_time, errors="coerce", utc=True)
            old = _read_cached_sw_index_history(cache_path, decision_time)
            if not old.empty:
                cache = pd.concat([old, cache], ignore_index=True)
            cache = cache.dropna(subset=["trade_date", "available_time"]).sort_values(["trade_date", "available_time"])
            cache = cache.drop_duplicates("trade_date", keep="last")
            cache.assign(
                trade_date=cache.trade_date.dt.strftime("%Y-%m-%d"),
                available_time=cache.available_time.astype(str),
            ).to_json(cache_path, orient="records", lines=True, force_ascii=False)
            history["industry_name"] = str(row.industry_name)
            return history, None
        except Exception as exc:
            return None, f"{code}:{type(exc).__name__}:{str(exc)[:160]}"

    workers = max(1, min(int(os.getenv("FHM_MAINLINE_HISTORY_WORKERS", "4")), 8))
    results, frames, failures = [], [], []
    for offset in range(0, len(groups), workers):
        batch = groups.iloc[offset:offset + workers]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            batch_results = list(executor.map(fetch_one, batch.itertuples(index=False)))
        results.extend(batch_results)
        frames.extend(frame for frame, error in batch_results if frame is not None)
        failures.extend(error for frame, error in batch_results if error is not None)
        if offset == 0 and not frames and batch_results and all(
            any(kind in error for kind in ("ReadTimeout", "ConnectTimeout", "ConnectionError", "SSLError"))
            for _, error in batch_results
        ):
            raise RuntimeError("SW industry history unavailable from alternate/live/cache providers: " + " | ".join(failures[:8]))
    minimum_industries = max(1, int(expected * 0.90 + 0.999999))
    if not frames:
        raise RuntimeError(f"SW industry history unavailable for all {expected} industries: " + " | ".join(failures[:8]))
    panel = pd.concat(frames, ignore_index=True)
    counts = panel.groupby("trade_date").industry_code.nunique()
    complete_dates = counts[counts >= minimum_industries].index
    panel = panel.loc[panel.trade_date.isin(complete_dates)].copy()
    if panel.empty:
        raise RuntimeError(f"SW industry coverage below {minimum_industries}/{expected}: " + " | ".join(failures[:8]))
    market = _read_index_archive(Path(archive_root) / "index_daily" / "000001.jsonl", decision_time=decision_time)
    panel["trade_date"] = pd.to_datetime(panel.trade_date, errors="coerce").dt.normalize()
    panel["industry_return_1d"] = pd.to_numeric(panel.get("daily_return"), errors="coerce") if "daily_return" in panel else panel.groupby("industry_code").close.pct_change()
    panel["industry_return_3d"] = pd.to_numeric(panel.return_3, errors="coerce")
    panel["industry_return_5d"] = pd.to_numeric(panel.return_5, errors="coerce")
    panel["turnover"] = pd.to_numeric(panel.amount, errors="coerce")
    daily_turnover = panel.groupby("trade_date").turnover.sum().rename("market_turnover")
    panel = panel.merge(daily_turnover, on="trade_date", validate="many_to_one")
    panel = panel.merge(market, on="trade_date", how="inner", validate="many_to_one")
    panel["expected_industry_count"] = expected
    required = ["industry_return_3d", "industry_return_5d", "market_return_3d", "market_return_5d", "turnover", "market_turnover"]
    panel = panel.dropna(subset=required)
    if panel.empty:
        raise RuntimeError("no complete aligned SW industry/market observations")
    audit = {
        "industry_count": expected, "minimum_industries_per_date": minimum_industries,
        "minimum_coverage": 0.90, "available_industries": int(panel.industry_code.nunique()),
        "failed_industry_count": int(len(failures)), "failed_industries": failures,
        "complete_trade_dates": int(panel.trade_date.nunique()),
        "latest_trade_date": panel.trade_date.max().strftime("%Y-%m-%d"),
        "source": "archived_SW_universe_plus_AkShare_first_official_SW_API_fallback_and_local_cache",
        "universe_source": universe_source, "cache_directory": str(cache_dir),
        "historical_pit_eligible": False,
    }
    return panel, audit


def _assemble_emotion(*, decision_time: pd.Timestamp, archive_root: Path) -> tuple[pd.DataFrame, dict]:
    result = assemble_market_emotion_history(
        archive_root, decision_time=decision_time.isoformat(), min_sessions=20
    )
    if result.get("status") != "READY":
        raise RuntimeError(
            "market emotion archive not ready: "
            + str(result.get("status")) + "/" + str(result.get("valid_sessions"))
        )
    history = result["history"].copy()
    # Align emotion rows to completed SSE index sessions before pool lookup.
    calendar = _read_index_archive(
        Path(archive_root) / "index_daily" / "000001.jsonl",
        decision_time=decision_time,
    )
    if calendar.empty:
        raise RuntimeError("completed SSE exchange-session calendar unavailable")
    calendar_dates = set(pd.to_datetime(calendar["trade_date"], errors="coerce").dt.normalize().dropna())
    history["trade_date"] = pd.to_datetime(history["trade_date"], errors="coerce").dt.normalize()
    history = history.loc[history["trade_date"].isin(calendar_dates)].copy()
    if history.empty:
        raise RuntimeError("no market-emotion rows align to completed SSE sessions")
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


def scan_cached_market_emotion(
    *, decision_time, archive_root: str | Path = Path("data/forward_archive"),
    min_sessions: int = 4,
) -> tuple[pd.DataFrame, dict]:
    """Read only already archived breadth/sentiment rows; never fetch history.

    This lightweight path keeps market-cycle context available in current-quote
    mode without enabling the full industry-history or policy discovery scan.
    """
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    try:
        result = assemble_market_emotion_history(
            Path(archive_root), decision_time=cutoff.isoformat(),
            min_sessions=min_sessions,
        )
        history = result.get("history", pd.DataFrame())
        if not isinstance(history, pd.DataFrame) or history.empty:
            return pd.DataFrame(), {
                "status": "UNKNOWN_NO_CACHED_EMOTION",
                "valid_sessions": int(result.get("valid_sessions", 0)),
                "required_sessions": min_sessions,
                "market_cycle": "UNKNOWN",
                "historical_pit_eligible": False,
                "trading_authorization": False,
            }
        # Align sentiment snapshots to completed exchange sessions. Some
        # quote providers stamp a weekend retrieval with the calendar date;
        # that is not a new trading session and must not advance the market cycle.
        calendar = _read_index_archive(
            Path(archive_root) / "index_daily" / "000001.jsonl",
            decision_time=cutoff,
        )
        if calendar.empty:
            return pd.DataFrame(), {
                "status": "UNKNOWN_NO_COMPLETED_MARKET_SESSION",
                "valid_sessions": 0,
                "required_sessions": min_sessions,
                "market_cycle": "UNKNOWN",
                "historical_pit_eligible": False,
                "trading_authorization": False,
                "cache_only": True,
            }
        calendar_dates = set(pd.to_datetime(calendar["trade_date"]).dt.normalize())
        history = history.copy()
        history["trade_date"] = pd.to_datetime(
            history["trade_date"], errors="coerce"
        ).dt.normalize()
        history = history.loc[history["trade_date"].isin(calendar_dates)].copy()
        if history["trade_date"].nunique() < min_sessions:
            return pd.DataFrame(), {
                "status": "UNKNOWN_INSUFFICIENT_COMPLETED_SESSIONS",
                "valid_sessions": int(history["trade_date"].nunique()),
                "required_sessions": min_sessions,
                "market_calendar_latest_trade_date": calendar["trade_date"].max().date().isoformat(),
                "market_cycle": "UNKNOWN",
                "historical_pit_eligible": False,
                "trading_authorization": False,
                "cache_only": True,
            }
        cycles = classify_market_cycle(history)
        latest = cycles.sort_values("trade_date").tail(1).copy()
        state = str(latest.iloc[0].get("market_cycle", "UNKNOWN"))
        trade_date = pd.Timestamp(latest.iloc[0]["trade_date"]).date().isoformat()
        ready = state != "UNKNOWN"
        audit = {
            "status": "READY_CACHED_RESEARCH_ONLY" if ready else "UNKNOWN_CACHED_HISTORY_WARMUP",
            "market_cycle": state,
            "cycle_reason": str(latest.iloc[0].get("cycle_reason", "unknown")),
            "latest_trade_date": trade_date,
            "market_calendar_latest_trade_date": calendar["trade_date"].max().date().isoformat(),
            "valid_sessions": int(len(cycles)),
            "required_sessions": min_sessions,
            "market_cycle_data_basis": str(latest.iloc[0].get("market_cycle_data_basis", "unknown")),
            "historical_pit_eligible": False,
            "trading_authorization": False,
            "cache_only": True,
        }
        latest["historical_pit_eligible"] = False
        latest["trading_authorization"] = False
        return latest, audit
    except Exception as exc:
        return pd.DataFrame(), {
            "status": "UNKNOWN_CACHED_EMOTION_ERROR",
            "reason": f"{type(exc).__name__}: {str(exc)[:220]}",
            "market_cycle": "UNKNOWN",
            "historical_pit_eligible": False,
            "trading_authorization": False,
            "cache_only": True,
        }


def _read_discovered_candidates(path: str | Path) -> list[dict]:
    """Read the append-only candidate archive so first_seen_at survives later scans."""
    target = Path(path)
    if not target.exists():
        return []
    rows = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict) and item.get("candidate_id"):
                rows.append(item)
    return rows


def scan_runtime_mainlines(
    *, decision_time, archive_root: str | Path = Path("data/forward_archive"),
    policy_path: str | Path = Path("data/policy_catalysts/official_policy_events.jsonl"),
    policy_candidate_path: str | Path = Path("data/policy_catalysts/discovered_candidates.jsonl"),
    global_news_candidate_path: str | Path = Path("data/global_news/discovered_candidates.jsonl"),
) -> tuple[pd.DataFrame, dict]:
    """Run the mainline/emotion/policy evidence chain as part of model execution."""
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    root = Path(archive_root)
    discovery_audit = {"status": "UNAVAILABLE", "reason": "not attempted"}
    policy_candidates = []
    try:
        policy_candidates, discovery_audit = discover_gov_policy_candidates()
        discovery_audit["new_candidates_archived"] = append_policy_candidates(
            policy_candidate_path, policy_candidates
        )
    except Exception as exc:
        discovery_audit = {
            "status": "UNAVAILABLE",
            "reason": f"{type(exc).__name__}: {str(exc)[:240]}",
            "candidate_only": True,
            "formal_stock_score_eligible": False,
        }
    global_news_audit = {"status": "UNAVAILABLE", "reason": "not attempted"}
    news_candidates = []
    try:
        news_candidates, global_news_audit = discover_global_news_candidates()
        global_news_audit["new_candidates_archived"] = append_global_news_candidates(
            global_news_candidate_path, news_candidates
        )
    except Exception as news_exc:
        global_news_audit = {
            "status": "UNAVAILABLE",
            "reason": f"{type(news_exc).__name__}: {str(news_exc)[:240]}",
            "candidate_only": True,
            "formal_stock_score_eligible": False,
        }
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
        # Industry strength/flow observations must survive an unavailable
        # emotion archive. Keep emotion explicitly UNKNOWN so the ignition
        # evidence gate remains closed, while PEAK/DECAY and raw mainline
        # diagnostics can still be reported from valid industry observations.
        try:
            emotion, emotion_audit = _assemble_emotion(
                decision_time=cutoff, archive_root=root
            )
            emotion_audit["status"] = "READY"
        except Exception as emotion_exc:
            emotion = pd.DataFrame(columns=["trade_date", "market_cycle"])
            emotion_audit = {
                "status": "UNAVAILABLE_UNKNOWN",
                "reason": f"{type(emotion_exc).__name__}: {str(emotion_exc)[:240]}",
                "sessions": 0,
                "historical_pit_eligible": False,
            }
        # Policy evidence is an optional overlay, not a prerequisite for
        # industry/emotion discovery. Missing or malformed archives remain
        # UNKNOWN and never mean "no catalyst"; they must not erase the scan.
        policy_columns = [
            "event_id", "published_at", "available_at", "source_url", "source_id",
            "title", "catalyst_type", "industry_codes", "impact_direction",
            "confidence", "confidence_basis", "availability_basis",
        ]
        if policy_path.exists():
            try:
                policy = pd.read_json(policy_path, lines=True)
                visible, policy_audit = visible_policy_events(policy, decision_time=cutoff)
            except Exception as policy_exc:
                visible = pd.DataFrame(columns=policy_columns)
                policy_audit = {
                    "status": "ARCHIVE_UNAVAILABLE_UNKNOWN",
                    "reason": f"{type(policy_exc).__name__}: {str(policy_exc)[:200]}",
                    "input_rows": 0, "visible_rows": 0,
                    "historical_pit_eligible": False,
                }
        else:
            visible = pd.DataFrame(columns=policy_columns)
            policy_audit = {
                "status": "ARCHIVE_MISSING_UNKNOWN",
                "reason": "policy archive absent; no negative-catalyst inference",
                "input_rows": 0, "visible_rows": 0,
                "historical_pit_eligible": False,
            }
        evidence = attach_startup_evidence(
            phases, emotion, visible, policy_archive_complete=False
        )
        evidence = evaluate_peak_and_decay_independently(evidence)
        evidence = gate_ignition_labels(evidence)
        latest = evidence.trade_date.max()
        latest_rows = evidence.loc[evidence.trade_date.eq(latest)].copy()
        # Re-read append-only archives after discovery. This preserves the
        # original first_seen_at clock across subsequent runs so post-event
        # reaction windows can accumulate instead of resetting every scan.
        archived_policy_candidates = _read_discovered_candidates(policy_candidate_path)
        archived_news_candidates = _read_discovered_candidates(global_news_candidate_path)
        sector_candidates, sector_mapping_audit = map_candidate_sectors(
            [*archived_policy_candidates, *archived_news_candidates],
            latest_rows.get("industry_name", pd.Series(dtype=str)).dropna().astype(str).unique(),
        )
        # Measure pre-discovery run-up and forward post-discovery repricing.
        # The first-seen clock is authoritative for model availability; publisher
        # timestamps remain metadata and never move the event clock backwards.
        price_in_history = panel.copy()
        price_in_history["industry_return_1d"] = pd.to_numeric(
            price_in_history.get("daily_return"), errors="coerce"
        )
        price_in_history["market_return_1d"] = pd.to_numeric(
            price_in_history.get("market_return_1d"), errors="coerce"
        )
        price_in_history["observation_available_time"] = cutoff.tz_convert("UTC")
        price_in_rows, price_in_audit = evaluate_event_price_in(
            sector_candidates, price_in_history, decision_time=cutoff
        )
        price_in_audit["candidates"] = price_in_rows.to_dict(orient="records")
        audit = {
            "status": "READY_DIAGNOSTIC_ONLY",
            "industry": industry_audit,
            "emotion": emotion_audit,
            "policy": {
                **policy_audit,
                "discovery": discovery_audit,
                "archive_complete": False,
                "visible_events": int(len(visible)),
                "absence_means_no_catalyst": False,
            },
            "international_news": {
                **global_news_audit,
                "archive_path": str(global_news_candidate_path),
                "candidate_only": True,
                "no_unreviewed_news_changes_stock_scores": True,
            },
            "event_price_in": price_in_audit,
            "catalyst_sector_candidates": {
                **sector_mapping_audit,
                "candidates": sector_candidates,
                "policy_and_global_news_are_review_only": True,
            },
            "latest_trade_date": latest.strftime("%Y-%m-%d"),
            "latest_industry_rows": int(len(latest_rows)),
            "phase_counts": latest_rows.mainline_phase.value_counts().to_dict(),
            "gated_phase_counts": latest_rows.mainline_phase_gated.value_counts().to_dict(),
            "startup_evidence_counts": latest_rows.startup_evidence_status.value_counts().to_dict(),
            "confirmed_startups": int(latest_rows.startup_evidence_status.eq("CONFIRMED").sum()),
            "persistence_ready_count": int(latest_rows.persistence_ready.eq(True).sum()),
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
