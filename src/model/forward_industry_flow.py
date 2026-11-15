"""Forward-only industry turnover-share observations for live research."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def attach_forward_industry_flow_changes(
    panel: pd.DataFrame,
    *,
    decision_time,
    archive_root: str | Path = Path("data/forward_archive"),
) -> tuple[pd.DataFrame, dict]:
    """Persist completed-session spot shares and derive observed 3-session changes.

    Only a weekday snapshot captured after the Shanghai close is archived.
    The archive is retrieval-time research data, never PIT evidence or a trade gate.
    """
    out = panel.copy() if isinstance(panel, pd.DataFrame) else pd.DataFrame()
    audit = {
        "status": "NO_PANEL",
        "archived": False,
        "observations": 0,
        "history_sessions": 0,
        "flow_change_ready": False,
        "historical_pit_eligible": False,
        "trading_authorization": False,
    }
    if out.empty:
        return out, audit
    required = {"industry_name", "trade_date", "turnover_share"}
    missing = required - set(out.columns)
    if missing:
        audit.update(status="MISSING_FIELDS", missing_fields=sorted(missing))
        return out, audit
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    local = cutoff.tz_convert("Asia/Shanghai")
    dates = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        audit.update(status="INVALID_TRADE_DATE")
        return out, audit
    trade_date = dates.max()
    if getattr(trade_date, "tzinfo", None) is not None:
        trade_date = trade_date.tz_localize(None)
    trade_date = pd.Timestamp(trade_date).normalize()
    trade_date_key = trade_date.strftime("%Y-%m-%d")
    shares = pd.to_numeric(out["turnover_share"], errors="coerce")
    snapshot = pd.DataFrame({
        "trade_date": dates.dt.strftime("%Y-%m-%d"),
        "industry_name": out["industry_name"].astype(str).str.strip(),
        "turnover_share": shares,
    }).dropna(subset=["turnover_share"])
    # Archive only the latest completed session, never relabel older history
    # with today's retrieval timestamp.
    snapshot = snapshot.loc[
        snapshot["trade_date"].eq(trade_date_key)
    ].copy()
    snapshot = snapshot.loc[
        snapshot["industry_name"].ne("")
        & snapshot["turnover_share"].between(0, 1, inclusive="both")
    ].copy()
    path = Path(archive_root) / "industry_flow_daily.jsonl"
    eligible = (
        local.weekday() < 5
        and local.time() >= pd.Timestamp("15:05").time()
        and trade_date_key == local.strftime("%Y-%m-%d")
        and not snapshot.empty
    )
    existing = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if {"trade_date", "industry_name", "turnover_share"}.issubset(row):
                    existing.append(row)
            except (ValueError, TypeError):
                continue
    history = pd.DataFrame(existing)
    if eligible:
        snapshot["captured_at"] = cutoff.isoformat()
        snapshot["availability_basis"] = "retrieval_time_only"
        snapshot["historical_pit_eligible"] = False
        history = pd.concat([history, snapshot], ignore_index=True)
        history["trade_date"] = pd.to_datetime(history["trade_date"], errors="coerce").dt.normalize().dt.strftime("%Y-%m-%d")
        history["captured_at"] = pd.to_datetime(history["captured_at"], errors="coerce", utc=True)
        history = history.dropna(subset=["trade_date", "captured_at"])
        history = history.sort_values("captured_at").drop_duplicates(
            ["trade_date", "industry_name"], keep="last"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        records = history.assign(captured_at=history.captured_at.astype(str)).to_dict("records")
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in records),
            encoding="utf-8",
        )
    if history.empty:
        out["turnover_share_change_1d"] = pd.NA
        out["turnover_share_change_3d"] = pd.NA
        audit.update(
            status="WAITING_COMPLETED_SESSION_ARCHIVE",
            observations=int(len(snapshot)),
            trade_date=trade_date.date().isoformat(),
        )
        return out, audit
    history["trade_date"] = pd.to_datetime(history["trade_date"], errors="coerce")
    history["turnover_share"] = pd.to_numeric(history["turnover_share"], errors="coerce")
    history = history.dropna(subset=["trade_date", "turnover_share"])
    history = history.sort_values(["industry_name", "trade_date"]).drop_duplicates(
        ["industry_name", "trade_date"], keep="last"
    )
    latest_rows = []
    for name, group in history.groupby("industry_name", sort=False):
        group = group.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
        # Work from the ordered session window rather than mixing timezone-aware
        # panel dates with naive archive dates. The latest archived row must be
        # the requested session; a 3-session change needs current + 3 prior rows.
        group = group.loc[group.trade_date.dt.strftime("%Y-%m-%d").le(trade_date_key)]
        if group.empty or group.iloc[-1].trade_date.strftime("%Y-%m-%d") != trade_date_key:
            continue
        window = group.tail(4)
        row = window.iloc[-1]
        prior1 = window.iloc[-2] if len(window) >= 2 else None
        prior3 = window.iloc[0] if len(window) >= 4 else None
        latest_rows.append({
            "industry_name": name,
            "turnover_share_change_1d": (
                float(row.turnover_share - prior1.turnover_share)
                if prior1 is not None else float("nan")
            ),
            "turnover_share_change_3d": (
                float(row.turnover_share - prior3.turnover_share)
                if prior3 is not None else float("nan")
            ),
            "industry_flow_observation_count": int(len(group)),
        })
    changes = pd.DataFrame(latest_rows)
    if not changes.empty:
        out = out.drop(columns=[
            c for c in ("turnover_share_change_1d", "turnover_share_change_3d",
                        "industry_flow_observation_count") if c in out
        ]).merge(changes, on="industry_name", how="left", validate="many_to_one")
    else:
        out["turnover_share_change_1d"] = pd.NA
        out["turnover_share_change_3d"] = pd.NA
        out["industry_flow_observation_count"] = 0
    ready = bool(
        not changes.empty
        and pd.to_numeric(changes.get("turnover_share_change_3d"), errors="coerce").notna().any()
    )
    audit.update({
        "status": "READY_FORWARD_FLOW_RESEARCH" if ready else "WARMING_UP_FORWARD_FLOW",
        "archived": bool(eligible),
        "observations": int(len(snapshot)),
        "history_sessions": int(history.trade_date.nunique()),
        "flow_change_ready": ready,
        "trade_date": trade_date.date().isoformat(),
        "archive_path": str(path),
        "historical_pit_eligible": False,
        "trading_authorization": False,
    })
    return out, audit
