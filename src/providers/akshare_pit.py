"""AKShare adapters for forward-only, auditable data capture.

AKShare historical endpoints return the data as observed at retrieval time;
they do not expose the original publication timestamp or a revision history.
Accordingly, these adapters stamp retrieval availability and explicitly mark
historical rows as unsuitable for historical point-in-time backtests.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from math import isfinite
from datetime import datetime, timezone

import pandas as pd


def _utc(value=None) -> pd.Timestamp:
    ts = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError("capture timestamps must be timezone-aware")
    return ts.tz_convert("UTC")


def _revision(frame: pd.DataFrame) -> str:
    payload = frame.to_json(orient="records", date_format="iso", force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_revision(row: pd.Series) -> str:
    """Hash one provider observation so rolling-window changes do not re-version every row."""
    payload = row.to_frame().T.to_json(orient="records", date_format="iso", force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_daily_history_snapshot(
    frame: pd.DataFrame, *, symbol: str, fetched_at=None
) -> pd.DataFrame:
    """Normalize stock_zh_a_hist output without inventing historical availability.

    All rows receive the retrieval timestamp, not their trade date. This is
    deliberately conservative: the provider does not supply original
    publication/revision timestamps for each historical row.
    """
    columns = {
        "日期": "trade_date", "开盘": "open", "最高": "high", "最低": "low",
        "收盘": "close", "成交量": "volume", "成交额": "amount", "换手率": "turnover_pct",
    }
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError("AKShare daily history schema missing: " + ",".join(sorted(missing)))
    captured = _utc(fetched_at)
    out = frame[list(columns)].rename(columns=columns).copy()
    out.insert(0, "symbol", str(symbol).zfill(6))
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "amount", "turnover_pct"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out["trade_date"].isna().any() or out[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("AKShare daily history contains invalid date or OHLC values")
    if out.duplicated(["symbol", "trade_date"]).any():
        raise ValueError("AKShare daily history contains duplicate symbol/trade_date")
    # Hash each provider row: a rolling request window must not re-version
    # every unchanged historical observation on every daily retrieval.
    out["source_revision"] = out.apply(_row_revision, axis=1)
    out["available_time"] = captured
    out["source_id"] = "akshare.stock_zh_a_hist"
    out["availability_basis"] = "retrieval_time_only"
    out["historical_pit_eligible"] = False
    return out



def normalize_index_history_snapshot(
    frame: pd.DataFrame, *, symbol: str, fetched_at=None
) -> pd.DataFrame:
    """Normalize index_zh_a_hist output as retrieval-time-only observations."""
    columns = {
        "日期": "trade_date", "开盘": "open", "最高": "high", "最低": "low",
        "收盘": "close", "成交量": "volume", "成交额": "amount",
    }
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError("AKShare index history schema missing: " + ",".join(sorted(missing)))
    captured = _utc(fetched_at)
    out = frame[list(columns)].rename(columns=columns).copy()
    out.insert(0, "index_symbol", str(symbol).zfill(6))
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "amount"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out["trade_date"].isna().any() or out.duplicated(["index_symbol", "trade_date"]).any():
        raise ValueError("AKShare index history contains invalid or duplicate dates")
    numeric = out[["open", "high", "low", "close", "volume", "amount"]]
    if numeric.isna().any().any() or not numeric.apply(lambda c: c.map(pd.notna)).all().all():
        raise ValueError("AKShare index history contains non-numeric values")
    if ((out[["open", "high", "low", "close"]] <= 0).any().any()
            or (out[["volume", "amount"]] < 0).any().any()
            or (out["high"] < out[["open", "low", "close"]].max(axis=1)).any()
            or (out["low"] > out[["open", "high", "close"]].min(axis=1)).any()):
        raise ValueError("AKShare index history contains invalid OHLC/volume/amount values")
    # Per-row hashes prevent a rolling 180-day request window from causing
    # unchanged historical index bars to be appended again every day.
    out["source_revision"] = out.apply(_row_revision, axis=1)
    out["available_time"] = captured
    out["source_id"] = "akshare.index_zh_a_hist"
    out["availability_basis"] = "retrieval_time_only"
    out["historical_pit_eligible"] = False
    return out

def fetch_current_industry_membership_snapshot(
    ak_module, *, clock=None, pause_seconds: float = 0.15, sleep_fn=time.sleep
) -> pd.DataFrame:
    """Capture current Eastmoney industry constituents as a dated snapshot only.

    No valid_from/valid_to is fabricated. Repeated daily snapshots can support
    prospective observation from the collection start, but cannot reconstruct
    membership before the first archived snapshot.
    """
    now = clock or (lambda: datetime.now(timezone.utc))
    industries = ak_module.stock_board_industry_name_em()
    required_industry = {"板块名称", "板块代码"}
    if industries is None or industries.empty or required_industry - set(industries.columns):
        raise ValueError("AKShare industry list schema changed or is empty")

    # One immutable identifier binds every industry response fetched by this call.
    # Per-industry observed_at/available_time remain the actual retrieval times.
    snapshot_id = str(uuid.uuid4())
    rows = []
    for industry in industries[["板块名称", "板块代码"]].drop_duplicates().itertuples(index=False):
        name, code = str(industry[0]), str(industry[1])
        members = ak_module.stock_board_industry_cons_em(symbol=name)
        if pause_seconds > 0:
            sleep_fn(pause_seconds)
        required_members = {"代码"}
        if members is None or required_members - set(members.columns):
            raise ValueError(f"AKShare constituents schema changed for {code}")
        captured = _utc(now())
        clean = members[["代码"]].copy()
        clean["symbol"] = clean["代码"].astype(str).str.zfill(6)
        clean["industry_code"] = code
        clean["industry_name"] = name
        clean["snapshot_id"] = snapshot_id
        clean["observed_at"] = captured
        clean["available_time"] = captured
        clean["availability_basis"] = "retrieval_time_only"
        clean["historical_pit_eligible"] = False
        clean["source_id"] = "akshare.stock_board_industry_cons_em"
        clean["snapshot_scope"] = "current_membership_observation_only"
        revision = _revision(clean[["symbol", "industry_code", "industry_name"]])
        clean["source_revision"] = revision
        rows.append(clean.drop(columns=["代码"]))
    if not rows:
        return pd.DataFrame(columns=[
            "symbol", "industry_code", "industry_name", "snapshot_id", "observed_at",
            "available_time", "source_id", "snapshot_scope", "source_revision",
        ])
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["symbol", "industry_code", "observed_at"]).any():
        raise ValueError("duplicate symbol/industry membership in one snapshot")
    return result


def normalize_market_breadth_snapshot(frame: pd.DataFrame, *, fetched_at=None) -> pd.DataFrame:
    """Summarize one current A-share quote snapshot; never infer historical limit events."""
    required = {"代码", "最新价", "涨跌幅"}
    if frame is None:
        raise ValueError("AKShare spot snapshot is empty or missing: 代码,最新价,涨跌幅")
    missing = required - set(frame.columns)
    if frame.empty or missing:
        raise ValueError(
            "AKShare spot snapshot is empty or missing: " + ",".join(sorted(missing))
        )
    captured = _utc(fetched_at)
    selected = ["代码", "最新价", "涨跌幅"] + (["成交额"] if "成交额" in frame.columns else [])
    out = frame[selected].copy()
    out["symbol"] = out["代码"].astype(str).str.zfill(6)
    if out["symbol"].duplicated().any():
        raise ValueError("AKShare spot snapshot contains duplicate symbols")
    out["last_price"] = pd.to_numeric(out["最新价"], errors="coerce")
    out["change_pct"] = pd.to_numeric(out["涨跌幅"], errors="coerce")
    valid = (
        out["last_price"].notna()
        & out["change_pct"].notna()
        & out["last_price"].gt(0)
        & out["last_price"].map(lambda value: pd.notna(value) and isfinite(float(value)))
        & out["change_pct"].map(lambda value: pd.notna(value) and isfinite(float(value)))
    )
    valid_rows = out.loc[valid]
    advances = int(valid_rows["change_pct"].gt(0).sum())
    declines = int(valid_rows["change_pct"].lt(0).sum())
    unchanged = int(valid_rows["change_pct"].eq(0).sum())
    valid_count = int(valid.sum())
    total_count = int(len(out))
    breadth_ratio = advances / valid_count if valid_count else None
    total_amount = None
    total_amount_status = "missing_provider_field"
    if "成交额" in out.columns:
        amounts = pd.to_numeric(out["成交额"], errors="coerce")
        if amounts.notna().all() and amounts.map(lambda v: isfinite(float(v))).all() and amounts.ge(0).all():
            total_amount = float(amounts.sum())
            total_amount_status = "complete_provider_snapshot_sum"
        else:
            total_amount_status = "invalid_or_incomplete_provider_values"
    summary = {
        "snapshot_time": captured,
        "provider_row_count": total_count,
        "valid_quote_count": valid_count,
        "missing_quote_count": total_count - valid_count,
        "advances": advances,
        "declines": declines,
        "unchanged": unchanged,
        "breadth_coverage": valid_count / total_count,
        "breadth": breadth_ratio,
        "total_amount": total_amount,
        "total_amount_status": total_amount_status,
        "snapshot_scope": "current_provider_snapshot_only",
        "historical_pit_eligible": False,
    }
    summary["source_revision"] = hashlib.sha256(
        json_dumps_stable({k: v for k, v in summary.items() if k != "snapshot_time"}).encode("utf-8")
    ).hexdigest()
    summary["available_time"] = captured
    summary["source_id"] = "akshare.stock_zh_a_spot_em"
    return pd.DataFrame([summary])


def json_dumps_stable(value: dict) -> str:
    """Serialize summary values deterministically for revision hashing."""
    import json
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
