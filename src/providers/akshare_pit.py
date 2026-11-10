"""AKShare adapters for forward-only, auditable data capture.

AKShare historical endpoints return the data as observed at retrieval time;
they do not expose the original publication timestamp or a revision history.
Accordingly, these adapters stamp retrieval availability and explicitly mark
historical rows as unsuitable for historical point-in-time backtests.
"""
from __future__ import annotations

import hashlib
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
    # Hash only provider payload fields: retrieval metadata must not make an
    # unchanged historical response look like a new source revision.
    out["source_revision"] = _revision(out)
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
    out["source_revision"] = _revision(out)
    out["available_time"] = captured
    out["source_id"] = "akshare.index_zh_a_hist"
    out["availability_basis"] = "retrieval_time_only"
    out["historical_pit_eligible"] = False
    return out

def fetch_current_industry_membership_snapshot(ak_module, *, clock=None) -> pd.DataFrame:
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

    rows = []
    for industry in industries[["板块名称", "板块代码"]].drop_duplicates().itertuples(index=False):
        name, code = str(industry[0]), str(industry[1])
        members = ak_module.stock_board_industry_cons_em(symbol=name)
        required_members = {"代码"}
        if members is None or required_members - set(members.columns):
            raise ValueError(f"AKShare constituents schema changed for {code}")
        captured = _utc(now())
        clean = members[["代码"]].copy()
        clean["symbol"] = clean["代码"].astype(str).str.zfill(6)
        clean["industry_code"] = code
        clean["industry_name"] = name
        clean["observed_at"] = captured
        clean["available_time"] = captured
        clean["source_id"] = "akshare.stock_board_industry_cons_em"
        clean["snapshot_scope"] = "current_membership_observation_only"
        revision = _revision(clean[["symbol", "industry_code", "industry_name"]])
        clean["source_revision"] = revision
        rows.append(clean.drop(columns=["代码"]))
    if not rows:
        return pd.DataFrame(columns=[
            "symbol", "industry_code", "industry_name", "observed_at",
            "available_time", "source_id", "snapshot_scope", "source_revision",
        ])
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["symbol", "industry_code", "observed_at"]).any():
        raise ValueError("duplicate symbol/industry membership in one snapshot")
    return result
