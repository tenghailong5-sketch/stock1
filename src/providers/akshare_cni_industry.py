"""CNI historical index-member industry observations.

The CNI endpoint exposes dated constituent snapshots with an industry label.
It does not expose the original publication timestamp/revision history, and
each query is limited to one index universe. These records are therefore
historical membership observations, NOT full-market coverage and NOT
historical point-in-time-eligible inputs.
"""
from __future__ import annotations

import hashlib
import uuid

import pandas as pd


CNI_BROAD_A_SHARE_INDEX = "399317"


REQUIRED_SOURCE_COLUMNS = {"日期", "样本代码", "所属行业"}
OUTPUT_COLUMNS = [
    "snapshot_id", "index_symbol", "symbol", "membership_date",
    "industry_scheme", "industry_code", "industry_name", "observed_at",
    "available_time", "availability_basis", "source_id", "source_revision",
    "membership_scope", "historical_pit_eligible",
]


def _aware_utc(value=None) -> pd.Timestamp:
    stamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("fetched_at must be a valid timezone-aware timestamp")
    return stamp.tz_convert("UTC")


def _row_hash(index_symbol: str, row: pd.Series) -> str:
    payload = index_symbol + "|" + "|".join(
        "" if pd.isna(value) else str(value) for value in row.tolist()
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_cni_historical_industry(
    frame: pd.DataFrame, *, index_symbol: str, fetched_at=None
) -> pd.DataFrame:
    """Normalize one index's CNI historical sample response.

    membership_date is the provider's dated sample snapshot. available_time
    is the actual retrieval time. Never substitute membership_date for
    available_time or mark these rows as historical PIT eligible.
    """
    missing = REQUIRED_SOURCE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            "CNI historical sample schema missing: " + ",".join(sorted(missing))
        )
    index_code = str(index_symbol).strip()
    if not index_code or not index_code.isdigit():
        raise ValueError("index_symbol must be a non-empty numeric index code")

    captured = _aware_utc(fetched_at)
    out = frame[["日期", "样本代码", "所属行业"]].copy()
    out = out.rename(columns={
        "日期": "membership_date",
        "样本代码": "symbol",
        "所属行业": "industry_name",
    })
    out["membership_date"] = pd.to_datetime(
        out["membership_date"], errors="coerce"
    )
    out["symbol"] = out["symbol"].astype("string").str.strip().str.zfill(6)
    out["industry_name"] = out["industry_name"].astype("string").str.strip()

    if out["membership_date"].isna().any():
        raise ValueError("CNI historical sample contains invalid membership_date")
    if out["symbol"].isna().any() or out["symbol"].eq("").any():
        raise ValueError("CNI historical sample contains invalid symbol")
    if not out["symbol"].str.fullmatch(r"\d{6}").all():
        raise ValueError("CNI historical sample contains invalid six-digit symbol")
    if out["industry_name"].isna().any() or out["industry_name"].eq("").any():
        raise ValueError("CNI historical sample contains missing industry")
    if out.duplicated(["symbol", "membership_date"]).any():
        raise ValueError("CNI historical sample contains duplicate symbol/date")

    out.insert(0, "index_symbol", index_code)
    out.insert(0, "snapshot_id", uuid.uuid4().hex)
    out["industry_scheme"] = "CNI"
    # The source provides an industry label, not a numeric classification code.
    out["industry_code"] = "CNI:" + out["industry_name"].astype(str)
    out["observed_at"] = captured
    out["available_time"] = captured
    out["availability_basis"] = "retrieval_time_only"
    out["source_id"] = "akshare.index_detail_hist_cni"
    out["source_revision"] = frame[["日期", "样本代码", "所属行业"]].apply(
        lambda row: _row_hash(index_code, row), axis=1
    )
    out["membership_scope"] = "index_constituents_only"
    out["historical_pit_eligible"] = False
    return out[OUTPUT_COLUMNS].sort_values(
        ["membership_date", "symbol"]
    ).reset_index(drop=True)


def fetch_cni_historical_industry(
    index_symbols=(CNI_BROAD_A_SHARE_INDEX,), *, ak_module=None, fetched_at=None
) -> pd.DataFrame:
    """Fetch explicit CNI index universes; fail as a whole on any failed index.

    No implicit 'all A shares' claim is made. Callers must choose and record
    the index universe they intend to cover.
    """
    symbols = [str(value).strip() for value in index_symbols]
    if not symbols or any(not value for value in symbols):
        raise ValueError("index_symbols must be a non-empty list")
    if len(symbols) != len(set(symbols)):
        raise ValueError("index_symbols contains duplicates")
    if ak_module is None:
        import akshare as ak_module  # lazy import for offline tests

    captured = _aware_utc(fetched_at)
    batches = []
    for symbol in symbols:
        if not symbol.isdigit():
            raise ValueError(f"invalid CNI index symbol: {symbol}")
        try:
            raw = ak_module.index_detail_hist_cni(symbol=symbol)
            normalized = normalize_cni_historical_industry(
                raw, index_symbol=symbol, fetched_at=captured
            )
        except Exception as exc:
            raise RuntimeError(
                f"CNI historical industry fetch failed for index {symbol}"
            ) from exc
        if normalized.empty:
            raise RuntimeError(
                f"CNI historical industry response is empty for index {symbol}"
            )
        batches.append(normalized)
    return pd.concat(batches, ignore_index=True)
