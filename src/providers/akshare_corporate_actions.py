"""Forward-only A-share corporate-action observation ledger.

AKShare/Sina history responses are retrieval-time observations. They may contain
old announcements discovered today, so they are never backdated to announcement
dates or admitted as historical PIT evidence. A record is usable only for
decisions at or after its first observed snapshot, and remains CAPTURED rather
than VERIFIED until a separately validated total-return reconstruction exists.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from src.collectors.forward_archive import append_jsonl_atomic


DIVIDEND_COLUMNS = {
    "公告日期": "announcement_date",
    "送股": "bonus_share_ratio",
    "送股(股)": "bonus_share_ratio",
    "转增": "capitalization_ratio",
    "转增(股)": "capitalization_ratio",
    "派息": "cash_dividend",
    "派息(税前)(元)": "cash_dividend",
    "进度": "progress",
    "除权除息日": "ex_date",
    "股权登记日": "record_date",
    "红股上市日": "share_listing_date",
}
RIGHTS_COLUMNS = {
    "公告日期": "announcement_date",
    "配股方案(每10股配股股数)": "rights_issue_ratio",
    "配股价格(元)": "rights_issue_price",
    "基准股本(股)": "base_share_capital",
    "除权日": "ex_date",
    "股权登记日": "record_date",
    "缴款起始日": "subscription_start_date",
    "缴款终止日": "subscription_end_date",
    "配股上市日": "share_listing_date",
    "募集资金合计(元)": "rights_issue_proceeds",
}
OUTPUT_COLUMNS = [
    "symbol", "event_type", "event_key", "announcement_date", "ex_date",
    "record_date", "share_listing_date", "bonus_share_ratio",
    "capitalization_ratio", "cash_dividend", "rights_issue_ratio",
    "rights_issue_price", "base_share_capital", "subscription_start_date",
    "subscription_end_date", "rights_issue_proceeds", "progress",
    "first_seen_at", "observed_at", "available_time", "source_id",
    "source_revision", "availability_basis", "historical_pit_eligible",
    "eligible_for_decisions_after_first_seen", "corporate_action_status",
]
DATE_FIELDS = {
    "announcement_date", "ex_date", "record_date", "share_listing_date",
    "subscription_start_date", "subscription_end_date",
}
NUMERIC_FIELDS = {
    "bonus_share_ratio", "capitalization_ratio", "cash_dividend",
    "rights_issue_ratio", "rights_issue_price", "base_share_capital",
    "rights_issue_proceeds",
}
SOURCE_ID = "akshare.stock_history_dividend_detail"


def _aware_utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("observed_at must be a valid timezone-aware timestamp")
    return stamp.tz_convert("UTC")


def _date_value(value):
    if value is None or pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw or raw in {"--", "-", "NaT", "nan", "None", "1900-01-01"}:
        return None
    stamp = pd.to_datetime(raw, errors="coerce")
    return None if pd.isna(stamp) else stamp.strftime("%Y-%m-%d")


def _number_value(value):
    if value is None or pd.isna(value):
        return None
    raw = str(value).strip().replace(",", "")
    if not raw or raw in {"--", "-", "NaT", "nan", "None"}:
        return None
    number = pd.to_numeric(raw, errors="coerce")
    if pd.isna(number) or not math.isfinite(float(number)):
        return None
    return float(number)


def _read_existing(path: Path) -> dict[str, str]:
    """Return first_seen_at by stable event identity; malformed archives fail closed."""
    if not path.exists():
        return {}
    first_seen = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"corrupt corporate-action archive at line {line_number}") from exc
        key = str(item.get("event_key", ""))
        seen = item.get("first_seen_at")
        if key and seen:
            previous = first_seen.get(key)
            if previous is None or pd.Timestamp(seen) < pd.Timestamp(previous):
                first_seen[key] = str(seen)
    return first_seen


def normalize_corporate_actions(
    frame: pd.DataFrame, *, symbol: str, indicator: str, observed_at,
    first_seen_by_key: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize one endpoint response while preserving first observation time."""
    if indicator not in {"分红", "配股"}:
        raise ValueError("indicator must be 分红 or 配股")
    symbol = str(symbol).strip().zfill(6)
    if not symbol.isdigit() or len(symbol) != 6:
        raise ValueError("symbol must be a six-digit A-share code")
    observed = _aware_utc(observed_at)
    mapping = DIVIDEND_COLUMNS if indicator == "分红" else RIGHTS_COLUMNS
    event_type = "dividend_bonus" if indicator == "分红" else "rights_issue"
    if frame is None:
        raise ValueError("corporate-action provider returned None")
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not set(mapping).intersection(frame.columns):
        raise ValueError(f"{indicator} response schema has no recognized columns")

    rows = []
    for _, source in frame.iterrows():
        row = {column: None for column in OUTPUT_COLUMNS}
        row.update({"symbol": symbol, "event_type": event_type})
        for source_name, target_name in mapping.items():
            if source_name not in frame.columns:
                continue
            value = source[source_name]
            row[target_name] = (
                _date_value(value) if target_name in DATE_FIELDS
                else _number_value(value) if target_name in NUMERIC_FIELDS
                else (None if pd.isna(value) else str(value).strip())
            )
        if not row["announcement_date"]:
            raise ValueError("corporate-action row missing valid announcement_date")
        identity = {
            "symbol": symbol, "event_type": event_type,
            # Announcement date is the stable event identity: ex/record dates
            # may be populated or revised in later observations.
            "announcement_date": row["announcement_date"],
        }
        row["event_key"] = hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        revision_payload = {key: row[key] for key in (
            "symbol", "event_type", "announcement_date", "ex_date", "record_date",
            "share_listing_date", "bonus_share_ratio", "capitalization_ratio",
            "cash_dividend", "rights_issue_ratio", "rights_issue_price",
            "base_share_capital", "subscription_start_date", "subscription_end_date",
            "rights_issue_proceeds", "progress",
        )}
        row["source_revision"] = hashlib.sha256(
            json.dumps(revision_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        first_seen = (first_seen_by_key or {}).get(row["event_key"])
        row["first_seen_at"] = _aware_utc(first_seen or observed).isoformat()
        row["observed_at"] = observed.isoformat()
        row["available_time"] = row["first_seen_at"]
        row["source_id"] = SOURCE_ID
        row["availability_basis"] = "first_observed_snapshot"
        row["historical_pit_eligible"] = False
        row["eligible_for_decisions_after_first_seen"] = True
        row["corporate_action_status"] = "CAPTURED"
        rows.append(row)

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if result["event_key"].duplicated().any():
        # One response containing conflicting duplicate event identities is ambiguous.
        duplicated = result[result["event_key"].duplicated(keep=False)]
        if duplicated.groupby("event_key")["source_revision"].nunique().gt(1).any():
            raise ValueError("corporate-action response contains conflicting duplicate events")
        result = result.drop_duplicates("event_key", keep="last")
    return result.reset_index(drop=True)


def archive_corporate_action_snapshot(
    symbol: str, *, ak_module, observed_at, archive_root=Path("data/forward_archive")
) -> dict:
    """Fetch dividend and rights history for one shortlisted symbol and append revisions."""
    root = Path(archive_root) / "corporate_actions"
    path = root / f"{str(symbol).zfill(6)}.jsonl"
    existing = _read_existing(path)
    provider = getattr(ak_module, "stock_history_dividend_detail", None)
    if not callable(provider):
        raise RuntimeError("AKShare stock_history_dividend_detail is unavailable")

    frames = []
    for indicator in ("分红", "配股"):
        raw = provider(symbol=str(symbol).zfill(6), indicator=indicator)
        # Stamp after each provider response returns; a slow request must not
        # inherit the batch-start time and become available before observation.
        response_observed = _aware_utc(
            observed_at() if callable(observed_at) else observed_at
        )
        normalized = normalize_corporate_actions(
            raw, symbol=symbol, indicator=indicator, observed_at=response_observed,
            first_seen_by_key=existing,
        )
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        return {"symbol": str(symbol).zfill(6), "rows_appended": 0, "status": "NO_EVENTS"}
    records = json.loads(pd.concat(frames, ignore_index=True).to_json(
        orient="records", force_ascii=False
    ))
    added = append_jsonl_atomic(
        path, records, ("event_key", "source_revision")
    )
    return {
        "symbol": str(symbol).zfill(6), "rows_appended": int(added),
        "status": "CAPTURED" if added else "UNCHANGED",
    }


def archive_shortlist_corporate_actions(
    candidates: pd.DataFrame, *, ak_module, observed_at,
    archive_root=Path("data/forward_archive"), max_symbols: int = 20,
) -> dict:
    """Archive events only for a bounded, currently observed candidate shortlist."""
    if max_symbols < 1:
        raise ValueError("max_symbols must be positive")
    if candidates is None or candidates.empty:
        return {"status": "NO_CANDIDATES", "symbols_checked": 0, "rows_appended": 0, "failures": {}}
    symbol_col = "symbol" if "symbol" in candidates else "code" if "code" in candidates else None
    if symbol_col is None:
        raise ValueError("candidate frame requires symbol or code")
    frame = candidates.copy()
    frame[symbol_col] = frame[symbol_col].astype(str).str.strip().str[-6:].str.zfill(6)
    frame = frame[frame[symbol_col].str.fullmatch(r"\d{6}", na=False)]
    sort_column = next((name for name in ("快速初筛分", "amount", "成交额") if name in frame), None)
    if sort_column:
        frame[sort_column] = pd.to_numeric(frame[sort_column], errors="coerce")
        frame = frame.sort_values(sort_column, ascending=False, na_position="last")
    symbols = frame[symbol_col].drop_duplicates().head(max_symbols).tolist()
    results, failures = [], {}
    for code in symbols:
        try:
            results.append(archive_corporate_action_snapshot(
                code, ak_module=ak_module, observed_at=observed_at,
                archive_root=archive_root,
            ))
        except Exception as exc:
            failures[code] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return {
        "status": "PARTIAL" if failures else "READY",
        "symbols_checked": len(symbols),
        "rows_appended": sum(item["rows_appended"] for item in results),
        "captured_symbols": sum(item["status"] == "CAPTURED" for item in results),
        "unchanged_symbols": sum(item["status"] == "UNCHANGED" for item in results),
        "no_event_symbols": sum(item["status"] == "NO_EVENTS" for item in results),
        "failures": failures,
        "historical_pit_eligible": False,
        "decision_eligible_only_after_first_seen": True,
        "corporate_action_status": "CAPTURED_NOT_VERIFIED",
    }
