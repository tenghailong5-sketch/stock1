#!/usr/bin/env python3
"""Prepare a Baostock-confirmed, stock-only catalogue for historical recovery.

The raw Sina discovery catalogue is intentionally broader than the formal
identity set. This script keeps a copy of its audit counts, but passes only
Baostock-confirmed SH/SZ stock delistings into CNEquity's recovery catalogue.
The resulting data remains forward-only/non-PIT.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _date(value: Any, *, label: str) -> date:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc


def validate_snapshot_freshness(
    observed_at: str,
    *,
    now: datetime | None = None,
    max_age: timedelta = timedelta(days=14),
) -> float:
    """Return snapshot age in hours; reject naive, future, or stale evidence."""
    try:
        observed = datetime.fromisoformat(str(observed_at).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid Baostock identity observed_at timestamp") from exc
    if observed.tzinfo is None:
        raise ValueError("Baostock identity observed_at must be timezone-aware")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("current time must be timezone-aware")
    age = current.astimezone(timezone.utc) - observed.astimezone(timezone.utc)
    if age < timedelta(minutes=-5):
        raise ValueError("Baostock identity snapshot timestamp is in the future")
    if age > max_age:
        raise ValueError("Baostock identity snapshot is stale")
    return max(0.0, age.total_seconds() / 3600.0)


def sanitize_catalog(
    existing_catalog: dict[str, Any],
    raw_catalog: dict[str, Any],
    formal_symbols: dict[str, str],
    listing_symbols: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge terminal observations but retain only formally delisted stocks."""
    formal = set(formal_symbols)
    if not formal:
        raise ValueError("formal delisted identity set is empty")

    raw_hits = raw_catalog.get("delisted", {})
    existing_hits = existing_catalog.get("delisted", {})
    if not isinstance(raw_hits, dict) or not isinstance(existing_hits, dict):
        raise ValueError("delisted catalogue entries must be objects")

    merged: dict[str, str] = {}
    for source in (existing_hits, raw_hits):
        for symbol, terminal in source.items():
            symbol = str(symbol).strip()
            if symbol not in formal:
                continue
            terminal_date = _date(terminal, label=f"terminal date for {symbol}")
            previous = merged.get(symbol)
            if previous is None or terminal_date > _date(previous, label=f"existing terminal date for {symbol}"):
                merged[symbol] = terminal_date.isoformat()

    # Seed every Baostock-confirmed formal delisting, even when the slower
    # Sina probe has not reached that code yet. Existing/raw terminal evidence
    # remains preferred; the official identity date fills catalogue gaps.
    identity_seeded = 0
    for symbol, terminal in formal_symbols.items():
        symbol = str(symbol).strip()
        if symbol not in merged:
            merged[symbol] = _date(
                terminal, label=f"formal identity terminal date for {symbol}"
            ).isoformat()
            identity_seeded += 1

    raw_symbols = {str(symbol).strip() for symbol in raw_hits}
    formal_hits = raw_symbols & formal
    current_listed_hits = raw_symbols & (listing_symbols - formal)
    unmatched_hits = raw_symbols - listing_symbols - formal

    sanitized = dict(raw_catalog)
    # Preserve any recovery metadata that the backfill checkpoint has added.
    for key, value in existing_catalog.items():
        if key not in sanitized:
            sanitized[key] = value
    sanitized["delisted"] = dict(sorted(merged.items()))
    # The main discovery cache is the latest authority for never-issued probes.
    sanitized["never_issued"] = sorted(set(raw_catalog.get("never_issued", [])))

    audit = {
        "raw_sina_probe_hits": len(raw_symbols),
        "formal_delisted_identity_total": len(formal),
        "formal_delisted_hits_in_raw_catalog": len(formal_hits),
        "currently_listed_hits_quarantined": len(current_listed_hits),
        "unmatched_identity_hits_quarantined": len(unmatched_hits),
        "formal_delistings_not_yet_in_raw_catalog": len(formal - raw_symbols),
        "sanitized_recovery_catalog_entries": len(merged),
        "formal_identity_seeded_into_recovery_catalog": identity_seeded,
        "current_listed_sample": sorted(current_listed_hits)[:20],
        "unmatched_sample": sorted(unmatched_hits)[:20],
        "historical_pit_eligible": False,
    }
    return sanitized, audit


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def prepare(
    *,
    config_path: Path,
    raw_catalog_snapshot: Path,
    start: date,
    end: date,
    output_dir: Path,
) -> dict[str, Any]:
    # Keep CNEquity imports local so the pure catalogue sanitizer is unit-testable.
    from cnequity.config import load_config
    from cnequity.quality.delisted_coverage import known_delisted_instruments

    config = load_config(str(config_path))
    identity_path = (
        Path(config.meta_root)
        / "quality"
        / "evidence"
        / "delisted_security_identity"
        / "baostock-v1.json"
    )
    if not identity_path.is_file():
        raise RuntimeError(f"missing Baostock identity evidence: {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (
        identity.get("status") != "complete"
        or identity.get("claim") != "delisted_security_identity"
        or identity.get("evidence_version") != 1
        or identity.get("source") != "baostock.query_stock_basic"
        or identity.get("universe_scope") != "sh_sz_stock"
        or identity.get("asset_type_filter") != "stock"
        or not isinstance(identity.get("listing_dates"), dict)
        or not isinstance(identity.get("delisted_symbols"), dict)
    ):
        raise RuntimeError("Baostock identity evidence is not a complete supported SH/SZ stock snapshot")

    identity_age_hours = validate_snapshot_freshness(identity.get("observed_at", ""))
    raw_catalog = json.loads(raw_catalog_snapshot.read_text(encoding="utf-8"))
    catalog_file = Path(config.meta_root) / "state" / "delisted_catalog.json"
    existing_catalog = (
        json.loads(catalog_file.read_text(encoding="utf-8"))
        if catalog_file.is_file()
        else {}
    )
    formal = known_delisted_instruments(config, end)
    listing_symbols = set(identity.get("listing_dates", {}))
    if any(not symbol.endswith((".SH", ".SZ")) for symbol in listing_symbols):
        raise RuntimeError("identity snapshot contains symbols outside SH/SZ stock scope")
    if any(not symbol.endswith((".SH", ".SZ")) for symbol in formal):
        raise RuntimeError("formal identity set contains symbols outside SH/SZ stock scope")
    sanitized, audit = sanitize_catalog(
        existing_catalog, raw_catalog, formal, listing_symbols
    )

    _atomic_json(catalog_file, sanitized)
    in_window = sorted(
        symbol for symbol, delist_date in formal.items()
        if _date(delist_date, label=f"formal delist date for {symbol}") >= start
    )
    all_symbols = sorted(formal)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "formal-delisted-symbols.txt").write_text(
        "\n".join(all_symbols) + ("\n" if all_symbols else ""), encoding="utf-8"
    )
    (output_dir / "formal-delisted-in-window-symbols.txt").write_text(
        "\n".join(in_window) + ("\n" if in_window else ""), encoding="utf-8"
    )

    report = {
        **audit,
        "identity_source": identity.get("source"),
        "identity_universe_scope": identity.get("universe_scope"),
        "identity_asset_type_filter": identity.get("asset_type_filter"),
        "identity_observed_at": identity.get("observed_at"),
        "identity_age_hours": round(identity_age_hours, 2),
        "identity_listing_rows": len(identity.get("listing_dates", {})),
        "identity_formal_delisted_rows": len(identity.get("delisted_symbols", {})),
        "formal_delistings_in_window": len(in_window),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "identity_evidence_path": str(identity_path),
        "sanitized_catalog_path": str(catalog_file),
        "historical_pit_eligible": False,
        "note": (
            "Current Baostock identity is used only to restrict recovery to "
            "SH/SZ stocks. Sina/TDX retrieval timestamps do not establish "
            "historical publication-time availability or complete corporate-action coverage."
        ),
    }
    _atomic_json(output_dir / "formal-recovery-preflight.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-catalog-snapshot", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2016, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(
        config_path=args.config,
        raw_catalog_snapshot=args.raw_catalog_snapshot,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
