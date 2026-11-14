"""Download and archive a full-market TongDaXin daily package."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.providers.tdx_daily_package import (
    audit_tdx_snapshot, fetch_tdx_daily_package,
)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def fetch_latest_available_package(*, today=None) -> pd.DataFrame:
    """Find the latest published weekday package; exchange holidays naturally 404."""
    local_today = (
        pd.Timestamp.now(tz=ZoneInfo("Asia/Shanghai")).date()
        if today is None else pd.Timestamp(today).date()
    )
    failures = []
    for offset in range(0, 16):
        day = local_today - timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        try:
            return fetch_tdx_daily_package(day.isoformat())
        except ValueError as exc:
            # The provider uses 404 for holidays, not-yet-published dates and
            # dates outside retention. Continue to the preceding weekday.
            failures.append(f"{day.isoformat()}:{str(exc)[:120]}")
    raise RuntimeError(
        "No published TongDaXin package found in the last 15 calendar days: "
        + " | ".join(failures[-6:])
    )


def collect(trade_date: str, output_root: Path) -> dict:
    frame = (
        fetch_latest_available_package()
        if str(trade_date).strip().lower() == "latest"
        else fetch_tdx_daily_package(trade_date)
    )
    audit = audit_tdx_snapshot(frame)
    day = audit["trade_date"]
    directory = output_root / day
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "full_market_daily.csv"
    manifest_path = directory / "manifest.json"
    _atomic_write(csv_path, frame.to_csv(index=False).encode("utf-8-sig"))
    _atomic_write(
        manifest_path,
        json.dumps(audit, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    return {**audit, "csv_path": str(csv_path), "manifest_path": str(manifest_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="latest", help="Trade date YYYY-MM-DD or latest")
    parser.add_argument("--output", default="data/market_daily")
    args = parser.parse_args()
    result = collect(args.date, Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
            handle.write(f"trade_date={result['trade_date']}\\n")


if __name__ == "__main__":
    main()
