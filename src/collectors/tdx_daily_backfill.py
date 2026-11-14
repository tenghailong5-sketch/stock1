"""Download and archive one full-market TongDaXin daily package."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from src.providers.tdx_daily_package import audit_tdx_snapshot, fetch_tdx_daily_package


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


def collect(trade_date: str, output_root: Path) -> dict:
    frame = fetch_tdx_daily_package(trade_date)
    audit = audit_tdx_snapshot(frame)
    day = audit["trade_date"]
    directory = output_root / day
    directory.mkdir(parents=True, exist_ok=True)
    # Keep one row per security; CSV is compact and easy to inspect/download.
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
    parser.add_argument("--date", required=True, help="Trade date YYYY-MM-DD")
    parser.add_argument("--output", default="data/market_daily")
    args = parser.parse_args()
    result = collect(args.date, Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
