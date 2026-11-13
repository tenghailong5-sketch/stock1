"""Small live qualification probe for CNINFO CSRC industry classification."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import akshare as ak

from src.providers.cninfo_industry import fetch_cninfo_current_industry_membership_snapshot

SAMPLES = ["000017", "002103", "300328", "920001"]


def main() -> None:
    # Test whether CNINFO accepts a comma-separated stock-code batch. If it
    # does, production can avoid one HTTP request per uncovered security.
    batch_report = {"requested": ["000017", "002103"], "returned_symbols": [], "error": None}
    try:
        batch = ak.stock_industry_change_cninfo(
            symbol="000017,002103", start_date="19900101",
            end_date=__import__("datetime").datetime.now().strftime("%Y%m%d"),
        )
        batch_report["columns"] = list(batch.columns) if batch is not None else []
        if batch is not None and not batch.empty and "证券代码" in batch.columns:
            batch_report["returned_symbols"] = sorted(
                batch["证券代码"].astype(str).str.extract(r"(\d{6})", expand=False).dropna().unique().tolist()
            )
        batch_report["returned_rows"] = int(len(batch)) if batch is not None else 0
    except Exception as exc:
        batch_report["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

    frame = fetch_cninfo_current_industry_membership_snapshot(
        ak, SAMPLES, pause_seconds=0.5
    )
    report = {
        "source_id": "cninfo.p_stock2110.csrc",
        "batch_probe": batch_report,
        "requested": SAMPLES,
        "returned_symbols": sorted(frame["symbol"].astype(str).tolist()) if not frame.empty else [],
        "returned_count": int(frame["symbol"].nunique()) if not frame.empty else 0,
        "failed_symbols": frame.attrs.get("failed_symbols", {}),
        "rows": frame.drop(columns=["observed_at", "available_time"], errors="ignore").to_dict("records"),
    }
    path = Path("output/cninfo_industry_probe.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if report["returned_count"] == 0:
        raise SystemExit("CNINFO live probe returned no CSRC classifications")


if __name__ == "__main__":
    main()
