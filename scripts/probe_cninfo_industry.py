"""Small live qualification probe for CNINFO CSRC industry classification."""
from __future__ import annotations

import json
from pathlib import Path

import akshare as ak

from src.providers.cninfo_industry import fetch_cninfo_current_industry_membership_snapshot

SAMPLES = ["000001", "600519", "300750"]


def main() -> None:
    frame = fetch_cninfo_current_industry_membership_snapshot(
        ak, SAMPLES, pause_seconds=0.5
    )
    report = {
        "source_id": "cninfo.p_stock2110.csrc",
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
