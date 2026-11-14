"""Probe historical Eastmoney limit-up, broken-board and strong-stock pools."""
from __future__ import annotations

import json
import time
from pathlib import Path

import akshare as ak


def probe(dates=("20260601", "20261113")) -> list[dict]:
    output = Path("output/market_emotion_pool_probe.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for date in dates:
        row = {"trade_date": date, "source_id": "akshare.eastmoney"}
        for label, function_name in (
            ("limit_up", "stock_zt_pool_em"),
            ("broken_board", "stock_zt_pool_zbgc_em"),
            ("strong_pool", "stock_zt_pool_strong_em"),
        ):
            try:
                frame = getattr(ak, function_name)(date=date)
                row[label] = {
                    "status": "READY",
                    "rows": int(len(frame)),
                    "columns": [str(x) for x in frame.columns],
                    "sample": frame.head(2).where(frame.notna(), None).to_dict("records"),
                }
            except Exception as exc:
                row[label] = {
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                }
            time.sleep(0.5)
        results.append(row)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False))
    return results


if __name__ == "__main__":
    probe()
