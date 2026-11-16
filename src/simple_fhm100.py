"""Small, honest FHM-100 live quote scanner.

This replaces the oversized, mostly diagnostic runtime with a bounded scanner
that uses only fields actually present in a live A-share quote snapshot.
It produces a research watchlist, never a buy instruction.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

OUT = Path("output")
MIN_AMOUNT = 50_000_000
MIN_CHANGE = -2.0
MAX_CHANGE = 7.5
MAX_CANDIDATES = 20


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise RuntimeError(f"{source} returned an empty A-share quote snapshot")
    if {"代码", "名称", "最新价", "成交额", "涨跌幅"}.issubset(frame.columns):
        out = frame[["代码", "名称", "最新价", "成交额", "涨跌幅"]].copy()
        out.columns = ["code", "name", "price", "amount", "pct_chg"]
    elif {"code", "name", "zxj", "zdf", "turnover"}.issubset(frame.columns):
        out = frame[["code", "name", "zxj", "zdf", "turnover"]].copy()
        out.columns = ["code", "name", "price", "pct_chg", "amount"]
        # Tencent quote turnover is in 10,000 CNY.
        out["amount"] = _numeric(out["amount"]) * 10_000
    else:
        raise RuntimeError(f"{source} quote schema changed")
    out["code"] = out["code"].astype(str).str.strip().str[-6:].str.zfill(6)
    out["name"] = out["name"].astype(str).str.strip()
    for col in ("price", "amount", "pct_chg"):
        out[col] = _numeric(out[col])
    out = out[
        out["code"].str.fullmatch(r"\d{6}", na=False)
        & out["price"].gt(0)
        & out["amount"].gt(0)
        & ~out["name"].str.contains(r"ST|退", case=False, na=False)
    ]
    return out.drop_duplicates("code").reset_index(drop=True)


def fetch_quotes(ak_module=ak) -> tuple[pd.DataFrame, str]:
    """Try independent public quote endpoints; fail visibly if all fail."""
    failures = []
    providers = (
        ("eastmoney", lambda: ak_module.stock_zh_a_spot_em()),
        ("tencent", lambda: ak_module.stock_zh_a_spot_tx()),
        ("sina", lambda: ak_module.stock_zh_a_spot()),
    )
    for name, fetch in providers:
        try:
            frame = _normalize(fetch(), source=name)
            if len(frame) < 1000:
                raise RuntimeError(f"partial quote universe: {len(frame)} rows")
            return frame, name
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {str(exc)[:160]}")
    raise RuntimeError("all quote providers failed; " + " | ".join(failures))


def rank_quotes(snapshot: pd.DataFrame, *, max_candidates: int = MAX_CANDIDATES,
                min_amount: float = MIN_AMOUNT) -> tuple[pd.DataFrame, dict]:
    """Rank observable short-term strength and liquidity; missing data stays missing."""
    required = {"code", "name", "price", "amount", "pct_chg"}
    missing = required - set(snapshot.columns)
    if missing:
        raise ValueError("quote snapshot missing columns: " + ",".join(sorted(missing)))
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    data = snapshot.copy()
    data["code"] = data["code"].astype(str).str.strip().str[-6:].str.zfill(6)
    for col in ("price", "amount", "pct_chg"):
        data[col] = _numeric(data[col])
    data = data.dropna(subset=["code", "price", "amount", "pct_chg"])
    data = data[~data["name"].astype(str).str.contains(r"ST|退", case=False, na=False)]
    data = data[data["price"].gt(0) & data["amount"].ge(min_amount)]
    data = data[data["pct_chg"].between(MIN_CHANGE, MAX_CHANGE)].copy()

    if data.empty:
        audit = {
            "status": "NO_ELIGIBLE_QUOTES", "quote_count": int(len(snapshot)),
            "eligible_count": 0, "candidate_count": 0,
            "score_inputs": ["pct_chg", "amount"],
            "buy_authorization": False,
        }
        return data, audit

    eligible_count = int(len(data))
    # Percentiles are cross-sectional within this run only. There is no claim
    # of historical calibration or predictive probability.
    data["momentum_percentile"] = data["pct_chg"].rank(pct=True) * 100.0
    data["liquidity_percentile"] = data["amount"].rank(pct=True) * 100.0
    data["simple_score"] = (
        data["momentum_percentile"] * 0.65
        + data["liquidity_percentile"] * 0.35
    )
    data = data.sort_values(
        ["simple_score", "pct_chg", "amount", "code"],
        ascending=[False, False, False, True],
    ).head(max_candidates).reset_index(drop=True)
    data.insert(0, "rank", np.arange(1, len(data) + 1))
    data["candidate_type"] = "LIVE_QUOTE_WATCH"
    data["candidate_reason"] = "当日强度与成交活跃度横截面排序；无历史趋势/行业主线确认"
    data["buy_authorization"] = False
    data["entry_price"] = np.nan
    data["stop_price"] = np.nan
    audit = {
        "status": "READY_WATCHLIST",
        "quote_count": int(len(snapshot)),
        "eligible_count": eligible_count,
        "candidate_count": int(len(data)),
        "minimum_amount_cny": float(min_amount),
        "daily_change_range_pct": [MIN_CHANGE, MAX_CHANGE],
        "score_weights": {"momentum_percentile": 0.65, "liquidity_percentile": 0.35},
        "score_basis": "current_snapshot_cross_section_only",
        "historical_trend_used": False,
        "industry_mainline_used": False,
        "news_policy_catalyst_used": False,
        "intraday_15m_confirmation_used": False,
        "m2_m3_m5_m8_used": False,
        "buy_authorization": False,
    }
    return data, audit


def main() -> int:
    OUT.mkdir(exist_ok=True)
    observed_at = datetime.now(timezone.utc).isoformat()
    try:
        snapshot, provider = fetch_quotes()
        candidates, audit = rank_quotes(snapshot)
        audit.update({
            "provider": provider,
            "observed_at_utc": observed_at,
            "limitations": [
                "当前快照没有历史趋势，不能判断1-2日持续性",
                "未接入行业主线、政策新闻和资金持续集中度",
                "未接入15分钟确认、账户风险和仓位计算",
                "候选仅为观察名单，不是买入信号",
            ],
        })
        snapshot.to_csv(OUT / "market_snapshot.csv", index=False, encoding="utf-8-sig")
        candidates.to_csv(OUT / "final_candidate_shortlist.csv", index=False, encoding="utf-8-sig")
        (OUT / "simple_scan_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        lines = [
            "# FHM-100 简化版行情观察报告",
            f"- 运行时间(UTC)：{observed_at}",
            f"- 行情源：{provider}",
            f"- 有效行情：{len(snapshot)} 只",
            f"- 观察候选：{len(candidates)} 只",
            "- 评分：当日涨跌幅横截面分位 65% + 成交额横截面分位 35%",
            "- 不输出买入授权、入场价或止损价；当前数据不足以可靠计算这些内容。",
            "",
            candidates[["rank", "code", "name", "price", "pct_chg", "amount", "simple_score"]].to_markdown(index=False)
            if not candidates.empty else "没有符合基础流动性和涨幅范围的候选。",
            "",
            "## 明确未实现",
            *[f"- {item}" for item in audit.get("limitations", [])],
        ]
        (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
        (OUT / "formal_gate.json").write_text(json.dumps({
            "status": audit["status"],
            "research_run_status": "COMPLETED",
            "candidate_count": int(len(candidates)),
            "buy_authorization": False,
            "decision_policy": "publish quote-only watchlist; never issue buy authorization",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(audit, ensure_ascii=False))
        return 0
    except Exception as exc:
        failure = {
            "status": "FAILED",
            "observed_at_utc": observed_at,
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "candidate_count": 0,
            "buy_authorization": False,
        }
        (OUT / "simple_scan_audit.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(failure, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
