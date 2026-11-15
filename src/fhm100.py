"""FHM-100 research runner. No fabricated official scores or M6 probabilities."""
from __future__ import annotations
import json, os, traceback, time, uuid, hashlib
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import akshare as ak
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from src.model.m4_factors import price_structure
from src.model.m4_pit_runtime import build_runtime_m4_pit_snapshot
from src.model.m4_observation import rank_price_observations
from src.model.pipeline import evaluate_candidates, factor_readiness_audit
from src.model.m4_industry_observations import select_latest_observed_industry_snapshot
from src.model.m3_assembler import assemble_m3_history
from src.collectors.forward_archive import append_jsonl_atomic
from src.providers.akshare_pit import normalize_daily_history_snapshot
from src.model.runtime_mainline import scan_runtime_mainlines
from src.model.sw_industry_research import select_sw_industry_research_snapshot
from src.model.sw_stock_mainline_research import build_sw_stock_mainline_research
from src.providers.sw_industry_history import fetch_sw_industry_history
from src.providers.akshare_corporate_actions import archive_shortlist_corporate_actions
from src.model.live_decision_integration import (
    aggregate_archived_stock_turnover,
    build_live_decision_gate_audit, load_risk_context_file, summarize_gate_blockers,
)
from src.model.mainline_flows import attach_pit_stock_turnover_concentration
from src.model.archive_readiness import audit_forward_archive
from src.model.shortline_architecture import audit_shortline_architecture
from src.model.research_ranking import build_research_opportunity_ranking

OUT=Path("output"); OUT.mkdir(exist_ok=True)
MIN_AMOUNT=100_000_000
RESEARCH_MIN_AMOUNT=50_000_000

def numeric(s): return pd.to_numeric(s,errors="coerce")


def archive_stock_history_snapshot(frame, symbol, *, captured_at, archive_root=Path("data/forward_archive"), source_id="akshare.stock_zh_a_hist"):
    """Append raw daily bars with their real retrieval time; never backdate availability."""
    normalized = normalize_daily_history_snapshot(
        frame, symbol=str(symbol).zfill(6), fetched_at=captured_at, source_id=source_id
    )
    snapshot_id = str(uuid.uuid4())
    normalized["snapshot_id"] = snapshot_id
    normalized["snapshot_scope"] = "current_symbol_history_retrieval"
    normalized["expected_bar_count"] = int(len(normalized))
    day = pd.Timestamp(captured_at).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d")
    records = json.loads(normalized.to_json(
        orient="records", date_format="iso", force_ascii=False
    ))
    rows_added = append_jsonl_atomic(
        archive_root / "stock_daily" / f"{str(symbol).zfill(6)}.jsonl",
        records,
        # Keep identical bars from different providers as separate observations;
        # source_revision hashes row values, not the provider identity.
        ("symbol", "trade_date", "source_id", "source_revision"),
    )
    revisions = sorted(normalized["source_revision"].astype(str).tolist())
    batch_revision = hashlib.sha256("|".join(revisions).encode("utf-8")).hexdigest()
    manifest = [{
        "symbol": str(symbol).zfill(6),
        "snapshot_id": snapshot_id,
        "captured_at": pd.Timestamp(captured_at).tz_convert("UTC").isoformat(),
        "source_id": source_id,
        "source_revision": batch_revision,
        "row_count": int(len(normalized)),
        "first_trade_date": normalized["trade_date"].min().strftime("%Y-%m-%d"),
        "last_trade_date": normalized["trade_date"].max().strftime("%Y-%m-%d"),
        "availability_basis": "retrieval_time_only",
        "historical_pit_eligible": False,
        "snapshot_scope": "current_symbol_history_retrieval",
    }]
    manifests_added = append_jsonl_atomic(
        archive_root / "stock_daily_manifest" / f"{str(symbol).zfill(6)}.jsonl",
        manifest,
        ("symbol", "snapshot_id"),
    )
    return {"rows_appended": rows_added, "manifests_appended": manifests_added}


def _is_eastmoney_rate_limit(error):
    return type(error).__name__ == "Push2RateLimitError" or isinstance(
        error, getattr(ak, "RateLimitError", type(None))
    )


def _normalize_spot_snapshot(frame):
    required = ["代码", "名称", "最新价", "成交额", "涨跌幅"]
    if frame is None or frame.empty or any(column not in frame.columns for column in required):
        raise RuntimeError("A-share Sina fallback snapshot schema changed or is empty")
    return frame[required].copy()


def _normalize_tencent_spot_snapshot(frame, *, minimum_rows=1):
    """Normalize Tencent raw A-share quote fields; turnover is in 10k yuan."""
    required = {"code", "name", "zxj", "zdf", "volume", "turnover"}
    if frame is None or frame.empty or required - set(frame.columns):
        missing = sorted(required - set(frame.columns)) if frame is not None else sorted(required)
        raise RuntimeError("Tencent A-share snapshot schema changed or is empty: " + ",".join(missing))
    if len(frame) < minimum_rows:
        raise RuntimeError(
            f"Tencent A-share snapshot appears incomplete: {len(frame)} rows; expected at least {minimum_rows}"
        )
    out = frame[["code", "name", "zxj", "zdf", "turnover"]].copy()
    out["code"] = out["code"].astype(str).str.strip().str.lower().str[-6:].str.zfill(6)
    out["name"] = out["name"].astype(str).str.strip()
    out["price"] = numeric(out["zxj"])
    out["pct_chg"] = numeric(out["zdf"])
    # Tencent raw turnover is denominated in ten-thousand yuan.
    out["amount"] = numeric(out["turnover"]) * 10_000
    return out[["code", "name", "price", "amount", "pct_chg"]]


def _is_retryable_spot_error(error):
    return _is_eastmoney_rate_limit(error) or isinstance(
        error, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
    )


def universe():
    try:
        d=ak.stock_zh_a_spot_em()
    except Exception as exc:
        if not _is_retryable_spot_error(exc):
            raise
        # Eastmoney may rate-limit the runner. Try Tencent's independent,
        # paginated A-share snapshot before Sina; never silently accept partial data.
        failures = [f"Eastmoney: {type(exc).__name__}: {str(exc)[:160]}"]
        d = None
        tencent = getattr(ak, "stock_zh_a_spot_tx", None)
        if callable(tencent):
            try:
                # 4,000 rows is only a coverage floor, not proof of complete exchange-wide coverage.
                d = _normalize_tencent_spot_snapshot(tencent(), minimum_rows=4000)
            except Exception as tencent_exc:
                failures.append(f"Tencent: {type(tencent_exc).__name__}: {str(tencent_exc)[:160]}")
        else:
            failures.append("Tencent: stock_zh_a_spot_tx is unavailable in installed AKShare")
        if d is None:
            try:
                d = _normalize_spot_snapshot(ak.stock_zh_a_spot())
            except Exception as sina_exc:
                failures.append(f"Sina: {type(sina_exc).__name__}: {str(sina_exc)[:160]}")
                raise RuntimeError("All A-share spot providers failed; " + " | ".join(failures)) from sina_exc
    if d is None or d.empty:
        raise RuntimeError("A-share spot snapshot is empty")
    required=["代码","名称","最新价","成交额","涨跌幅"]
    if not all(column in d.columns for column in required):
        if {"code", "name", "price", "amount", "pct_chg"}.issubset(d.columns):
            d = d[["code", "name", "price", "amount", "pct_chg"]].copy()
        else:
            raise RuntimeError("A-share snapshot empty or schema changed")
    else:
        d=d[required].copy(); d.columns=["code","name","price","amount","pct_chg"]
    d["code"]=d.code.astype(str).str.zfill(6)
    for c in ["price","amount","pct_chg"]: d[c]=numeric(d[c])
    # Keep every valid currently quoted A-share, including STAR Market and BSE.
    # Full-market spot coverage is cheap; historical depth requests are tiered
    # separately by select_history_universe() to keep the daily job bounded.
    d=d[~d.name.str.contains(r"ST|退",na=False)]
    d=d[(d.price>0)&(d.amount>0)].copy()
    return d.sort_values(["amount","code"],ascending=[False,True]).drop_duplicates("code").reset_index(drop=True)


def _validated_spot_frame(snapshot):
    required = {"code", "name", "price", "amount", "pct_chg"}
    missing = required - set(snapshot.columns)
    if missing:
        raise ValueError(f"full-market snapshot missing columns: {sorted(missing)}")
    frame = snapshot.copy()
    frame["code"] = frame["code"].astype(str).str.strip().str[-6:].str.zfill(6)
    for column in ("price", "amount", "pct_chg"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[
        frame["code"].str.fullmatch(r"\d{6}", na=False)
        & (frame["price"] > 0) & (frame["amount"] > 0)
    ].drop_duplicates("code")


def select_history_universe(snapshot, *, max_symbols=600):
    """Select the bounded, liquidity-ranked primary deep-history pool."""
    if max_symbols < 1:
        raise ValueError("max_symbols must be positive")
    frame = _validated_spot_frame(snapshot)
    return frame.sort_values(["amount", "code"], ascending=[False, True]).head(max_symbols).reset_index(drop=True)


def select_research_history_universe(
    snapshot, *, max_symbols=800, min_candidate_amount=RESEARCH_MIN_AMOUNT,
    liquidity_share=0.20,
):
    """Select a broad short-term research pool without excluding limit-up leaders.

    max_symbols=None means scan every valid symbol in the live quote snapshot.
    The pool is retrieval priority only; inclusion is not a formal candidate or
    a buy signal. Financials and daily movers above 8% remain observable.
    """
    if max_symbols is not None and max_symbols < 1:
        raise ValueError("max_symbols must be positive or None")
    if not 0 < liquidity_share <= 1:
        raise ValueError("liquidity_share must be in (0, 1]")
    frame = _validated_spot_frame(snapshot)
    frame = frame.sort_values(
        ["amount", "code"], ascending=[False, True]
    ).reset_index(drop=True)
    if max_symbols is None or max_symbols >= len(frame):
        return frame.reset_index(drop=True)

    anchor_count = max(1, min(max_symbols, int(round(max_symbols * liquidity_share))))
    anchor = frame.head(anchor_count)
    remaining = max_symbols - len(anchor)
    if remaining <= 0:
        return anchor.reset_index(drop=True)

    # Include strong active movers, including limit-up/near-limit-up names.
    # This is a history-retrieval rule, not permission to chase a high opening.
    momentum = frame[
        (~frame["code"].isin(set(anchor["code"])))
        & (frame["amount"] >= min_candidate_amount)
        & (frame["pct_chg"] > 0)
    ].sort_values(
        ["pct_chg", "amount", "code"], ascending=[False, False, True]
    ).head(remaining)
    selected = pd.concat([anchor, momentum], ignore_index=True)
    if len(selected) < max_symbols:
        fill = frame[
            ~frame["code"].isin(set(selected["code"]))
        ].head(max_symbols - len(selected))
        selected = pd.concat([selected, fill], ignore_index=True)
    return selected.drop_duplicates("code").head(max_symbols).reset_index(drop=True)


def select_history_universe_with_supplement(
    snapshot, *, max_symbols=600, max_supplement_symbols=200,
    min_candidate_amount=MIN_AMOUNT,
):
    """Add a capped quote-prefilter pool beyond the primary liquidity-ranked history pool.

    Supplement eligibility is only a retrieval prefilter (turnover >= threshold
    and positive daily change); it is not an FHM pass or a buy signal.
    """
    if max_supplement_symbols < 0:
        raise ValueError("max_supplement_symbols cannot be negative")
    frame = _validated_spot_frame(snapshot)
    primary = select_history_universe(frame, max_symbols=max_symbols)
    primary_codes = set(primary["code"])
    eligible = frame[
        (~frame["code"].isin(primary_codes))
        & (frame["amount"] >= min_candidate_amount)
        & (frame["pct_chg"] > 0)
    ].sort_values(["amount", "code"], ascending=[False, True])
    supplement = eligible.head(max_supplement_symbols).copy()
    combined = pd.concat([primary, supplement], ignore_index=True)
    audit = {
        "primary_history_symbols": int(len(primary)),
        "supplement_rule": f"daily_amount>={int(min_candidate_amount)} and pct_chg>0",
        "supplement_eligible_symbols": int(len(eligible)),
        "supplement_selected_symbols": int(len(supplement)),
        "supplement_deferred_symbols": int(max(0, len(eligible) - len(supplement))),
        "supplement_limit": int(max_supplement_symbols),
        "history_scan_symbols": int(len(combined)),
        "supplement_is_formal_candidate": False,
    }
    return combined, audit


# [live-scan] Verify Tencent universe fallback with live market data

def _tencent_history_symbol(code):
    """Map six-digit A-share codes to Tencent exchange-prefixed symbols."""
    value = str(code).strip().lower()
    if value.startswith(("bj", "sh", "sz")):
        return value
    digits = value.zfill(6)
    if digits.startswith(("4", "8", "92")):
        return "bj" + digits
    if digits.startswith("6"):
        return "sh" + digits
    return "sz" + digits


def _download_tencent_history(code, start, *, cause):
    """Fetch and strictly normalize Tencent daily bars after Eastmoney failure."""
    symbol = _tencent_history_symbol(code)
    frame = ak.stock_zh_a_hist_tx(
        symbol=symbol, start_date=start,
        end_date=datetime.now().strftime("%Y%m%d"), adjust=""
    )
    if frame is None or frame.empty:
        raise RuntimeError("Tencent fallback returned empty daily history") from cause
    aliases = {
        "date": "日期", "open": "开盘", "high": "最高", "low": "最低",
        "close": "收盘", "volume": "成交量", "amount": "成交额",
    }
    if any(column not in frame.columns for column in aliases):
        raise RuntimeError("Tencent fallback daily-history schema changed") from cause
    normalized = frame.rename(columns=aliases).copy()
    normalized["换手率"] = np.nan
    normalized.attrs["source_id"] = "akshare.stock_zh_a_hist_tx"
    return normalized


def _download_history(code, start, max_attempts=3):
    """Retry transient transport failures, then fail over to Tencent."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    for attempt in range(max_attempts):
        try:
            return ak.stock_zh_a_hist(
                symbol=code, period="daily", start_date=start,
                end_date=datetime.now().strftime("%Y%m%d"), adjust=""
            )  # raw prices only; no look-ahead-adjusted history
        except Exception as exc:
            if _is_eastmoney_rate_limit(exc):
                return _download_tencent_history(code, start, cause=exc)
            if not isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
                raise
            if attempt + 1 < max_attempts:
                time.sleep(2 ** attempt)
                continue
            # Do not keep retrying a provider that timed out three times.
            # Switch once to the independent free Tencent daily-history source.
            return _download_tencent_history(code, start, cause=exc)


def history(code,start,archive_root=Path("data/forward_archive")):
    raw=_download_history(code,start)
    captured_at=datetime.now(timezone.utc)  # stamp immediately after provider response
    if raw is None or len(raw)<21: return None
    cols={"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","成交额":"amount"}
    if any(c not in raw.columns for c in cols): return None
    h=raw.rename(columns=cols).copy()
    for c in ["open","close","high","low","volume","amount"]: h[c]=numeric(h[c])
    h=h.dropna(subset=["open","close","high","low"]).sort_values("date")
    if len(h)<21: return None
    # Preserve the provider's original Chinese schema for the auditable archive.
    archive_frame=raw.loc[h.index].copy()
    if "换手率" not in archive_frame.columns:
        archive_frame["换手率"]=np.nan  # unavailable stays missing; never impute zero
    source_id = str(raw.attrs.get("source_id", "akshare.stock_zh_a_hist"))
    archive_stock_history_snapshot(
        archive_frame, code, captured_at=captured_at, archive_root=archive_root,
        source_id=source_id,
    )
    # The factor engine requires row-level lineage. Preserve the provider's
    # actual retrieval timestamp and content hash for every historical bar;
    # never replace these with the trade date or a batch-wide fabricated value.
    lineage = normalize_daily_history_snapshot(
        archive_frame, symbol=str(code).zfill(6), fetched_at=captured_at,
        source_id=source_id,
    )[["trade_date", "available_time", "source_id", "source_revision"]]
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h.merge(lineage, left_on="date", right_on="trade_date", how="left",
                validate="one_to_one").drop(columns=["trade_date"])
    if h[["available_time", "source_id", "source_revision"]].isna().any().any():
        raise ValueError("daily history lineage could not be attached to every bar")
    return h

def _scan_one_symbol(s, start, archive_root, history_sink=None):
    try:
        h = history(s.code, start, archive_root=archive_root)
        if h is None:
            raise ValueError("insufficient history or schema mismatch")
        if history_sink is not None:
            history_sink[str(s.code).zfill(6)] = h.copy()
        decision_at = datetime.now(timezone.utc)
        m4_bars = h.rename(columns={"date": "trade_date"}).copy()
        m4_bars["symbol"] = s.code
        m4 = price_structure(m4_bars, decision_time=decision_at.isoformat()).iloc[-1]
        raw_close = pd.to_numeric(h["close"], errors="coerce").dropna()
        raw_amount = pd.to_numeric(h["amount"], errors="coerce")
        raw_volume = pd.to_numeric(h["volume"], errors="coerce")
        quick_r5 = raw_close.pct_change(5).iloc[-1] if len(raw_close) > 5 else np.nan
        quick_r20 = raw_close.pct_change(20).iloc[-1] if len(raw_close) > 20 else np.nan
        quick_amount_ma20 = raw_amount.tail(20).mean()
        quick_volume_ratio = (
            raw_volume.iloc[-1] / raw_volume.tail(20).mean()
            if len(raw_volume) >= 20 and raw_volume.tail(20).mean() > 0 else np.nan
        )
        available_at = pd.Timestamp(m4.available_time).to_pydatetime()
        liq = pd.notna(m4.amount_ma20) and m4.amount_ma20 >= MIN_AMOUNT
        trend = (
            pd.notna(m4.return_5) and pd.notna(m4.return_20)
            and m4.return_5 > 0 and m4.return_20 > 0
        )
        structure = pd.notna(m4.CLV) and m4.CLV >= .60
        pullback = pd.notna(m4.Pullback) and m4.Pullback > -.12
        flags = []
        if not liq:
            flags.append("20日平均成交额不足1亿元/缺失")
        if m4.cross_session_price_status != "READY":
            flags.append("缺少经公司行为账本验证的点时总收益价格序列；跨日趋势/突破/回撤禁用")
        if not trend:
            flags.append("5日及20日趋势未同时确认")
        if not structure:
            flags.append("CLV收盘结构未确认")
        if not pullback:
            flags.append("回撤过深/缺失")
        row = {
            "symbol": s.code, "trade_date": m4.trade_date.strftime("%Y-%m-%d"),
            "available_time": available_at.isoformat(),
            "BodyStrength": m4.BodyStrength, "return_5": m4.return_5,
            "return_10": m4.return_10, "return_20": m4.return_20,
            "breakout20": m4.breakout20, "upper_shadow_ratio": m4.upper_shadow_ratio,
            "代码": s.code, "名称": s["name"], "最新价": s.price, "涨跌幅%": s.pct_chg,
            "当日成交额": s.amount, "20日平均成交额": m4.amount_ma20,
            "5日收益%": m4.return_5 * 100 if pd.notna(m4.return_5) else np.nan,
            "20日收益%": m4.return_20 * 100 if pd.notna(m4.return_20) else np.nan,
            "CLV": m4.CLV, "20日高点回撤%": m4.Pullback * 100 if pd.notna(m4.Pullback) else np.nan,
            "M4突破20日高点%": m4.breakout20 * 100 if pd.notna(m4.breakout20) else np.nan,
            "M4量比5日/20日": m4.volume_ma5_to_ma20,
            "M4距20日高点交易日": m4.pullback_duration,
            "M4高量滞涨": bool(m4.high_volume_stall) if pd.notna(m4.high_volume_stall) else None,
            "M4缩量回撤": bool(m4.low_volume_pullback) if pd.notna(m4.low_volume_pullback) else None,
            "快速5日收益%": quick_r5 * 100 if pd.notna(quick_r5) else np.nan,
            "快速20日收益%": quick_r20 * 100 if pd.notna(quick_r20) else np.nan,
            "快速量比20日": quick_volume_ratio,
            "快速20日平均成交额": quick_amount_ma20,
            "M4数据口径": "原始OHLC仅用于当日K线结构；跨日特征仅接受经公司行为账本验证的点时总收益序列；本次抓取时间戳不具备历史PIT资格",
            "M4跨日价格状态": m4.cross_session_price_status,
            "流动性观察通过": bool(liq),
            "技术观察通过": bool(trend and structure and pullback),
            "未通过项": "；".join(flags) if flags else "基础观察条件通过",
            "公司行为口径": "无VERIFIED点时总收益序列时，跨日收益/突破/回撤留空；原始价格不替代总收益",
        }
        return row, None
    except Exception as exc:
        return None, {"代码": s.code, "原因": str(exc)[:220]}


def scan(u, archive_root=Path("data/forward_archive"), history_sink=None, max_workers=None):
    """Fetch per-symbol history concurrently with a bounded, configurable worker pool."""
    start = (datetime.now() - pd.Timedelta(days=160)).strftime("%Y%m%d")
    if max_workers is None:
        max_workers = int(os.getenv("FHM_HISTORY_SCAN_WORKERS", "4"))
    max_workers = max(1, min(int(max_workers), 8))
    symbols = [row for _, row in u.iterrows()]
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda stock: _scan_one_symbol(stock, start, archive_root, history_sink),
            symbols,
        )
        for row, error in results:
            if row is not None:
                rows.append(row)
            if error is not None:
                errors.append(error)
    return pd.DataFrame(rows), pd.DataFrame(errors)

def ai_review(text):
    key=os.getenv("AI_API_KEY")
    if not key: return "未配置AI_API_KEY，跳过AI点评。"
    base=os.getenv("AI_API_BASE_URL","https://api.openai.com/v1").rstrip("/")
    model=os.getenv("AI_MODEL","gpt-4o-mini")
    prompt="只根据报告作风险点评，不得编造FHM分数、收益概率或买卖指令；明确数据缺口。\n"+text[:10000]
    try:
        r=requests.post(base+"/chat/completions",headers={"Authorization":"Bearer "+key},
          json={"model":model,"messages":[{"role":"user","content":prompt}],"temperature":.2},timeout=90)
        r.raise_for_status(); return r.json()["choices"][0]["message"]["content"]
    except Exception as e: return "AI点评失败："+str(e)[:250]

def push(report):
    token=os.getenv("PUSHPLUS_TOKEN")
    if not token: return "未配置PUSHPLUS_TOKEN，未推送。"
    r=requests.post("https://www.pushplus.plus/send",json={"token":token,
      "title":"FHM-100研究扫描报告","content":report[:18000],"template":"markdown"},timeout=30)
    r.raise_for_status(); d=r.json()
    if str(d.get("code")) != "200":
        raise RuntimeError(
            "PushPlus rejected the message: "
            f"code={d.get('code')} msg={str(d.get('msg', ''))[:200]}"
        )
    return f"PushPlus: {d.get('msg')} (code={d.get('code')})"


def load_m3_market_context(decision_time=None, archive_root=Path("data/forward_archive")):
    """Load only a verified, post-close M3 state; every failure remains fail-closed."""
    decision = decision_time or datetime.now(timezone.utc)
    try:
        result = assemble_m3_history(
            archive_root, index_symbol="000001", decision_time=decision
        )
    except Exception as exc:
        return {
            "status": "INVALID_DATA",
            "reason": f"m3_assembly_error:{type(exc).__name__}:{str(exc)[:180]}",
            "market_state": None, "market_environment_score": None,
            "risk_scale": None, "position_scale": None, "valid_sessions": 0,
            "latest_trade_date": None, "historical_backtest_eligible": False,
            "quality_warnings": [],
        }
    status = str(result.get("status", "INVALID_DATA"))
    state = result.get("market_state") if status == "READY" else None
    if state not in {"RISK_ON", "NEUTRAL", "RISK_OFF"}:
        state = None
        if status == "READY":
            status = "INVALID_DATA"
    environment_score = result.get("market_environment_score") if state else None
    try:
        environment_score = float(environment_score)
        if not np.isfinite(environment_score) or not 0.0 <= environment_score <= 100.0:
            environment_score = None
    except (TypeError, ValueError, OverflowError):
        environment_score = None
    return {
        "status": status, "reason": result.get("reason"), "market_state": state,
        "market_environment_score": environment_score,
        "risk_scale": result.get("risk_scale", result.get("position_scale")) if state else None,
        "position_scale": result.get("risk_scale", result.get("position_scale")) if state else None,
        "valid_sessions": int(result.get("valid_sessions", 0) or 0),
        "latest_trade_date": result.get("latest_trade_date"),
        "historical_backtest_eligible": False,
        "quality_warnings": result.get("quality_warnings", []),
    }

def audit_forward_industry_membership(archive_root, expected_symbols, *, decision_time):
    """Audit latest complete forward-only industry snapshot against this run's universe.

    This is an audit artifact only. Current membership observations are never
    treated as historical PIT membership or silently promoted into M5 factors.
    """
    root = Path(archive_root) / "industry_membership"
    files = sorted(root.glob("*.jsonl")) if root.exists() else []
    frames = []
    archive_errors = []
    for path in files:
        try:
            frame = pd.read_json(path, lines=True, dtype={"symbol": str})
        except (ValueError, OSError) as exc:
            archive_errors.append(
                f"{path.name}:{type(exc).__name__}:{str(exc)[:160]}"
            )
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), {
            "status": "invalid_archive_file" if archive_errors else "no_snapshot",
            "reason": " | ".join(archive_errors[:8]) if archive_errors else None,
            "archive_file_error_count": len(archive_errors),
            "expected_universe_size": len(set(map(str, expected_symbols))),
            "selected_symbol_count": 0,
            "membership_coverage": 0.0,
            "historical_pit_eligible": False,
        }
    membership = pd.concat(frames, ignore_index=True, sort=False)
    if "source_id" not in membership:
        return pd.DataFrame(), {
            "status": "invalid_archive_schema",
            "expected_universe_size": len(set(map(str, expected_symbols))),
            "selected_symbol_count": 0,
            "membership_coverage": 0.0,
            "historical_pit_eligible": False,
        }
    supported_sources = {
        "akshare.stock_sector_detail_sina",
        "akshare.stock_board_industry_cons_em",
        "akshare.stock_individual_info_em",
        "baostock.query_stock_industry",
        # Official SW2021 XLS snapshot is the selected complete taxonomy in the
        # forward collector. Excluding it here silently downgraded a 100%
        # snapshot to Sina's partial 46.9% snapshot during the daily audit.
        "swsresearch.sw2021_stock_classification",
    }
    membership = membership.loc[
        membership["source_id"].astype(str).isin(supported_sources)
    ].copy()
    if membership.empty:
        return pd.DataFrame(), {
            "status": "no_supported_forward_snapshot",
            "expected_universe_size": len(set(map(str, expected_symbols))),
            "selected_symbol_count": 0,
            "membership_coverage": 0.0,
            "historical_pit_eligible": False,
        }
    selected, audit = select_latest_observed_industry_snapshot(
        membership, expected_symbols, decision_time=decision_time,
        max_age="10D", min_symbol_coverage=0.90,
    )
    audit["archive_file_error_count"] = len(archive_errors)
    audit["archive_file_errors"] = archive_errors
    return selected, audit


def build_m5_diagnostic_input(watchlist, market_context, mainline=None, *, pit_factor_snapshot=None, decision_time=None):
    """Map only observed live factors into M5; leave unverified factors/gates missing.

    Industry strength comes from the current effective-date SW research overlay
    and the runtime industry panel. It is retrieval-time research context, not
    historical PIT evidence. Same-session CLV and 20-day traded value are ranked
    only within the current bounded research pool. No cross-session price,
    residual-strength, event, or risk factor is inferred from raw OHLC.
    """
    state = market_context.get("market_state")
    environment = market_context.get("market_environment_score")
    if watchlist is None or watchlist.empty:
        return pd.DataFrame(columns=["symbol", "market_state"])

    base = watchlist.copy().reset_index(drop=True)
    symbol_col = "symbol" if "symbol" in base else "代码"
    if symbol_col not in base:
        raise ValueError("watchlist requires symbol or 代码")
    base["symbol"] = base[symbol_col].astype(str).str.strip().str.zfill(6)

    # Optional formal M4->M5 bridge. It accepts only a precomputed, audited
    # PIT snapshot; current runtime overlays remain descriptive diagnostics.
    pit_by_symbol = {}
    pit_snapshot_status = "not_supplied"
    if pit_factor_snapshot is not None:
        required_pit = {
            "symbol", "trade_date", "available_time", "source_id",
            "source_revision", "availability_basis", "historical_pit_eligible",
        }
        missing_pit = required_pit - set(pit_factor_snapshot.columns)
        if missing_pit:
            raise ValueError("PIT factor snapshot missing: " + ",".join(sorted(missing_pit)))
        if decision_time is None:
            raise ValueError("decision_time is required for PIT factor mapping")
        cutoff = pd.Timestamp(decision_time)
        if cutoff.tzinfo is None:
            raise ValueError("decision_time must be timezone-aware")
        pit = pit_factor_snapshot.copy()
        pit["symbol"] = pit["symbol"].astype(str).str.strip().str.zfill(6)
        pit["trade_date"] = pd.to_datetime(pit["trade_date"], errors="coerce")
        available = pd.to_datetime(pit["available_time"], errors="coerce")
        if available.isna().any() or available.dt.tz is None:
            raise ValueError("PIT factor available_time must be timezone-aware")
        pit["available_time"] = available.dt.tz_convert("UTC")
        cutoff_utc = cutoff.tz_convert("UTC")
        decision_date = cutoff.tz_convert("Asia/Shanghai").date()
        eligible = (
            pit["historical_pit_eligible"].map(lambda value: isinstance(value, (bool, np.bool_)) and bool(value))
            & pit["availability_basis"].astype(str).str.strip().eq("derived_from_certified_pit_inputs")
            & pit["trade_date"].notna()
            & (pit["trade_date"].dt.date <= decision_date)
            & (pit["available_time"] <= cutoff_utc)
            & pit["source_id"].notna()
            & pit["source_id"].astype(str).str.strip().ne("")
            & pit["source_revision"].notna()
            & pit["source_revision"].astype(str).str.strip().ne("")
        )
        pit = pit.loc[eligible].copy()
        # Duplicate rows are ambiguous; exclude that symbol rather than choosing
        # an arbitrary provider/revision row.
        duplicate_symbols = set(pit.loc[pit["symbol"].duplicated(keep=False), "symbol"])
        pit = pit.loc[~pit["symbol"].isin(duplicate_symbols)]
        pit_by_symbol = pit.set_index("symbol").to_dict(orient="index")
        pit_snapshot_status = "validated" if pit_by_symbol else "no_eligible_rows"

    def _cross_section_percentile(column):
        if column not in base:
            return pd.Series(np.nan, index=base.index, dtype=float)
        values = pd.to_numeric(base[column], errors="coerce")
        valid = values.notna()
        result = pd.Series(np.nan, index=base.index, dtype=float)
        if valid.any():
            result.loc[valid] = values.loc[valid].rank(method="average", pct=True) * 100.0
        return result

    # Only current effective-date SW mapping is used for descriptive industry
    # strength. The source is explicitly retrieval-time-only in its audit.
    industry_strength = pd.Series(np.nan, index=base.index, dtype=float)
    industry_context_by_code = {}
    industry_code_col = "sw_history_industry_code"
    if industry_code_col in base and mainline is not None and not mainline.empty:
        required = {"industry_code", "trade_date", "strength_percentile"}
        if required.issubset(mainline.columns):
            panel = mainline.copy()
            panel["industry_code"] = panel["industry_code"].astype(str).str.strip()
            panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
            panel["strength_percentile"] = pd.to_numeric(
                panel["strength_percentile"], errors="coerce"
            )
            panel = panel.dropna(subset=["trade_date"])
            if not panel.empty:
                latest_date = panel["trade_date"].max()
                latest = panel.loc[panel["trade_date"].eq(latest_date)]
                # Ambiguous duplicate industry rows are not resolved by row order.
                counts = latest.groupby("industry_code").size()
                unique_codes = counts[counts.eq(1)].index
                unique_latest = latest.loc[latest.industry_code.isin(unique_codes)].copy()
                lookup = unique_latest.set_index("industry_code")["strength_percentile"].to_dict()
                context_columns = [
                    "mainline_phase", "phase_reason", "strength_percentile",
                    "turnover_share", "turnover_share_change_3d",
                    "startup_evidence_status", "policy_catalyst_component",
                    "negative_policy_event_component", "emotion_reversal_component",
                    "capital_concentration_component", "historical_pit_eligible",
                ]
                for record in unique_latest.to_dict(orient="records"):
                    code = str(record.get("industry_code", "")).strip()
                    industry_context_by_code[code] = {
                        key: record.get(key) for key in context_columns if key in record
                    }
                mapped_codes = base[industry_code_col].astype(str).str.strip()
                industry_strength = mapped_codes.map(lookup) * 100.0
                industry_strength = industry_strength.where(
                    industry_strength.between(0.0, 100.0)
                )

    # Negative policy evidence is a hard-risk observation, not a positive
    # catalyst score. Incomplete archives leave non-matches unknown.
    negative_policy_risk = pd.Series(None, index=base.index, dtype=object)
    if (
        mainline is not None and not mainline.empty
        and {"industry_code", "trade_date", "negative_policy_event_component"}.issubset(mainline.columns)
        and industry_code_col in base
    ):
        policy_panel = mainline.copy()
        policy_panel["trade_date"] = pd.to_datetime(
            policy_panel["trade_date"], errors="coerce"
        )
        policy_panel = policy_panel.dropna(subset=["trade_date"])
        if not policy_panel.empty:
            policy_latest_date = policy_panel["trade_date"].max()
            policy_latest = policy_panel.loc[
                policy_panel["trade_date"].eq(policy_latest_date)
            ].copy()
            policy_latest["industry_code"] = (
                policy_latest["industry_code"].astype(str).str.strip()
            )
            policy_counts = policy_latest.groupby("industry_code").size()
            unique_policy_codes = policy_counts[policy_counts.eq(1)].index
            policy_lookup = (
                policy_latest.loc[
                    policy_latest.industry_code.isin(unique_policy_codes)
                ].set_index("industry_code")["negative_policy_event_component"]
                .to_dict()
            )
            mapped_policy = (
                base[industry_code_col].astype(str).str.strip().map(policy_lookup)
            )
            negative_policy_risk = mapped_policy.map(
                lambda value: (not bool(value))
                if pd.notna(value) else None
            )

    structure_rank = _cross_section_percentile("CLV")
    trade_quality_rank = _cross_section_percentile("20日平均成交额")
    amount_ma20 = pd.to_numeric(
        base.get("20日平均成交额", pd.Series(np.nan, index=base.index)),
        errors="coerce",
    )
    rows = []
    factor_map = {
        "industry_relative_strength": "industry_relative_strength_pct",
        "stock_residual_strength": "stock_residual_strength_pct",
        "price_structure": "price_structure_pct",
        "trade_quality": "trade_quality_pct",
        "pre_pullback_strength": "pre_pullback_strength_pct",
        "pullback_reversal_structure": "pullback_reversal_structure_pct",
    }
    gate_map = (
        "industry_top_30pct", "stock_residual_top_20pct", "amount_ma20_ge_100m",
        "breakout_or_confirmed_retest",
        "no_high_volume_stall_or_repeated_rejection",
        "pre_pullback_residual_positive", "pullback_between_0_5_and_4pct",
        "pullback_volume_le_80pct_prior5d", "stop_or_reversal_confirmed",
        "industry_not_weak", "no_material_negative_event",
        "liquidity_and_risk_pass",
    )
    for idx, stock in base.iterrows():
        industry_value = industry_strength.loc[idx]
        industry_code = str(stock.get(industry_code_col, "")).strip()
        industry_context = industry_context_by_code.get(industry_code, {})
        pit_row = pit_by_symbol.get(stock["symbol"], {})
        row = {
            "symbol": stock["symbol"],
            "mainline_phase": industry_context.get("mainline_phase", "UNKNOWN"),
            "mainline_phase_reason": industry_context.get("phase_reason"),
            "mainline_strength_percentile": industry_context.get("strength_percentile"),
            "mainline_turnover_share": industry_context.get("turnover_share"),
            "mainline_turnover_share_change_3d": industry_context.get("turnover_share_change_3d"),
            "mainline_startup_evidence_status": industry_context.get("startup_evidence_status", "UNKNOWN"),
            "mainline_policy_catalyst_component": industry_context.get("policy_catalyst_component"),
            "mainline_emotion_reversal_component": industry_context.get("emotion_reversal_component"),
            "mainline_capital_concentration_component": industry_context.get("capital_concentration_component"),
            "mainline_negative_policy_component": industry_context.get("negative_policy_event_component"),
            "mainline_historical_pit_eligible": (
                bool(industry_context.get("historical_pit_eligible"))
                if isinstance(industry_context.get("historical_pit_eligible"), (bool, np.bool_))
                else False
            ),
            "market_state": state,
            # A/C factors that have an explicit live observation and a declared
            # bounded-universe ranking basis.
            "industry_relative_strength": industry_value,
            "industry_strength": industry_value,
            "price_structure": structure_rank.loc[idx],
            "trade_quality": trade_quality_rank.loc[idx],
            # A/B/C factors not supported by this run's PIT data remain absent.
            "industry_top_30pct": (
                bool(industry_value >= 70.0) if pd.notna(industry_value) else None
            ),
            "stock_residual_top_20pct": None,
            "amount_ma20_ge_100m": (
                bool(amount_ma20.loc[idx] >= MIN_AMOUNT)
                if pd.notna(amount_ma20.loc[idx]) else None
            ),
            "breakout_or_confirmed_retest": None,
            "no_high_volume_stall_or_repeated_rejection": None,
            "pre_pullback_residual_positive": None,
            "pullback_between_0_5_and_4pct": None,
            "pullback_volume_le_80pct_prior5d": None,
            "stop_or_reversal_confirmed": None,
            "industry_not_weak": (
                bool(industry_value >= 30.0) if pd.notna(industry_value) else None
            ),
            "no_material_negative_event": negative_policy_risk.loc[idx],
            "liquidity_and_risk_pass": None,
            "m5_factor_scope": "bounded_live_research_pool_percentile",
            "industry_factor_provenance": (
                "retrieval_time_only_sw_overlay" if pd.notna(industry_value)
                else "missing_or_ambiguous_industry_mapping"
            ),
        }
        if pit_row:
            # Percentiles are already ranked within their declared PIT peer set.
            # Never synthesize a score from a raw return or a forward-only overlay.
            for output_key, source_key in factor_map.items():
                value = pit_row.get(source_key)
                try:
                    number = float(value)
                    if np.isfinite(number) and 0.0 <= number <= 100.0:
                        row[output_key] = number
                except (TypeError, ValueError, OverflowError):
                    pass
            if "industry_relative_strength" in row:
                row["industry_strength"] = row["industry_relative_strength"]
            for gate in gate_map:
                value = pit_row.get(gate)
                row[gate] = bool(value) if isinstance(value, (bool, np.bool_)) else None
            row["stock_residual_top_20pct"] = (
                bool(pit_row["stock_residual_top_20pct"])
                if isinstance(pit_row.get("stock_residual_top_20pct"), (bool, np.bool_))
                else None
            )
            # PIT-derived liquidity gate overrides the live watchlist observation.
            amount_gate = pit_row.get("amount_ma20_ge_100m")
            row["amount_ma20_ge_100m"] = (
                bool(amount_gate) if isinstance(amount_gate, (bool, np.bool_)) else None
            )
            row["m5_pit_factor_provenance"] = "verified_pit_snapshot"
        else:
            row["m5_pit_factor_provenance"] = "missing_or_ineligible_pit_snapshot"
        row["m5_pit_snapshot_status"] = pit_snapshot_status
        if environment is not None:
            try:
                score = float(environment)
                if np.isfinite(score) and 0.0 <= score <= 100.0:
                    row["market_environment"] = score
            except (TypeError, ValueError, OverflowError):
                pass
        rows.append(row)
    return pd.DataFrame(rows)


def empty_scan_result(error):
    """Return explicit empty outputs when the live universe provider is unavailable."""
    # Keep this empty-result schema unique: main() applies the same final sort to all scans.
    columns = [
        "代码", "名称", "最新价", "涨跌幅%", "当日成交额", "20日平均成交额",
        "5日收益%", "20日收益%", "CLV", "20日高点回撤%",
        "M4突破20日高点%", "M4量比5日/20日", "M4距20日高点交易日",
        "M4高量滞涨", "M4缩量回撤", "M4数据口径", "M4跨日价格状态", "流动性观察通过",
        "技术观察通过", "未通过项", "公司行为口径",
    ]
    return pd.DataFrame(columns=columns), pd.DataFrame([{
        "代码": "UNIVERSE", "原因": f"行情股票池不可用：{type(error).__name__}: {str(error)[:220]}"
    }])


def ensure_watchlist_schema(watchlist):
    """Ensure an all-failed history scan still has the report's sortable columns."""
    required = {"技术观察通过", "流动性观察通过", "当日成交额"}
    if watchlist.empty and not required.issubset(watchlist.columns):
        empty_watchlist, _ = empty_scan_result(
            RuntimeError("no valid stock-history rows returned by scan")
        )
        return empty_watchlist
    return watchlist


def summarize_m5_blockers(frame):
    """Count explicit M5 rejection reasons without converting diagnostics into signals."""
    columns = ["strategy", "blocker", "count"]
    if frame is None or frame.empty or "reasons" not in frame.columns:
        return pd.DataFrame(columns=columns)
    records = []
    for row in frame.to_dict(orient="records"):
        reasons = row.get("reasons")
        if isinstance(reasons, str):
            try:
                reasons = json.loads(reasons)
            except (TypeError, ValueError):
                reasons = [reasons] if reasons else []
        if not isinstance(reasons, (list, tuple)):
            reasons = []
        for reason in reasons:
            if reason:
                records.append({
                    "strategy": str(row.get("strategy", "UNKNOWN")),
                    "blocker": str(reason),
                })
    if not records:
        return pd.DataFrame(columns=columns)
    return (pd.DataFrame(records)
            .groupby(["strategy", "blocker"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values(["strategy", "count", "blocker"],
                         ascending=[True, False, True])
            .reset_index(drop=True))


def main():
    # Run PIT archive readiness audit at every model invocation. It is
    # diagnostic only and never upgrades retrieval-time history to PIT.
    try:
        archive_readiness = audit_forward_archive(Path("data/forward_archive"))
    except Exception as exc:
        archive_readiness = {
            "status": "AUDIT_UNAVAILABLE",
            "reason": f"{type(exc).__name__}: {str(exc)[:240]}",
            "file_count": 0, "datasets": [], "files": [],
        }
    (OUT / "archive_readiness.json").write_text(
        json.dumps(archive_readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    corporate_action_audit = {
        "status": "UNAVAILABLE", "symbols_checked": 0,
        "rows_appended": 0, "historical_pit_eligible": False,
    }
    try:
        # Fetch and retain the complete live A-share quote universe. Only the
        # deeper per-symbol history requests use the bounded research pool (500 by default).
        u=universe()
        OUT.mkdir(exist_ok=True)
        u.to_csv(OUT/"market_snapshot.csv", index=False, encoding="utf-8-sig")
        # Bounded first-pass research pool; set FHM_HISTORY_SCAN_MAX_SYMBOLS=all
        # only when explicit full-market history retrieval is intended.
        # Scan inclusion never grants M5 eligibility or buy authorization.
        scan_limit_raw = os.getenv("FHM_HISTORY_SCAN_MAX_SYMBOLS", "500").strip().lower()
        if scan_limit_raw in {"all", "full", "unlimited", "0"}:
            scan_limit = None
        else:
            scan_limit = int(scan_limit_raw)
        history_universe = select_research_history_universe(u, max_symbols=scan_limit)
        history_pool_audit = {
            "selection_mode": "liquidity_anchor_plus_active_momentum_and_market_fill",
            "primary_history_symbols": int(len(history_universe)),
            "history_scan_symbols": int(len(history_universe)),
            "history_scan_limit": "ALL" if scan_limit is None else int(scan_limit),
            "full_market_history_scan": bool(len(history_universe) == len(u)),
            "financial_names_excluded_from_research_pool": False,
            "daily_gain_chase_cap_pct": None,
            "momentum_candidates_available": int(((u["amount"] >= MIN_AMOUNT) & (u["pct_chg"] > 0)).sum()),
            "formal_candidate_filter": False,
        }
        (OUT/"history_pool_audit.json").write_text(
            json.dumps(history_pool_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        kronos_histories = {}
        w,e=scan(history_universe, history_sink=kronos_histories)
        # [live-scan] verify all-failed history reporting and archive persistence
        w=ensure_watchlist_schema(w)
        if not w.empty:
            w, observation_audit = rank_price_observations(
                w, expected_universe_size=len(history_universe),
                scope_id=f"shortline_research_pool_{len(history_universe)}"
            )
        else:
            observation_audit = pd.DataFrame()
        # Capture corporate-action changes only for the bounded live shortlist.
        # Each event becomes available at first observation, never at its old
        # announcement date; CAPTURED records are not VERIFIED total returns.
        corporate_action_audit = {
            "status": "UNAVAILABLE", "symbols_checked": 0,
            "rows_appended": 0, "historical_pit_eligible": False,
        }
        try:
            corporate_action_audit = archive_shortlist_corporate_actions(
                w, ak_module=ak, observed_at=lambda: datetime.now(timezone.utc),
                archive_root=Path("data/forward_archive"), max_symbols=20,
            )
        except Exception as action_exc:
            corporate_action_audit = {
                "status": "UNAVAILABLE",
                "reason": f"{type(action_exc).__name__}: {str(action_exc)[:240]}",
                "historical_pit_eligible": False,
            }
        (OUT/"corporate_action_ledger_audit.json").write_text(
            json.dumps(corporate_action_audit, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as exc:
        # Provider outages/rate limits must produce an auditable report, not a false clean scan.
        u=pd.DataFrame()
        history_universe=pd.DataFrame()
        history_pool_audit = {
            "status": "UNAVAILABLE",
            "reason": str(exc)[:300],
            "primary_history_symbols": 0,
            "supplement_selected_symbols": 0,
            "supplement_deferred_symbols": 0,
            "history_scan_symbols": 0,
        }
        (OUT/"history_pool_audit.json").write_text(
            json.dumps(history_pool_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        w,e=empty_scan_result(exc)
        observation_audit = pd.DataFrame()
    # Freeze one decision cutoff only after all live stock-history requests finish.
    # M3 and every captured stock bar are therefore evaluated against the same
    # cutoff; no provider response fetched after the cutoff can leak into the run.
    now=datetime.now(timezone.utc)
    # Runtime mainline scan is part of this model invocation; no scheduler is involved.
    runtime_mainline, runtime_mainline_audit = scan_runtime_mainlines(decision_time=now)
    runtime_mainline.to_csv(OUT/"runtime_mainline_latest.csv", index=False, encoding="utf-8-sig")
    (OUT/"runtime_mainline_audit.json").write_text(
        json.dumps(runtime_mainline_audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if not w.empty and "available_time" in w.columns:
        stock_available = pd.to_datetime(w["available_time"], errors="coerce", utc=True)
        if stock_available.isna().any() or (stock_available > pd.Timestamp(now)).any():
            raise RuntimeError("stock snapshot availability exceeds the unified decision cutoff")
        latest_stock_available = stock_available.max().isoformat()
    else:
        latest_stock_available = None
    m3 = load_m3_market_context(decision_time=now)
    expected_symbols = u["code"].astype(str).str.zfill(6).tolist() if "code" in u else []
    industry_membership, industry_audit = audit_forward_industry_membership(
        Path("data/forward_archive"), expected_symbols, decision_time=now
    )
    industry_membership.to_csv(
        OUT/"industry_membership_forward_audit.csv", index=False, encoding="utf-8-sig"
    )
    (OUT/"industry_membership_forward_audit.json").write_text(
        json.dumps(industry_audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # Shenwan effective-date overlay is research context only. Retrieval-time
    # history never enters PIT membership authorization, M5 scores, or ranking.
    sw_history_audit = {"status": "UNAVAILABLE", "historical_pit_eligible": False,
                        "trading_authorization": False}
    try:
        sw_history = fetch_sw_industry_history()
        sw_research, sw_history_audit = select_sw_industry_research_snapshot(
            sw_history, expected_symbols,
            as_of_date=now.astimezone(ZoneInfo("Asia/Shanghai")).date(),
        )
        sw_stock_mainline, sw_stock_mainline_audit = build_sw_stock_mainline_research(
            sw_research, runtime_mainline,
            as_of_date=now.astimezone(ZoneInfo("Asia/Shanghai")).date(),
        )
        sw_stock_mainline.to_csv(
            OUT/"sw_stock_mainline_research.csv", index=False, encoding="utf-8-sig"
        )
        (OUT/"sw_stock_mainline_research_audit.json").write_text(
            json.dumps(sw_stock_mainline_audit, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        sw_research.to_csv(
            OUT/"sw_industry_history_research.csv", index=False, encoding="utf-8-sig"
        )
        if not sw_research.empty and "symbol" in w.columns:
            overlay = sw_research[[
                "symbol", "industry_code", "l1_code", "l2_code", "valid_from", "valid_to"
            ]].rename(columns={
                "industry_code": "sw_history_industry_code",
                "l1_code": "sw_history_l1_code",
                "l2_code": "sw_history_l2_code",
                "valid_from": "sw_history_valid_from",
                "valid_to": "sw_history_valid_to",
            })
            w["symbol"] = w["symbol"].astype(str).str.strip().str.zfill(6)
            w = w.merge(overlay, on="symbol", how="left", validate="many_to_one")
    except Exception as exc:
        sw_history_audit = {
            "status": "UNAVAILABLE",
            "reason": f"{type(exc).__name__}: {str(exc)[:240]}",
            "historical_pit_eligible": False,
            "trading_authorization": False,
        }
        sw_stock_mainline_audit = {
            "status": "UNAVAILABLE",
            "reason": f"{type(exc).__name__}: {str(exc)[:240]}",
            "historical_pit_eligible": False,
            "trading_authorization": False,
            "research_only": True,
        }
        pd.DataFrame().to_csv(
            OUT/"sw_stock_mainline_research.csv", index=False, encoding="utf-8-sig"
        )
        (OUT/"sw_stock_mainline_research_audit.json").write_text(
            json.dumps(sw_stock_mainline_audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (OUT/"sw_industry_history_research_audit.json").write_text(
        json.dumps(sw_history_audit, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # PR128: aggregate individual stock traded value by PIT industry membership,
    # then attach concentration/share changes to the M3 mainline panel.
    latest_stock_day = (
        pd.to_datetime(w["trade_date"], errors="coerce").max()
        if not w.empty and "trade_date" in w else
        pd.to_datetime(runtime_mainline["trade_date"], errors="coerce").max()
        if not runtime_mainline.empty and "trade_date" in runtime_mainline else
        pd.NaT
    )
    pit_stock_turnover = pd.DataFrame(columns=[
        "trade_date", "industry_code", "turnover", "available_time",
        "decision_time", "availability_basis", "historical_pit_eligible",
    ])
    stock_turnover_audit = {
        "status": "no_target_trade_date", "historical_pit_eligible": False,
    }
    if pd.notna(latest_stock_day) and expected_symbols:
        pit_stock_turnover, stock_turnover_audit = aggregate_archived_stock_turnover(
            Path("data/forward_archive"), expected_symbols,
            trade_date=latest_stock_day, decision_time=now,
        )
    if not runtime_mainline.empty:
        runtime_mainline = attach_pit_stock_turnover_concentration(
            runtime_mainline, pit_stock_turnover
        )
    runtime_mainline.to_csv(
        OUT/"runtime_mainline_with_stock_concentration.csv",
        index=False, encoding="utf-8-sig",
    )
    (OUT/"stock_turnover_concentration_audit.json").write_text(
        json.dumps(stock_turnover_audit, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    # Cross-sectional quick score is an exploratory ranking only. It uses raw
    # unadjusted history and is not an official FHM-100 A/B/C score.
    quick_columns = ["快速5日收益%", "快速20日收益%", "CLV", "快速量比20日", "快速20日平均成交额"]
    if not w.empty and all(column in w.columns for column in quick_columns):
        percentile = w[quick_columns].apply(pd.to_numeric, errors="coerce").rank(pct=True)
        # Research ordering is not a formal buy gate. Rank with any three
        # observed dimensions; do not discard early reversals, pullbacks,
        # lower-CLV structures or smaller-cap theme leaders at this stage.
        w["快速初筛分"] = (percentile.mean(axis=1) * 100).where(percentile.notna().sum(axis=1) >= 3)
        w["快速初筛通过"] = w["快速初筛分"].notna()
    else:
        w["快速初筛分"] = np.nan
        w["快速初筛通过"] = False
    w=w.sort_values(["快速初筛通过","快速初筛分","技术观察通过","流动性观察通过","当日成交额"],
                    ascending=[False,False,False,False,False])
    # Load explicitly certified PIT archives when configured. Ordinary
    # forward/retrieval-time archives are never upgraded into historical PIT.
    m4_pit_factor_snapshot, m4_pit_factor_audit, m4_pit_snapshot_audit = (
        build_runtime_m4_pit_snapshot(
            decision_time=now,
            input_dir=os.getenv("FHM_M4_PIT_INPUT_DIR", "").strip() or None,
            coverage_floor=os.getenv("FHM_M4_PIT_COVERAGE_FLOOR", "0.80"),
        )
    )
    m4_pit_factor_snapshot.to_csv(
        OUT/"m4_pit_factor_snapshot.csv", index=False, encoding="utf-8-sig"
    )
    m4_pit_factor_audit.to_csv(
        OUT/"m4_pit_factor_audit.csv", index=False, encoding="utf-8-sig"
    )
    (OUT/"m4_pit_snapshot_audit.json").write_text(
        json.dumps(m4_pit_snapshot_audit, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    m5_input = build_m5_diagnostic_input(
        w, m3, runtime_mainline,
        pit_factor_snapshot=m4_pit_factor_snapshot,
        decision_time=now,
    )
    # Keep broad discovery moving independently of formal M5 readiness. This
    # ranking uses only observed same-session/industry diagnostics and cannot
    # be mistaken for a formal score or buy authorization.
    research_opportunities, research_opportunity_audit = build_research_opportunity_ranking(
        w, m5_input, min_observed_factors=3,
    )
    research_opportunities.to_csv(
        OUT/"research_opportunity_rank.csv", index=False, encoding="utf-8-sig"
    )
    (OUT/"research_opportunity_audit.json").write_text(
        json.dumps(research_opportunity_audit, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    m5 = evaluate_candidates(m5_input, stage="close") if not m5_input.empty else pd.DataFrame(
        columns=["symbol","strategy","stage","score","eligible","candidate_band","minimum_score","reasons"])
    m5.to_csv(OUT/"m5_diagnostics.csv",index=False,encoding="utf-8-sig")
    m5_readiness = factor_readiness_audit(m5_input, stage="close")
    m5_readiness.to_csv(OUT/"m5_factor_readiness.csv", index=False, encoding="utf-8-sig")
    shortline_architecture, shortline_pillar_audit = audit_shortline_architecture(m5_input)
    (OUT/"shortline_architecture_audit.json").write_text(
        json.dumps(shortline_architecture, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    shortline_pillar_audit.to_csv(
        OUT/"shortline_pillar_audit.csv", index=False, encoding="utf-8-sig"
    )
    m5_blockers = summarize_m5_blockers(m5)
    m5_blockers.to_csv(OUT/"m5_blocker_summary.csv",index=False,encoding="utf-8-sig")

    # Optional CNEquity historical mainline replay. It is a separate research
    # lane: reconstructed historical membership/industry indices never become
    # historical PIT trading evidence or alter today's live candidate gates.
    cnequity_mainline_audit = {
        "status": "DISABLED",
        "historical_pit_eligible": False,
        "trading_authorization": False,
    }
    cnequity_enabled = os.getenv(
        "FHM_CNEQUITY_MAINLINE_RESEARCH", "0"
    ).strip().lower() in {"1", "true", "yes"}
    if cnequity_enabled:
        try:
            from src.providers.cnequity_mainline import load_cnequity_mainline_inputs
            from src.model.mainline_flows import classify_mainlines

            cnequity_root = (
                os.getenv("FHM_CNEQUITY_DATA_ROOT")
                or os.getenv("CNE_DATA_ROOT")
            )
            cnequity_start = os.getenv("FHM_CNEQUITY_RESEARCH_START", "2020-01-01")
            cnequity_end = now.astimezone(
                ZoneInfo("Asia/Shanghai")
            ).date().isoformat()
            cnequity_panel, cnequity_mainline_audit = load_cnequity_mainline_inputs(
                start=cnequity_start,
                end=cnequity_end,
                data_root=cnequity_root,
            )
            if not cnequity_panel.empty:
                cnequity_history = classify_mainlines(cnequity_panel)
                cnequity_history.to_csv(
                    OUT/"cnequity_historical_mainline_research.csv",
                    index=False, encoding="utf-8-sig",
                )
                cnequity_mainline_audit["phase_counts"] = (
                    cnequity_history["mainline_phase"].value_counts().to_dict()
                )
                cnequity_mainline_audit["status"] = "READY_RESEARCH_ONLY"
            else:
                cnequity_mainline_audit["status"] = "NO_ALIGNED_HISTORY"
        except Exception as exc:
            cnequity_mainline_audit = {
                "status": "UNAVAILABLE",
                "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
                "historical_pit_eligible": False,
                "trading_authorization": False,
            }
    (OUT/"cnequity_historical_mainline_audit.json").write_text(
        json.dumps(cnequity_mainline_audit, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # Optional Kronos research sidecar. It runs only when explicitly enabled,
    # uses the exact history snapshots already fetched above, and never alters
    # candidate ordering, M5 scores, or decision-gate authorization.
    kronos_audit = {"status": "DISABLED", "historical_pit_eligible": False}
    if os.getenv("FHM_KRONOS_ENABLED", "0").strip().lower() in {"1", "true", "yes"}:
        try:
            from src.model.kronos_batch import forecast_candidate_pool
            calendar = ak.tool_trade_date_hist_sina()
            if "trade_date" not in calendar.columns:
                raise ValueError("official trading calendar missing trade_date")
            sessions = pd.to_datetime(calendar["trade_date"], errors="coerce").dropna().sort_values()
            decision_market_date = now.astimezone(ZoneInfo('Asia/Shanghai')).date()
            future_sessions = sessions[sessions.dt.date > decision_market_date].head(5)
            if len(future_sessions) < 5:
                raise ValueError("official calendar has fewer than five future sessions")
            candidate_rows = w.copy()
            if "快速初筛分" in candidate_rows:
                candidate_rows = candidate_rows.sort_values("快速初筛分", ascending=False)
            candidate_symbols = candidate_rows["symbol"].astype(str).str.zfill(6).drop_duplicates().head(20).tolist()
            selected_histories = {symbol: kronos_histories[symbol] for symbol in candidate_symbols if symbol in kronos_histories}
            kronos_result = forecast_candidate_pool(
                selected_histories, decision_time=now,
                future_timestamps=future_sessions.tolist(), max_candidates=20,
            )
            kronos_result["status"] = "RESEARCH_ONLY_" + str(kronos_result.get("status"))
            kronos_audit = kronos_result
            pd.DataFrame([
                {"symbol": symbol, **result.get("metrics", {})}
                for symbol, result in kronos_result.get("results", {}).items()
            ]).to_csv(OUT/"kronos_forecasts.csv", index=False, encoding="utf-8-sig")
        except Exception as exc:
            kronos_audit = {
                "status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {str(exc)[:240]}",
                "historical_pit_eligible": False, "trading_signal": False,
            }
    (OUT/"kronos_audit.json").write_text(
        json.dumps(kronos_audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # Execute M2 -> M3 -> M5 -> M8 on every strategy row. Missing PIT,
    # factor, 15-minute or portfolio-risk inputs remain explicit blockers.
    # Bound the expensive live 15m confirmation pass by default. The full
    # M5 candidate pool remains in diagnostics; operators can opt into all
    # candidates with FHM_15M_SCAN_LIMIT=all.
    minute_limit_raw = os.getenv("FHM_15M_SCAN_LIMIT", "80").strip().lower()
    minute_scan_limit = (
        None if minute_limit_raw in {"all", "full", "unlimited", "0"}
        else int(minute_limit_raw)
    )
    risk_context_by_symbol, risk_context_audit = load_risk_context_file(
        os.getenv("FHM_RISK_CONTEXT_FILE", "").strip() or None,
        raw_json=os.getenv("FHM_RISK_CONTEXT_JSON", ""),
    )
    decision_gates, decision_gate_audit = build_live_decision_gate_audit(
        w, m5, m3, runtime_mainline, industry_membership,
        decision_time=now, minute_scan_limit=minute_scan_limit,
        risk_context_by_symbol=risk_context_by_symbol,
    )
    decision_gate_audit["risk_context_source"] = risk_context_audit
    (OUT/"m8_risk_context_audit.json").write_text(
        json.dumps(risk_context_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    decision_gates.to_csv(
        OUT/"four_layer_decision_gates.csv", index=False, encoding="utf-8-sig"
    )
    gate_blockers = summarize_gate_blockers(decision_gates)
    gate_blockers.to_csv(
        OUT/"four_layer_gate_blockers.csv", index=False, encoding="utf-8-sig"
    )
    (OUT/"four_layer_decision_audit.json").write_text(
        json.dumps(decision_gate_audit, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # A successful research run is not equivalent to a formal FHM-100 signal.
    # Keep data blockers separate from unfinished model integration so an empty
    # universe can never be misreported as "no stocks passed the strategy".
    formal_blockers = []
    if archive_readiness.get("status") != "PIT_READY":
        formal_blockers.append("HISTORICAL_PIT_ARCHIVE_NOT_READY")
    if len(u) == 0:
        formal_blockers.append("LIVE_UNIVERSE_UNAVAILABLE")
    if len(w) == 0:
        formal_blockers.append("NO_VALID_STOCK_HISTORY")
    if m3.get("status") != "READY":
        formal_blockers.append("M3_NOT_READY:" + str(m3.get("status")))
    if industry_audit.get("status") != "ready_forward_only":
        formal_blockers.append("FORWARD_INDUSTRY_SNAPSHOT_NOT_READY:" + str(industry_audit.get("status")))
    # M5 is enforced per strategy/candidate in the four-layer gate above;
    # M6 is an optional prediction sidecar, not a prerequisite of M2->M3->M5->M8.
    # Keep only the independent execution-validation requirement here.
    formal_blockers.append("M9_EXECUTION_AWARE_BACKTEST_NOT_ADMITTED")
    if decision_gate_audit.get("m8_risk_context_complete_symbol_count", 0) == 0:
        formal_blockers.append("M8_DYNAMIC_RISK_CONTEXT_NOT_AVAILABLE")
    if decision_gate_audit.get("allowed_count", 0) == 0:
        formal_blockers.append("FOUR_LAYER_DECISION_GATE_BLOCKED_ALL_CANDIDATES")
    formal_status = "BLOCKED_FORMAL_SIGNAL" if formal_blockers else "READY"
    (OUT/"formal_gate.json").write_text(json.dumps({
        "status": formal_status,
        "blockers": formal_blockers,
        "live_universe_count": int(len(u)),
        "history_scan_universe_count": int(len(history_universe)),
        "history_pool_audit": history_pool_audit,
        "valid_history_count": int(len(w)),
        "m3_status": str(m3.get("status")),
        "m5_is_diagnostic_only": True,
        "four_layer_decision_integrated": True,
        "four_layer_decision_audit": decision_gate_audit,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT/"m3_market_context.json").write_text(json.dumps(m3, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    w.to_csv(OUT/"watchlist.csv",index=False,encoding="utf-8-sig")
    e.to_csv(OUT/"data_errors.csv",index=False,encoding="utf-8-sig")
    observation_audit.to_csv(OUT/"m4_observation_audit.csv",index=False,encoding="utf-8-sig")
    c=w[w["技术观察通过"]&w["流动性观察通过"]]
    lines=["# FHM-100 自动研究扫描报告",f"- 运行时间：{now.isoformat()}",
      f"- 全市场实时行情快照：{len(u)}只（含沪深北及科创板；见market_snapshot.csv）",
      f"- 深度历史扫描池：{len(history_universe)}只；扫描上限={history_pool_audit.get('history_scan_limit')}；全市场扫描={history_pool_audit.get('full_market_history_scan', False)}",
      f"- 扫描范围包含金融股及涨幅超过8%的活跃股；纳入扫描不等于模型通过或追涨买入",
      f"- 历史数据成功：{len(w)}",f"- 数据异常：{len(e)}",
      f"- 行情采集状态：{'正常' if len(u) else '失败/不可用；本次不产生候选'}",
      f"- 基础技术观察名单：{len(c)}",
      f"- 快速初筛通过：{int(w['快速初筛通过'].sum()) if '快速初筛通过' in w else 0}只（临时探索排名，不是正式FHM分数）",
      f"- M4价格/流动性观察因子已计算：{len(w)}",
      f"- M4观察分位审计记录：{len(observation_audit)}条",
      f"- M4 PIT→M5桥接：{m4_pit_snapshot_audit.get('status')}；本次未伪造PIT输入",
      f"- 个股行情最新可用时间：{latest_stock_available or '无'}；统一决策时点：{now.isoformat()}",
      f"- 历史PIT档案审计：{archive_readiness.get('status')}；档案文件={archive_readiness.get('file_count', 0)}；明细见archive_readiness.json",
      f"- 运行时行业主线扫描：{runtime_mainline_audit.get('status')}；最新交易日={runtime_mainline_audit.get('latest_trade_date', '无')}；最新行业数={runtime_mainline_audit.get('latest_industry_rows', 0)}；启动证据确认={runtime_mainline_audit.get('confirmed_startups', 0)}（诊断用途，不接入正式评分）",
      f"- 市场情绪证据：{runtime_mainline_audit.get('emotion', {}).get('sessions', 0)}个历史会话；政策归档完整性={runtime_mainline_audit.get('policy', {}).get('archive_complete', False)}（缺失政策不判定为无催化）",
      f"- M3市场状态：{m3['status']}；状态={m3['market_state'] or '未确认'}；有效会话={m3['valid_sessions']}；最新日期={m3['latest_trade_date'] or '无'}",
      f"- M3仓位系数：{m3['position_scale'] if m3['position_scale'] is not None else '不可用'}；原因={m3['reason'] or '无'}",
      f"- 前瞻行业快照：{industry_audit.get('status')}；覆盖={industry_audit.get('selected_symbol_count', 0)}/{industry_audit.get('expected_universe_size', 0)}；仅作当日观察审计，不具备历史PIT资格",
      f"- 申万有效日期行业映射：{sw_history_audit.get('status')}；覆盖={sw_history_audit.get('matched_symbols', 0)}/{sw_history_audit.get('expected_symbols', 0)}；仅作研究展示，不参与评分或交易授权",
      f"- 申万个股-行业主线研究映射：{sw_stock_mainline_audit.get('status')}；覆盖={sw_stock_mainline_audit.get('matched_symbols', 0)}/{sw_stock_mainline_audit.get('active_membership_symbols', 0)}；仅作研究展示，不具备历史PIT或交易授权",
      f"- 公司行为前瞻账本：{corporate_action_audit.get('status')}；候选扫描={corporate_action_audit.get('symbols_checked', 0)}只；新增事件版本={corporate_action_audit.get('rows_appended', 0)}；仅首次观察之后可用，未验证总回报，不参与评分或交易授权",
      f"- 五层短线架构审计：{shortline_architecture.get('status')}；候选={shortline_architecture.get('candidate_count', 0)}；PIT可用={shortline_architecture.get('pit_backed_candidate_count', 0)}；权重保持冻结",
      f"- 五层短线架构审计：{shortline_architecture.get('status')}；候选={shortline_architecture.get('candidate_count', 0)}；PIT可用={shortline_architecture.get('pit_backed_candidate_count', 0)}；权重保持冻结",
      f"- 研究机会排序：{research_opportunity_audit.get('ranked_count', 0)}/{research_opportunity_audit.get('row_count', 0)}只；仅用当日K线结构、成交质量和当前行业观察，不使用未验证跨日收益，不是正式M5分数或买入授权",
      f"- M5诊断记录：{len(m5)}（每只股票A/B/C各一条；不代表正式评分）",
      f"- 四层决策门：{decision_gate_audit.get('status')}；检查记录={decision_gate_audit.get('candidate_count', 0)}；15分钟扫描={decision_gate_audit.get('minute_scanned_count', 0)}；准入={decision_gate_audit.get('allowed_count', 0)}；阻断={decision_gate_audit.get('blocked_count', 0)}",
      f"- 正式模型状态：{formal_status}",
      "## 正式模型阻塞项",
      *[f"- {reason}" for reason in formal_blockers],
      "## 运行时行业主线观察（非正式选股信号）",
      (runtime_mainline.head(15).to_string(index=False) if not runtime_mainline.empty else "本次主线扫描不可用：" + str(runtime_mainline_audit.get('reason', runtime_mainline_audit.get('status')))),
      "## 四层决策门阻塞项",
      (gate_blockers.head(20).to_string(index=False) if len(gate_blockers) else "无四层决策记录。"),
      "## M5主要阻塞项",
      (m5_blockers.head(15).to_string(index=False) if len(m5_blockers) else "本次没有股票诊断记录；这不代表模型阻塞项为零。"), "",
      "## 模型状态（重要）",
      "**本次不是正式FHM-100信号。** 行业相对强度、市场/行业滚动Beta残差、完整涨跌停事件、经样本外校准的M6收益预测、M8组合风险及M9回测尚未完成。",
      "M4已新增CLV、BodyStrength、5/10/20日收益、20日突破和上影线的横截面观察分位，并输出m4_observation_audit.csv。分位仅在同一交易日、声明的当前高成交额股票池完整覆盖且因子覆盖率达到80%；低于80%则不生成分位时生成；不完整则全部留空。它们不是价格结构/成交质量复合分，也未接入M5。M3仅在前瞻归档状态READY时传入M5；行业/残差分位及策略硬门槛仍未全部接入，因此M5仍是诊断而非正式评分。",
      "不输出A/B/C正式评分、P_up、预期收益或买入指令；观察名单不等于达到80分门槛。AI仅作文字点评，不参与评分、不放宽硬门槛。",
      "价格口径：本扫描仅用未复权原始日线；尚无点时公司行为账本，分红送转附近的跨日收益/均线可能失真，相关观察值不得进入正式模型评分或回测标签。",
      "## 研究机会排序（研究观察，不是正式M5分数/买入指令）",
      (research_opportunities.head(20).to_markdown(index=False)
       if not research_opportunities.empty else "本次没有达到至少3项可观察因子的研究排名记录。"),
      "## 快速初筛排名（临时研究结果，非正式FHM信号/非买入指令）",
      (w[w["快速初筛通过"]].sort_values("快速初筛分", ascending=False).head(20).to_markdown(index=False)
       if "快速初筛通过" in w and w["快速初筛通过"].any()
       else "暂无股票通过快速初筛条件。"),
      "## 技术观察名单（非买入信号）",
      c.head(30).to_markdown(index=False) if len(c) else "暂无满足基础技术观察条件的股票。",
      "## 数据异常",e.head(30).to_markdown(index=False) if len(e) else "无记录"]
    report="\n".join(lines)+"\n\n## AI辅助风险点评\n"+ai_review("\n".join(lines))
    (OUT/"report.md").write_text(report,encoding="utf-8")
    push_error = None
    try:
        status = push(report)
    except Exception as exc:
        status = "PushPlus推送失败：" + str(exc)[:300]
        push_error = exc
    (OUT/"run_status.json").write_text(json.dumps({"time":now.isoformat(),"universe":len(u),
      "history_scan_universe":len(history_universe),"history_pool_audit":history_pool_audit,
      "history_ok":len(w),"errors":len(e),"watchlist":len(c),"push":status,
      "archive_readiness":archive_readiness,
      "m4_pit_snapshot_audit":m4_pit_snapshot_audit,
      "research_opportunity_audit":research_opportunity_audit,
      "shortline_architecture_audit":shortline_architecture,
      "runtime_mainline_audit":runtime_mainline_audit,
      "sw_industry_history_research_audit":sw_history_audit,
      "sw_stock_mainline_research_audit":sw_stock_mainline_audit,
      "corporate_action_ledger_audit":corporate_action_audit,
      "formal_status":formal_status,"formal_blockers":formal_blockers},ensure_ascii=False,indent=2),encoding="utf-8")
    print(report); print(status)
    if push_error is not None:
        raise RuntimeError(status) from push_error

if __name__=="__main__":
    try: main()
    except Exception: traceback.print_exc(); raise
