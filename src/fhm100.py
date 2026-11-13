"""FHM-100 research runner. No fabricated official scores or M6 probabilities."""
from __future__ import annotations
import json, os, traceback, time, uuid, hashlib
from datetime import datetime, timezone
from pathlib import Path
import akshare as ak
import numpy as np
import pandas as pd
import requests
from src.model.m4_factors import price_structure
from src.model.m4_observation import rank_price_observations
from src.model.pipeline import evaluate_candidates
from src.model.m4_industry_observations import select_latest_observed_industry_snapshot
from src.model.m3_assembler import assemble_m3_history
from src.collectors.forward_archive import append_jsonl_atomic
from src.providers.akshare_pit import normalize_daily_history_snapshot

OUT=Path("output"); OUT.mkdir(exist_ok=True)
MIN_AMOUNT=100_000_000

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


def select_history_universe(snapshot, *, max_symbols=600):
    """Select a bounded, liquidity-ranked deep-history pool from the full spot universe."""
    if max_symbols < 1:
        raise ValueError("max_symbols must be positive")
    required = {"code", "name", "price", "amount", "pct_chg"}
    missing = required - set(snapshot.columns)
    if missing:
        raise ValueError(f"full-market snapshot missing columns: {sorted(missing)}")
    frame = snapshot.copy()
    frame["code"] = frame["code"].astype(str).str.strip().str[-6:].str.zfill(6)
    for column in ("price", "amount", "pct_chg"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["code"].str.fullmatch(r"\d{6}", na=False)
        & (frame["price"] > 0) & (frame["amount"] > 0)
    ].drop_duplicates("code")
    return frame.sort_values(["amount", "code"], ascending=[False, True]).head(max_symbols).reset_index(drop=True)


# [live-scan] Verify Tencent universe fallback with live market data

def _download_tencent_history(code, start, *, cause):
    """Fetch and strictly normalize Tencent daily bars after Eastmoney failure."""
    prefix = "sh" if str(code).startswith("6") else "sz"
    frame = ak.stock_zh_a_hist_tx(
        symbol=prefix + str(code).zfill(6), start_date=start,
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
    if raw is None or len(raw)<25: return None
    cols={"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","成交额":"amount"}
    if any(c not in raw.columns for c in cols): return None
    h=raw.rename(columns=cols).copy()
    for c in ["open","close","high","low","volume","amount"]: h[c]=numeric(h[c])
    h=h.dropna(subset=["open","close","high","low"]).sort_values("date")
    if len(h)<25: return None
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

def scan(u,archive_root=Path("data/forward_archive")):
    rows=[]; errors=[]; start=(datetime.now()-pd.Timedelta(days=160)).strftime("%Y%m%d")
    for _,s in u.iterrows():
        try:
            h=history(s.code,start,archive_root=archive_root)
            if h is None: raise ValueError("insufficient history or schema mismatch")
            # M4 price primitives use the raw bars captured in this run.
            # Retrieval timestamps do not certify historical PIT availability.
            decision_at = datetime.now(timezone.utc)
            m4_bars = h.rename(columns={"date": "trade_date"}).copy()
            m4_bars["symbol"] = s.code
            m4 = price_structure(m4_bars, decision_time=decision_at.isoformat()).iloc[-1]
            available_at = pd.Timestamp(m4.available_time).to_pydatetime()
            liq=pd.notna(m4.amount_ma20) and m4.amount_ma20>=MIN_AMOUNT
            trend=(pd.notna(m4.return_5) and pd.notna(m4.return_20)
                   and m4.return_5>0 and m4.return_20>0)
            structure=pd.notna(m4.CLV) and m4.CLV>=.60
            pullback=pd.notna(m4.Pullback) and m4.Pullback>-.12
            flags=[]
            if not liq: flags.append("20日平均成交额不足1亿元/缺失")
            if m4.cross_session_price_status != "READY":
                flags.append("缺少经公司行为账本验证的点时总收益价格序列；跨日趋势/突破/回撤禁用")
            if not trend: flags.append("5日及20日趋势未同时确认")
            if not structure: flags.append("CLV收盘结构未确认")
            if not pullback: flags.append("回撤过深/缺失")
            rows.append({"symbol":s.code,"trade_date":m4.trade_date.strftime("%Y-%m-%d"),
              "available_time":available_at.isoformat(),
              "BodyStrength":m4.BodyStrength,"return_5":m4.return_5,"return_10":m4.return_10,
              "return_20":m4.return_20,"breakout20":m4.breakout20,
              "upper_shadow_ratio":m4.upper_shadow_ratio,
              "代码":s.code,"名称":s["name"],"最新价":s.price,"涨跌幅%":s.pct_chg,
              "当日成交额":s.amount,"20日平均成交额":m4.amount_ma20,
              "5日收益%":m4.return_5*100 if pd.notna(m4.return_5) else np.nan,
              "20日收益%":m4.return_20*100 if pd.notna(m4.return_20) else np.nan,
              "CLV":m4.CLV,"20日高点回撤%":m4.Pullback*100 if pd.notna(m4.Pullback) else np.nan,
              "M4突破20日高点%":m4.breakout20*100 if pd.notna(m4.breakout20) else np.nan,
              "M4量比5日/20日":m4.volume_ma5_to_ma20,
              "M4距20日高点交易日":m4.pullback_duration,
              "M4高量滞涨":bool(m4.high_volume_stall) if pd.notna(m4.high_volume_stall) else None,
              "M4缩量回撤":bool(m4.low_volume_pullback) if pd.notna(m4.low_volume_pullback) else None,
              "M4数据口径":"原始OHLC仅用于当日K线结构；跨日特征仅接受经公司行为账本验证的点时总收益序列；本次抓取时间戳不具备历史PIT资格",
              "M4跨日价格状态":m4.cross_session_price_status,
              "流动性观察通过":bool(liq),"技术观察通过":bool(trend and structure and pullback),
              "未通过项":"；".join(flags) if flags else "基础观察条件通过",
              "公司行为口径":"无VERIFIED点时总收益序列时，跨日收益/突破/回撤留空；原始价格不替代总收益"})
        except Exception as e: errors.append({"代码":s.code,"原因":str(e)[:220]})
    return pd.DataFrame(rows),pd.DataFrame(errors)

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
            "position_scale": None, "valid_sessions": 0,
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
        "position_scale": result.get("position_scale") if state else None,
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
    for path in files:
        try:
            frame = pd.read_json(path, lines=True, dtype={"symbol": str})
        except (ValueError, OSError) as exc:
            return pd.DataFrame(), {
                "status": "invalid_archive_file",
                "reason": f"{path.name}:{type(exc).__name__}:{str(exc)[:160]}",
                "expected_universe_size": len(set(map(str, expected_symbols))),
                "selected_symbol_count": 0,
                "membership_coverage": 0.0,
                "historical_pit_eligible": False,
            }
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), {
            "status": "no_snapshot",
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
    membership = membership.loc[
        membership["source_id"].astype(str).eq("akshare.stock_sector_detail_sina")
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
        membership, expected_symbols, decision_time=decision_time, max_age="2D"
    )
    return selected, audit


def build_m5_diagnostic_input(watchlist, market_context):
    """Pass only independently READY M3 market-environment scores into M5.

    This supplies one defined A/C factor; it does not make either strategy
    scoreable because all other weighted factors and hard gates remain required.
    """
    state = market_context.get("market_state")
    environment = market_context.get("market_environment_score")
    rows = []
    for code in watchlist["代码"]:
        row = {"symbol": str(code), "market_state": state}
        if environment is not None:
            row["market_environment"] = environment
        rows.append(row)
    return pd.DataFrame(rows, columns=None)


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
    try:
        # Fetch and retain the complete live A-share quote universe. Only the
        # deeper per-symbol history requests are bounded to the most liquid 600.
        u=universe()
        OUT.mkdir(exist_ok=True)
        u.to_csv(OUT/"market_snapshot.csv", index=False, encoding="utf-8-sig")
        history_universe=select_history_universe(u, max_symbols=600)
        w,e=scan(history_universe)
        # [live-scan] verify all-failed history reporting and archive persistence
        w=ensure_watchlist_schema(w)
        if not w.empty:
            w, observation_audit = rank_price_observations(
                w, expected_universe_size=len(history_universe),
                scope_id="top_600_liquidity_history_pool"
            )
        else:
            observation_audit = pd.DataFrame()
    except Exception as exc:
        # Provider outages/rate limits must produce an auditable report, not a false clean scan.
        u=pd.DataFrame()
        history_universe=pd.DataFrame()
        w,e=empty_scan_result(exc)
        observation_audit = pd.DataFrame()
    # Freeze one decision cutoff only after all live stock-history requests finish.
    # M3 and every captured stock bar are therefore evaluated against the same
    # cutoff; no provider response fetched after the cutoff can leak into the run.
    now=datetime.now(timezone.utc)
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
    w=w.sort_values(["技术观察通过","流动性观察通过","当日成交额"],ascending=[False,False,False])
    # M5 remains diagnostic-only. A/C receive the defined M3 market factor
    # only when M3 is independently READY; all other factors and gates remain fail-closed.
    m5_input = build_m5_diagnostic_input(w, m3)
    m5 = evaluate_candidates(m5_input, stage="close") if not m5_input.empty else pd.DataFrame(
        columns=["symbol","strategy","stage","score","eligible","candidate_band","minimum_score","reasons"])
    m5.to_csv(OUT/"m5_diagnostics.csv",index=False,encoding="utf-8-sig")
    m5_blockers = summarize_m5_blockers(m5)
    m5_blockers.to_csv(OUT/"m5_blocker_summary.csv",index=False,encoding="utf-8-sig")

    # A successful research run is not equivalent to a formal FHM-100 signal.
    # Keep data blockers separate from unfinished model integration so an empty
    # universe can never be misreported as "no stocks passed the strategy".
    formal_blockers = []
    if len(u) == 0:
        formal_blockers.append("LIVE_UNIVERSE_UNAVAILABLE")
    if len(w) == 0:
        formal_blockers.append("NO_VALID_STOCK_HISTORY")
    if m3.get("status") != "READY":
        formal_blockers.append("M3_NOT_READY:" + str(m3.get("status")))
    if industry_audit.get("status") != "ready_forward_only":
        formal_blockers.append("FORWARD_INDUSTRY_SNAPSHOT_NOT_READY:" + str(industry_audit.get("status")))
    formal_blockers.extend([
        "M5_FACTOR_PIPELINE_NOT_CONNECTED_TO_LIVE_CANDIDATES",
        "M6_OUT_OF_SAMPLE_PREDICTION_NOT_VALIDATED",
        "M8_PORTFOLIO_RISK_PIPELINE_NOT_CONNECTED",
        "M9_EXECUTION_AWARE_BACKTEST_NOT_ADMITTED",
    ])
    formal_status = "BLOCKED_FORMAL_SIGNAL" if formal_blockers else "READY"
    (OUT/"formal_gate.json").write_text(json.dumps({
        "status": formal_status,
        "blockers": formal_blockers,
        "live_universe_count": int(len(u)),
        "history_scan_universe_count": int(len(history_universe)),
        "valid_history_count": int(len(w)),
        "m3_status": str(m3.get("status")),
        "m5_is_diagnostic_only": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT/"m3_market_context.json").write_text(json.dumps(m3, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    w.to_csv(OUT/"watchlist.csv",index=False,encoding="utf-8-sig")
    e.to_csv(OUT/"data_errors.csv",index=False,encoding="utf-8-sig")
    observation_audit.to_csv(OUT/"m4_observation_audit.csv",index=False,encoding="utf-8-sig")
    c=w[w["技术观察通过"]&w["流动性观察通过"]]
    lines=["# FHM-100 自动研究扫描报告",f"- 运行时间：{now.isoformat()}",
      f"- 全市场实时行情快照：{len(u)}只（含沪深北及科创板；见market_snapshot.csv）",
      f"- 深度历史扫描池：{len(history_universe)}只（按当日成交额排序，最多600只）",
      f"- 历史数据成功：{len(w)}",f"- 数据异常：{len(e)}",
      f"- 行情采集状态：{'正常' if len(u) else '失败/不可用；本次不产生候选'}",
      f"- 基础技术观察名单：{len(c)}",
      f"- M4价格/流动性观察因子已计算：{len(w)}",
      f"- M4观察分位审计记录：{len(observation_audit)}条",
      f"- 个股行情最新可用时间：{latest_stock_available or '无'}；统一决策时点：{now.isoformat()}",
      f"- M3市场状态：{m3['status']}；状态={m3['market_state'] or '未确认'}；有效会话={m3['valid_sessions']}；最新日期={m3['latest_trade_date'] or '无'}",
      f"- M3仓位系数：{m3['position_scale'] if m3['position_scale'] is not None else '不可用'}；原因={m3['reason'] or '无'}",
      f"- 前瞻行业快照：{industry_audit.get('status')}；覆盖={industry_audit.get('selected_symbol_count', 0)}/{industry_audit.get('expected_universe_size', 0)}；仅作当日观察审计，不具备历史PIT资格",
      f"- M5诊断记录：{len(m5)}（每只股票A/B/C各一条；不代表正式评分）",
      f"- 正式模型状态：{formal_status}",
      "## 正式模型阻塞项",
      *[f"- {reason}" for reason in formal_blockers],
      "## M5主要阻塞项",
      (m5_blockers.head(15).to_string(index=False) if len(m5_blockers) else "本次没有股票诊断记录；这不代表模型阻塞项为零。"), "",
      "## 模型状态（重要）",
      "**本次不是正式FHM-100信号。** 行业相对强度、市场/行业滚动Beta残差、完整涨跌停事件、经样本外校准的M6收益预测、M8组合风险及M9回测尚未完成。",
      "M4已新增CLV、BodyStrength、5/10/20日收益、20日突破和上影线的横截面观察分位，并输出m4_observation_audit.csv。分位仅在同一交易日、声明的当前高成交额股票池完整覆盖且因子覆盖率达到95%时生成；不完整则全部留空。它们不是价格结构/成交质量复合分，也未接入M5。M3仅在前瞻归档状态READY时传入M5；行业/残差分位及策略硬门槛仍未全部接入，因此M5仍是诊断而非正式评分。",
      "不输出A/B/C正式评分、P_up、预期收益或买入指令；观察名单不等于达到80分门槛。AI仅作文字点评，不参与评分、不放宽硬门槛。",
      "价格口径：本扫描仅用未复权原始日线；尚无点时公司行为账本，分红送转附近的跨日收益/均线可能失真，相关观察值不得进入正式模型评分或回测标签。",
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
      "history_scan_universe":len(history_universe),"history_ok":len(w),"errors":len(e),"watchlist":len(c),"push":status,
      "formal_status":formal_status,"formal_blockers":formal_blockers},ensure_ascii=False,indent=2),encoding="utf-8")
    print(report); print(status)
    if push_error is not None:
        raise RuntimeError(status) from push_error

if __name__=="__main__":
    try: main()
    except Exception: traceback.print_exc(); raise
