"""FHM-100 research runner. No fabricated official scores or M6 probabilities."""
from __future__ import annotations
import json, os, traceback, time
from datetime import datetime, timezone
from pathlib import Path
import akshare as ak
import numpy as np
import pandas as pd
import requests
from src.model.m4_factors import price_structure
from src.model.pipeline import evaluate_candidates
from src.model.m3_assembler import assemble_m3_history

OUT=Path("output"); OUT.mkdir(exist_ok=True)
MIN_AMOUNT=100_000_000

def numeric(s): return pd.to_numeric(s,errors="coerce")

def universe():
    d=ak.stock_zh_a_spot_em()
    required=["代码","名称","最新价","成交额","涨跌幅"]
    if d is None or d.empty or any(c not in d.columns for c in required):
        raise RuntimeError("A-share snapshot empty or schema changed")
    d=d[required].copy(); d.columns=["code","name","price","amount","pct_chg"]
    d["code"]=d.code.astype(str).str.zfill(6)
    for c in ["price","amount","pct_chg"]: d[c]=numeric(d[c])
    d=d[~d.code.str.startswith(("688","689","8","4"))]
    d=d[~d.name.str.contains(r"ST|退",na=False)]
    return d[(d.price>0)&(d.amount>0)].sort_values("amount",ascending=False).head(350)

def _download_history(code, start, max_attempts=3):
    """Retry only transient transport failures; never hide schema/data errors."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    for attempt in range(max_attempts):
        try:
            return ak.stock_zh_a_hist(
                symbol=code, period="daily", start_date=start,
                end_date=datetime.now().strftime("%Y%m%d"), adjust=""
            )  # raw prices only; no look-ahead-adjusted history
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt + 1 >= max_attempts:
                raise
            time.sleep(2 ** attempt)


def history(code,start):
    h=_download_history(code,start)
    if h is None or len(h)<25: return None
    cols={"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","成交额":"amount"}
    if any(c not in h.columns for c in cols): return None
    h=h.rename(columns=cols)
    for c in ["open","close","high","low","volume","amount"]: h[c]=numeric(h[c])
    h=h.dropna(subset=["open","close","high","low"]).sort_values("date")
    return h if len(h)>=25 else None

def features(h):
    x=h.copy(); span=(x.high-x.low).replace(0,np.nan)
    x["ret5"]=x.close.pct_change(5); x["ret20"]=x.close.pct_change(20)
    x["amount_ma20"]=x.amount.rolling(20).mean()
    x["clv"]=(x.close-x.low)/span
    x["upper_shadow"]=(x.high-np.maximum(x.open,x.close))/span
    x["high20_prev"]=x.high.shift(1).rolling(20).max()
    x["breakout20"]=x.close/x.high20_prev-1
    x["pullback"]=x.close/x.high.rolling(20).max()-1
    x["volume_ma5"]=x.volume.rolling(5).mean()
    x["volume_ma20"]=x.volume.rolling(20).mean()
    return x

def scan(u):
    rows=[]; errors=[]; start=(datetime.now()-pd.Timedelta(days=160)).strftime("%Y%m%d")
    for _,s in u.iterrows():
        try:
            h=history(s.code,start)
            if h is None: raise ValueError("insufficient history or schema mismatch")
            # M4 price primitives use the raw bars captured in this run.
            # Retrieval timestamps do not certify historical PIT availability.
            available_at = datetime.now(timezone.utc)
            m4_bars = h.rename(columns={"date": "trade_date"}).copy()
            m4_bars["symbol"] = s.code
            m4_bars["available_time"] = available_at
            m4 = price_structure(m4_bars, decision_time=available_at.isoformat()).iloc[-1]
            z=features(h).iloc[-1]
            liq=pd.notna(m4.amount_ma20) and m4.amount_ma20>=MIN_AMOUNT
            trend=pd.notna(z.ret5) and pd.notna(z.ret20) and z.ret5>0 and z.ret20>0
            structure=pd.notna(z.clv) and z.clv>=.60
            pullback=pd.notna(z.pullback) and z.pullback>-.12
            flags=[]
            if not liq: flags.append("20日平均成交额不足1亿元/缺失")
            if not trend: flags.append("5日及20日趋势未同时确认")
            if not structure: flags.append("CLV收盘结构未确认")
            if not pullback: flags.append("回撤过深/缺失")
            rows.append({"代码":s.code,"名称":s["name"],"最新价":s.price,"涨跌幅%":s.pct_chg,
              "当日成交额":s.amount,"20日平均成交额":m4.amount_ma20,
              "5日收益%":m4.return_5*100 if pd.notna(m4.return_5) else np.nan,
              "20日收益%":m4.return_20*100 if pd.notna(m4.return_20) else np.nan,
              "CLV":m4.CLV,"20日高点回撤%":m4.Pullback*100 if pd.notna(m4.Pullback) else np.nan,
              "M4突破20日高点%":m4.breakout20*100 if pd.notna(m4.breakout20) else np.nan,
              "M4量比5日/20日":m4.volume_ma5_to_ma20,
              "M4距20日高点交易日":m4.pullback_duration,
              "M4高量滞涨":bool(m4.high_volume_stall) if pd.notna(m4.high_volume_stall) else None,
              "M4缩量回撤":bool(m4.low_volume_pullback) if pd.notna(m4.low_volume_pullback) else None,
              "M4数据口径":"原始日线+本次抓取时间戳；仅当前观察，不具备历史PIT资格",
              "流动性观察通过":bool(liq),"技术观察通过":bool(trend and structure and pullback),
              "未通过项":"；".join(flags) if flags else "基础观察条件通过",
              "公司行为口径":"未接入点时公司行为账本；原始价格观察值，不可作为正式收益因子"})
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
            "market_state": None, "position_scale": None, "valid_sessions": 0,
            "latest_trade_date": None, "historical_backtest_eligible": False,
            "quality_warnings": [],
        }
    status = str(result.get("status", "INVALID_DATA"))
    state = result.get("market_state") if status == "READY" else None
    if state not in {"RISK_ON", "NEUTRAL", "RISK_OFF"}:
        state = None
        if status == "READY":
            status = "INVALID_DATA"
    return {
        "status": status, "reason": result.get("reason"), "market_state": state,
        "position_scale": result.get("position_scale") if state else None,
        "valid_sessions": int(result.get("valid_sessions", 0) or 0),
        "latest_trade_date": result.get("latest_trade_date"),
        "historical_backtest_eligible": False,
        "quality_warnings": result.get("quality_warnings", []),
    }

def main():
    now=datetime.now(timezone.utc).astimezone()
    m3 = load_m3_market_context(decision_time=now)
    u=universe(); w,e=scan(u)
    w=w.sort_values(["技术观察通过","流动性观察通过","当日成交额"],ascending=[False,False,False])
    # M5 remains diagnostic-only: M3 state is supplied only when independently READY; factors and hard gates are still incomplete.
    m5_input = pd.DataFrame([{"symbol": str(code), "market_state": m3["market_state"]} for code in w["代码"]])
    m5 = evaluate_candidates(m5_input, stage="close") if not m5_input.empty else pd.DataFrame(
        columns=["symbol","strategy","stage","score","eligible","candidate_band","minimum_score","reasons"])
    m5.to_csv(OUT/"m5_diagnostics.csv",index=False,encoding="utf-8-sig")
    (OUT/"m3_market_context.json").write_text(json.dumps(m3, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    w.to_csv(OUT/"watchlist.csv",index=False,encoding="utf-8-sig")
    e.to_csv(OUT/"data_errors.csv",index=False,encoding="utf-8-sig")
    c=w[w["技术观察通过"]&w["流动性观察通过"]]
    lines=["# FHM-100 自动研究扫描报告",f"- 运行时间：{now.isoformat()}",
      f"- 快照股票池：{len(u)}",f"- 历史数据成功：{len(w)}",f"- 数据异常：{len(e)}",
      f"- 基础技术观察名单：{len(c)}",
      f"- M4价格/流动性观察因子已计算：{len(w)}",
      f"- M3市场状态：{m3['status']}；状态={m3['market_state'] or '未确认'}；有效会话={m3['valid_sessions']}；最新日期={m3['latest_trade_date'] or '无'}",
      f"- M3仓位系数：{m3['position_scale'] if m3['position_scale'] is not None else '不可用'}；原因={m3['reason'] or '无'}",
      f"- M5诊断记录：{len(m5)}（每只股票A/B/C各一条；不代表正式评分）","",
      "## 模型状态（重要）",
      "**本次不是正式FHM-100信号。** 行业相对强度、市场/行业滚动Beta残差、完整涨跌停事件、经样本外校准的M6收益预测、M8组合风险及M9回测尚未完成。",
      "M4价格结构和流动性观察已进入扫描输出；M3现已从前瞻归档读取市场状态，只有状态READY才传入M5，否则保持缺失并拒绝正式评分。M4尚未映射为M5加权因子，行业/残差分位及策略硬门槛也未全部接入，因此M5仍是诊断而非正式评分。详见m3_market_context.json和m5_diagnostics.csv。",
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
      "history_ok":len(w),"errors":len(e),"watchlist":len(c),"push":status},ensure_ascii=False,indent=2),encoding="utf-8")
    print(report); print(status)
    if push_error is not None:
        raise RuntimeError(status) from push_error

if __name__=="__main__":
    try: main()
    except Exception: traceback.print_exc(); raise
