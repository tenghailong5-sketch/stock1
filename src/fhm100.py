"""FHM-100 research runner.

This implementation deliberately does NOT fabricate official strategy scores or M6
probabilities when required point-in-time industry/event/training data is absent.
It produces a transparent data-quality report and a non-trading watchlist only.
"""
from __future__ import annotations
import json, os, sys, traceback
from datetime import datetime, timezone
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
import requests

OUT = Path("output")
OUT.mkdir(exist_ok=True)
EXCLUDED_PREFIX = ("688", "689", "8", "4")
MIN_AMOUNT = 100_000_000  # CNY; FHM-100 A strategy initial filter


def num(s):
    return pd.to_numeric(s, errors="coerce")


def get_universe():
    spot = ak.stock_zh_a_spot_em()
    if spot is None or spot.empty:
        raise RuntimeError("AkShare returned an empty A-share snapshot")
    code_col = "代码"
    required = [code_col, "名称", "最新价", "成交额", "涨跌幅"]
    missing = [c for c in required if c not in spot.columns]
    if missing:
        raise RuntimeError(f"Spot schema changed; missing columns: {missing}")
    df = spot[required].copy()
    df.columns = ["code", "name", "price", "amount", "pct_chg"]
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["price"], df["amount"], df["pct_chg"] = map(num, [df.price, df.amount, df.pct_chg])
    df = df[~df.code.str.startswith(EXCLUDED_PREFIX)]
    df = df[~df.name.str.contains(r"ST|退", na=False)]
    df = df[(df.price > 0) & (df.amount > 0)]
    return df.sort_values("amount", ascending=False).head(350).reset_index(drop=True)


def get_history(code, start):
    h = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start,
                          end_date=datetime.now().strftime("%Y%m%d"),
                          adjust="qfq")
    if h is None or len(h) < 25:
        return None
    cols = {"日期":"date","开盘":"open","收盘":"close","最高":"high",
            "最低":"low","成交量":"volume","成交额":"amount","涨跌幅":"pct_chg"}
    h = h.rename(columns=cols)
    needed = list(cols.values())
    if any(c not in h.columns for c in needed):
        return None
    for c in needed[1:]:
        h[c] = num(h[c])
    return h.dropna(subset=["close","high","low","open"]).sort_values("date")


def inspect_history(h):
    x = h.copy()
    close, high, low = x.close, x.high, x.low
    prev = close.shift(1)
    x["ret1"] = close.pct_change()
    x["ret5"] = close.pct_change(5)
    x["ret20"] = close.pct_change(20)
    x["amount_ma20"] = x.amount.rolling(20).mean()
    x["volume_ma5"] = x.volume.rolling(5).mean()
    x["volume_ma20"] = x.volume.rolling(20).mean()
    span = (high-low).replace(0, np.nan)
    x["clv"] = (close-low)/span
    x["body_strength"] = (close-x.open)/span
    x["upper_shadow"] = (high-np.maximum(x.open,close))/span
    x["high20_prev"] = high.shift(1).rolling(20).max()
    x["breakout20"] = close/x.high20_prev-1
    x["pullback"] = close/high.rolling(20).max()-1
    x["prev_close"] = prev
    return x


def build_watchlist(universe):
    rows, errors = [], []
    start = (datetime.now() - pd.Timedelta(days=160)).strftime("%Y%m%d")
    for _, s in universe.iterrows():
        code = s.code
        try:
            h = get_history(code, start)
            if h is None:
                errors.append({"code":code,"reason":"insufficient_history_or_schema"})
                continue
            x = inspect_history(h).iloc[-1]
            amount20 = x.get("amount_ma20")
            # These are observable technical diagnostics, NOT FHM-100 scores.
            eligible_liquidity = pd.notna(amount20) and amount20 >= MIN_AMOUNT
            trend = pd.notna(x.get("ret5")) and pd.notna(x.get("ret20")) and x.ret5 > 0 and x.ret20 > 0
            structure = pd.notna(x.get("clv")) and x.clv >= 0.60
            not_extended = pd.notna(x.get("pullback")) and x.pullback > -0.12
            flags = []
            if not eligible_liquidity: flags.append("20日平均成交额不足1亿元或缺失")
            if not trend: flags.append("5/20日趋势未同时确认")
            if not structure: flags.append("收盘位置未确认")
            if not not_extended: flags.append("回撤过深或数据缺失")
            rows.append({
                "代码":code,"名称":s["name"],"最新价":s.price,"当日涨跌幅%":s.pct_chg,
                "当日成交额":s.amount,"20日平均成交额":amount20,
                "5日收益%":x.get("ret5",np.nan)*100 if pd.notna(x.get("ret5")) else np.nan,
                "20日收益%":x.get("ret20",np.nan)*100 if pd.notna(x.get("ret20")) else np.nan,
                "CLV":x.get("clv"),"20日高点回撤%":x.get("pullback",np.nan)*100 if pd.notna(x.get("pullback")) else np.nan,
                "流动性观察通过":bool(eligible_liquidity),"技术观察通过":bool(trend and structure and not_extended),
                "未通过项":"；".join(flags) if flags else "基础技术观察条件通过"
            })
        except Exception as e:
            errors.append({"code":code,"reason":str(e)[:240]})
    return pd.DataFrame(rows), pd.DataFrame(errors)


def ai_comment(report_text):
    key = os.getenv("AI_API_KEY")
    if not key:
        return "AI点评未启用：未配置 AI_API_KEY。"
    base = os.getenv("AI_API_BASE_URL","https://api.openai.com/v1").rstrip("/")
    model = os.getenv("AI_MODEL","gpt-4o-mini")
    prompt = ("你是A股研究助理。只根据输入报告做风险点评，不得编造数据、正式FHM评分、"
              "收益概率或买卖指令。明确数据缺口。\n\n"+report_text[:12000])
    try:
        r = requests.post(base+"/chat/completions",
            headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},
            json={"model":model,"messages":[{"role":"user","content":prompt}],"temperature":0.2},
            timeout=90)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return "AI点评调用失败："+str(e)[:300]


def pushplus(title, content):
    token=os.getenv("PUSHPLUS_TOKEN")
    if not token: return "PushPlus未发送：未配置 PUSHPLUS_TOKEN。"
    r=requests.post("https://www.pushplus.plus/send",
        json={"token":token,"title":title,"content":content,"template":"markdown"},timeout=30)
    r.raise_for_status()
    data=r.json()
    return f"PushPlus返回：{data.get('msg','unknown')}（code={data.get('code','?')}）"


def main():
    now=datetime.now(timezone.utc).astimezone()
    universe=get_universe()
    watch, errors=build_watchlist(universe)
    watch=watch.sort_values(["技术观察通过","流动性观察通过","当日成交额"],ascending=[False,False,False])
    watch.to_csv(OUT/"watchlist.csv",index=False,encoding="utf-8-sig")
    errors.to_csv(OUT/"data_errors.csv",index=False,encoding="utf-8-sig")
    candidates=watch[watch["技术观察通过"] & watch["流动性观察通过"]]
    lines=[
      "# FHM-100 自动研究扫描报告",
      f"- 运行时间：{now.isoformat()}",
      f"- 行情快照股票池：{len(universe)}",
      f"- 成功取得历史数据：{len(watch)}",
      f"- 历史数据失败：{len(errors)}",
      f"- 基础技术观察名单：{len(candidates)}",
      "",
      "## 重要状态",
      "- **本次不是正式FHM-100信号**：行业相对强度、个股残差Beta、完整B策略事件字段、校准后的M6预测及M8组合风险数据尚未接入。",
      "- 不输出A/B/C正式评分、P_up、预期收益或买入指令；不得将技术观察名单视为达到80分门槛。",
      "- AI仅做文字风险点评，不参与评分、不放宽硬门槛。",
      "",
      "## 技术观察名单（非买入信号）",
      candidates.head(30).to_markdown(index=False) if not candidates.empty else "暂无满足基础技术观察条件的股票。",
      "",
      "## 数据异常",
      errors.head(30).to_markdown(index=False) if not errors.empty else "无历史数据异常记录。"
    ]
    report="\n".join(lines)
    ai=ai_comment(report)
    report += "\n\n## AI辅助风险点评\n\n"+ai+"\n"
    (OUT/"report.md").write_text(report,encoding="utf-8")
    push_status=pushplus("FHM-100研究扫描报告",report[:18000])
    (OUT/"run_status.json").write_text(json.dumps({
      "run_time":now.isoformat(),"universe":len(universe),"history_ok":len(watch),
      "history_errors":len(errors),"watchlist":len(candidates),"pushplus":push_status
    },ensure_ascii=False,indent=2),encoding="utf-8")
    print(report)
    print(push_status)


if __name__=="__main__":
    try: main()
    except Exception:
        traceback.print_exc()
        raise
