"""Resumable Tencent A-share daily history backfill; fetched rows are not PIT data."""
from __future__ import annotations
import argparse, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import akshare as ak
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data/tencent_history/checkpoint.json"
OUT = ROOT / "output/tencent_history"
MARKETS = {"6":"sh","0":"sz","3":"sz","4":"bj","8":"bj","9":"bj"}
CODE_COLS = ("证券代码","A股代码","股票代码","代码","证券代码 ")
NAME_COLS = ("证券简称","A股简称","股票简称","名称","证券名称")
BAR_COLS = {"date","open","high","low","close","volume","amount"}

def norm_code(raw):
    digits = "".join(c for c in str(raw).strip() if c.isdigit())
    if not digits: return None
    code = digits[-6:].zfill(6)
    return f"{MARKETS[code[0]]}{code}" if code[0] in MARKETS else None

def norm_list(frame, label):
    if frame is None or frame.empty: raise ValueError(f"{label}: empty")
    cc = next((c for c in CODE_COLS if c in frame.columns), None)
    nc = next((c for c in NAME_COLS if c in frame.columns), None)
    if not cc: raise ValueError(f"{label}: no code column")
    out = pd.DataFrame({"raw":frame[cc].astype(str)})
    out["name"] = frame[nc].astype(str) if nc else ""
    out["symbol"] = out.raw.map(norm_code)
    out = out.dropna(subset=["symbol"]).drop_duplicates("symbol")
    if out.empty: raise ValueError(f"{label}: no A-share symbols")
    out["universe_source"] = label
    return out[["symbol","name","universe_source"]]

def universe():
    tasks = [
        ("SSE main",lambda:ak.stock_info_sh_name_code(symbol="主板A股")),
        ("SSE STAR",lambda:ak.stock_info_sh_name_code(symbol="科创板")),
        ("SZSE A",lambda:ak.stock_info_sz_name_code(symbol="A股列表")),
        ("BSE",lambda:ak.stock_info_bj_name_code()),
    ]
    frames, diag = [], []
    for label, fn in tasks:
        for attempt in range(1,4):
            try:
                f=norm_list(fn(),label); frames.append(f)
                diag.append({"source":label,"symbols":len(f),"attempt":attempt}); break
            except Exception as exc:
                if attempt==3: raise RuntimeError(f"Universe {label} failed: {exc}") from exc
                time.sleep(2**(attempt-1))
    u=pd.concat(frames,ignore_index=True).drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    if len(u)<3000: raise RuntimeError(f"Universe too small: {len(u)}")
    digest=hashlib.sha256("\n".join(u.symbol.tolist()).encode()).hexdigest()
    return u,diag,digest

def fetch_bars(symbol,start,end):
    f=ak.stock_zh_a_hist_tx(symbol=symbol,start_date=start,end_date=end,adjust="")
    if f is None or f.empty: raise ValueError("empty response")
    missing=sorted(BAR_COLS-set(f.columns))
    if missing: raise ValueError(f"missing columns: {missing}")
    d=f[list(BAR_COLS)].copy()
    d["date"]=pd.to_datetime(d.date,errors="coerce")
    cols=["open","high","low","close","volume","amount"]
    for c in cols: d[c]=pd.to_numeric(d[c],errors="coerce")
    if d.isna().any().any() or not np.isfinite(d[cols].to_numpy(float)).all(): raise ValueError("null/non-finite OHLCVA")
    if d.date.duplicated().any(): raise ValueError("duplicate dates")
    if not (d[["open","high","low","close"]]>0).all().all(): raise ValueError("non-positive OHLC")
    if (d.high<d[["open","low","close"]].max(axis=1)).any(): raise ValueError("invalid high")
    if (d.low>d[["open","high","close"]].min(axis=1)).any(): raise ValueError("invalid low")
    if (d[["volume","amount"]]<0).any().any(): raise ValueError("negative volume/amount")
    d=d.sort_values("date").reset_index(drop=True)
    d.insert(0,"symbol",symbol); d["source"]="tencent"; d["adjust"]="none"
    return d

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--batch-size",type=int,default=50)
    ap.add_argument("--start-date",default="20100101")
    ap.add_argument("--delay",type=float,default=.25)
    a=ap.parse_args()
    if not 1<=a.batch_size<=200: raise SystemExit("batch-size must be 1..200")
    OUT.mkdir(parents=True,exist_ok=True)
    u,diag,uhash=universe()
    if STATE.exists():
        state=json.loads(STATE.read_text(encoding="utf-8"))
        if state.get("universe_sha256")!=uhash or state.get("universe_size")!=len(u):
            raise RuntimeError("Universe changed; refusing unsafe checkpoint resume")
    else:
        state={"schema_version":1,"start_date":a.start_date,"universe_sha256":uhash,
          "universe_size":len(u),"next_index":0,"batch_number":0,"retry_queue":{},
          "permanent_failures":{},"completed_symbols":0}
    if state.get("start_date")!=a.start_date: raise RuntimeError("start-date differs from checkpoint")
    retry=dict(state.get("retry_queue",{}))
    retry_batch=list(retry)[:a.batch_size]
    new_start=int(state["next_index"])
    new_batch=u.iloc[new_start:new_start+a.batch_size-len(retry_batch)].symbol.tolist()
    state["next_index"]=new_start+len(new_batch)
    work=[(s,True) for s in retry_batch]+[(s,False) for s in new_batch]
    end=datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    fetched=datetime.now(timezone.utc).isoformat()
    frames,outcomes=[],[]
    for i,(symbol,is_retry) in enumerate(work,1):
        prev=retry.get(symbol,{})
        result=error=None
        for attempt in range(1,4):
            try: result=fetch_bars(symbol,a.start_date,end); break
            except Exception as exc:
                error=f"{type(exc).__name__}: {str(exc)[:220]}"
                if attempt<3: time.sleep(min(2**(attempt-1),4))
        if result is not None:
            frames.append(result); retry.pop(symbol,None)
            if not is_retry: state["completed_symbols"]=int(state.get("completed_symbols",0))+1
            item={"symbol":symbol,"status":"ok","rows":len(result),"first_date":result.date.min().strftime("%Y-%m-%d"),"last_date":result.date.max().strftime("%Y-%m-%d"),"retry":is_retry}
        else:
            n=int(prev.get("attempts",0))+1
            if n>=5:
                retry.pop(symbol,None); state.setdefault("permanent_failures",{})[symbol]={"attempts":n,"last_error":error}
            else: retry[symbol]={"attempts":n,"last_error":error}
            item={"symbol":symbol,"status":"failed","error":error,"attempts_across_runs":n,"retry":is_retry}
        outcomes.append(item); print(json.dumps({"progress":f"{i}/{len(work)}",**item},ensure_ascii=False),flush=True)
        time.sleep(max(0,a.delay))
    bn=int(state.get("batch_number",0))
    manifest={"batch_number":bn,"provider":"Tencent/AKShare stock_zh_a_hist_tx","adjust":"none",
      "requested_start":a.start_date,"requested_end":end,"fetched_at_utc":fetched,
      "universe_size":len(u),"universe_sha256":uhash,"universe_sources":diag,
      "cursor_start":new_start,"cursor_end":state["next_index"],"symbols_attempted":len(work),
      "symbols_succeeded":sum(x["status"]=="ok" for x in outcomes),
      "symbols_failed_this_batch":sum(x["status"]=="failed" for x in outcomes),
      "retry_queue_remaining":len(retry),"permanent_failure_count":len(state.get("permanent_failures",{})),
      "historical_pit_eligible":False,
      "pit_note":"Fetched now; no historical public-availability timestamps are supplied.",
      "symbols":outcomes}
    if frames:
        d=pd.concat(frames,ignore_index=True); d["fetched_at_utc"]=fetched
        payload=OUT/f"batch_{bn:05d}.csv.gz"; tmp=payload.with_suffix(".csv.gz.tmp")
        d.to_csv(tmp,index=False,compression="gzip"); tmp.replace(payload)
        manifest.update({"rows":len(d),"payload_file":payload.name,"payload_bytes":payload.stat().st_size,
          "payload_sha256":hashlib.sha256(payload.read_bytes()).hexdigest()})
    else: manifest.update({"rows":0,"payload_file":None})
    atomic_json(OUT/f"batch_{bn:05d}.manifest.json",manifest)
    state.update({"retry_queue":retry,"batch_number":bn+1,"updated_at_utc":datetime.now(timezone.utc).isoformat(),
      "end_date":end,"remaining_unseen_symbols":max(0,len(u)-state["next_index"]),
      "retry_queue_count":len(retry),"permanent_failure_count":len(state.get("permanent_failures",{}))})
    state["coverage_complete"]=state["next_index"]>=len(u) and not retry and not state.get("permanent_failures")
    atomic_json(STATE,state)
    print(json.dumps({"batch":bn,"rows":manifest["rows"],"succeeded":manifest["symbols_succeeded"],
      "failed":manifest["symbols_failed_this_batch"],"remaining_unseen":state["remaining_unseen_symbols"],
      "retry_queue":len(retry),"permanent_failures":state["permanent_failure_count"],
      "coverage_complete":state["coverage_complete"]},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__": main()
