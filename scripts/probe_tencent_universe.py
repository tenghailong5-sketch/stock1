"""Qualify Tencent history using exchange-native A-share security masters."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import akshare as ak
import pandas as pd

OUT = Path("output")
OUT.mkdir(exist_ok=True)
PREFIX_MARKET = {"6": "sh", "0": "sz", "3": "sz", "4": "bj", "8": "bj", "9": "bj"}
CODE_COLUMNS = ("证券代码", "A股代码", "股票代码", "代码", "证券代码 ")
NAME_COLUMNS = ("证券简称", "A股简称", "股票简称", "名称", "证券名称")


def normalize_code(code):
    digits = "".join(ch for ch in str(code).strip() if ch.isdigit())
    if not digits:
        return None
    value = digits[-6:].zfill(6)
    market = PREFIX_MARKET.get(value[0])
    return f"{market}{value}" if market else None


def normalize_exchange_frame(frame, source):
    if frame is None or frame.empty:
        raise ValueError(f"{source}: empty_response")
    code_col = next((c for c in CODE_COLUMNS if c in frame.columns), None)
    name_col = next((c for c in NAME_COLUMNS if c in frame.columns), None)
    if not code_col:
        raise ValueError(f"{source}: no_code_column; columns={list(frame.columns)[:15]}")
    out = pd.DataFrame({"raw_code": frame[code_col].astype(str)})
    out["name"] = frame[name_col].astype(str) if name_col else ""
    out["symbol"] = out["raw_code"].map(normalize_code)
    out = out.dropna(subset=["symbol"]).drop_duplicates("symbol")
    if out.empty:
        raise ValueError(f"{source}: no_valid_A_share_codes")
    out["source"] = source
    return out[["symbol", "name", "source"]]


def retry(label, fn, attempts=3):
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            return fn(), {"source": label, "attempt": attempt, "previous_errors": errors}
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {str(exc)[:180]}")
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
    raise RuntimeError(f"{label} failed: {' | '.join(errors)}")


def fetch_exchange_universe():
    # These three interfaces are exchange-specific; deliberately avoid the
    # Eastmoney Push2 endpoints that were rate-limited in the previous run.
    tasks = [
        ("SSE main-board A shares", lambda: ak.stock_info_sh_name_code(symbol="主板A股")),
        ("SSE STAR Market", lambda: ak.stock_info_sh_name_code(symbol="科创板")),
        ("SZSE A shares", lambda: ak.stock_info_sz_name_code(symbol="A股列表")),
        ("BSE listed shares", lambda: ak.stock_info_bj_name_code()),
    ]
    frames, diagnostics = [], []
    for label, fn in tasks:
        raw, meta = retry(label, fn)
        normalized = normalize_exchange_frame(raw, label)
        diagnostics.append({**meta, "rows": int(len(normalized))})
        frames.append(normalized)
    universe = pd.concat(frames, ignore_index=True).drop_duplicates("symbol").sort_values("symbol")
    if len(universe) < 3000:
        raise RuntimeError(f"exchange universe unexpectedly small: {len(universe)}; details={diagnostics}")
    return universe.reset_index(drop=True), diagnostics




def stratified_sample(universe: pd.DataFrame, sample_size: int = 100) -> list[str]:
    """Deterministic proportional sample across SH/SZ/BJ."""
    if sample_size < 3:
        raise ValueError("sample_size must be at least 3")
    groups = {p: universe[universe["symbol"].astype(str).str.startswith(p)].sort_values("symbol").reset_index(drop=True) for p in ("sh", "sz", "bj")}
    if any(g.empty for g in groups.values()):
        raise ValueError("all three exchanges must be represented")
    total = sum(len(g) for g in groups.values())
    quotas = {p: sample_size * len(g) / total for p, g in groups.items()}
    allocation = {p: int(q) for p, q in quotas.items()}
    left = sample_size - sum(allocation.values())
    for p in sorted(groups, key=lambda k: quotas[k] - allocation[k], reverse=True)[:left]:
        allocation[p] += 1
    picks = []
    for p, g in groups.items():
        n = allocation[p]
        positions = sorted(set(round(i * (len(g)-1) / max(n-1, 1)) for i in range(n)))
        picks.extend(g.iloc[positions]["symbol"].astype(str).tolist())
    if len(picks) != sample_size or len(set(picks)) != sample_size:
        raise ValueError("sample allocation did not produce the requested unique count")
    return picks



def fetch_corporate_action_audit(ak_module, symbol: str, *, sleep_fn=time.sleep):
    """Check dividend and rights-issue API coverage; never mark retrieval-time data PIT."""
    today = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    specs = [
        ("dividend", "stock_dividend_cninfo", {"symbol": symbol}),
        ("allotment", "stock_allotment_cninfo", {
            "symbol": symbol, "start_date": "19700101", "end_date": today
        }),
    ]
    statuses, frames = [], []
    for action_type, api_name, kwargs in specs:
        api = getattr(ak_module, api_name, None)
        status = {
            "symbol": symbol, "action_type": action_type,
            "source_id": "akshare." + api_name,
        }
        if not callable(api):
            statuses.append({**status, "status": "unavailable", "row_count": 0,
                             "error": "provider_function_missing", "columns": []})
            continue
        frame, error = None, None
        for attempt in range(1, 4):
            try:
                frame = api(**kwargs)
                if frame is None or not isinstance(frame, pd.DataFrame) or len(frame.columns) == 0:
                    raise ValueError("provider returned no tabular schema")
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:180]}"
                if attempt < 3:
                    sleep_fn(min(2 ** (attempt - 1), 4))
        if error is not None:
            statuses.append({**status, "status": "error", "row_count": 0,
                             "error": error, "columns": []})
            continue
        captured = datetime.now(timezone.utc).isoformat()
        statuses.append({
            **status, "status": "ok_response", "row_count": int(len(frame)),
            "error": None, "columns": [str(col) for col in frame.columns],
            "available_time": captured, "availability_basis": "retrieval_time_only",
            "historical_pit_eligible": False,
        })
        if not frame.empty:
            data = frame.copy()
            data.insert(0, "symbol", symbol)
            data.insert(1, "action_type", action_type)
            data.insert(2, "source_id", "akshare." + api_name)
            data.insert(3, "available_time", captured)
            data["availability_basis"] = "retrieval_time_only"
            data["historical_pit_eligible"] = False
            frames.append(data)
        sleep_fn(0.5)
    return statuses, frames


def attach_row_lineage(frame: pd.DataFrame, symbol: str, fetched_at: str) -> pd.DataFrame:
    """Attach auditable retrieval-time lineage without implying historical PIT."""
    out = frame.copy()
    out.insert(0, "symbol", str(symbol))
    revisions = []
    for row in out[["date", "open", "close", "high", "low", "volume", "amount"]].to_dict(orient="records"):
        canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
        revisions.append(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
    out["available_time"] = fetched_at
    out["source_id"] = "akshare.stock_zh_a_hist_tx"
    out["source_revision"] = revisions
    out["historical_pit_eligible"] = False
    out["availability_basis"] = "retrieval_time_only"
    return out


def main():
    checked = datetime.now(timezone.utc)
    end = checked.date()
    start = end - timedelta(days=220)
    report = {
        "provider": "Tencent via AKShare stock_zh_a_hist_tx",
        "universe_provider": "SSE + SZSE + BSE exchange-specific security lists via AKShare",
        "checked_at_utc": checked.isoformat(),
        "requested_start": start.strftime("%Y%m%d"),
        "requested_end": end.strftime("%Y%m%d"),
        "historical_pit_eligible": False,
        "pit_note": "Historical bars fetched now are not historical point-in-time snapshots.",
    }
    try:
        universe, diagnostics = fetch_exchange_universe()
        report["universe_sources"] = diagnostics
        report["universe"] = {
            "rows": int(len(universe)),
            "exchange_counts": {
                prefix: int(universe["symbol"].str.startswith(prefix).sum())
                for prefix in ("sh", "sz", "bj")
            },
        }
    except Exception as exc:
        report["universe_error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
        report["qualified"] = False
        report["qualification_scope"] = "not qualified: exchange-native symbol universe unavailable"
        (OUT / "tencent_100_symbol_probe.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit("Exchange-native universe acquisition failed")

    # Fixed 100-symbol sample, allocated proportionally across SH/SZ/BJ.
    picks = stratified_sample(universe, sample_size=100)
    report["sample_size"] = len(picks)
    report["sample_method"] = "deterministic_exchange_proportional_stratification"
    report["sample_exchange_counts"] = {p: sum(s.startswith(p) for s in picks) for p in ("sh", "sz", "bj")}

    items, frames, action_frames, action_statuses = [], [], [], []
    for index, symbol in enumerate(picks, 1):
        item = {"symbol": symbol}
        try:
            frame = ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=report["requested_start"],
                end_date=report["requested_end"],
                adjust="",
            )
            fetched_at = datetime.now(timezone.utc).isoformat()
            required = {"date", "open", "close", "high", "low", "volume", "amount"}
            if frame is None or frame.empty:
                raise ValueError("empty_response")
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(f"missing_columns={missing}")
            data = frame[list(required)].copy()
            data["date"] = pd.to_datetime(data["date"], errors="coerce")
            for col in ("open", "close", "high", "low", "volume", "amount"):
                data[col] = pd.to_numeric(data[col], errors="coerce")
            if data.isna().any().any():
                raise ValueError("null_or_non_numeric_values")
            if data["date"].duplicated().any():
                raise ValueError("duplicate_dates")
            if not (data[["open", "close", "high", "low"]] > 0).all().all():
                raise ValueError("non_positive_ohlc")
            if (data["high"] < data[["open", "close", "low"]].max(axis=1)).any():
                raise ValueError("invalid_high")
            if (data["low"] > data[["open", "close", "high"]].min(axis=1)).any():
                raise ValueError("invalid_low")
            if (data[["volume", "amount"]] < 0).any().any():
                raise ValueError("negative_volume_or_amount")
            data = data.sort_values("date").reset_index(drop=True)
            data = attach_row_lineage(data, symbol, fetched_at)
            frames.append(data)
            item.update({
                "status": "OK",
                "available_time": fetched_at,
                "source_id": "akshare.stock_zh_a_hist_tx",
                "historical_pit_eligible": False,
                "rows": int(len(data)),
                "first_date": data["date"].min().strftime("%Y-%m-%d"),
                "last_date": data["date"].max().strftime("%Y-%m-%d"),
            })
        except Exception as exc:
            item.update({"status": "ERROR", "error": f"{type(exc).__name__}: {str(exc)[:220]}"})
        items.append(item)
        statuses, captured_actions = fetch_corporate_action_audit(ak, symbol)
        action_statuses.extend(statuses)
        action_frames.extend(captured_actions)
        print(json.dumps({
            "progress": f"{index}/{len(picks)}", **item,
            "corporate_actions": statuses,
        }, ensure_ascii=False), flush=True)
        time.sleep(0.3)

    report["items"] = items
    report["corporate_action_audit"] = {
        "symbols_attempted": len(picks),
        "dividend_symbols_ok": sum(x["status"] == "ok_response" for x in action_statuses if x["action_type"] == "dividend"),
        "allotment_symbols_ok": sum(x["status"] == "ok_response" for x in action_statuses if x["action_type"] == "allotment"),
        "dividend_event_rows": sum(x["row_count"] for x in action_statuses if x["action_type"] == "dividend" and x["status"] == "ok_response"),
        "allotment_event_rows": sum(x["row_count"] for x in action_statuses if x["action_type"] == "allotment" and x["status"] == "ok_response"),
        "statuses": action_statuses,
        "historical_pit_eligible": False,
        "note": "Provider coverage reconnaissance only. Retrieval-time rows do not establish historical publication availability or a verified total-return ledger.",
    }
    if action_frames:
        action_data = pd.concat(action_frames, ignore_index=True, sort=False)
        action_path = OUT / "tencent_100_symbol_corporate_actions.csv"
        action_data.to_csv(action_path, index=False)
        report["corporate_action_audit"]["artifact_file"] = str(action_path)
        report["corporate_action_audit"]["artifact_bytes"] = action_path.stat().st_size
        report["corporate_action_audit"]["artifact_sha256"] = hashlib.sha256(action_path.read_bytes()).hexdigest()
    report["passed_symbols"] = sum(x["status"] == "OK" for x in items)
    report["failed_symbols"] = [x for x in items if x["status"] != "OK"]
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        counts = combined.groupby("symbol")["date"].nunique()
        report["total_rows"] = int(len(combined))
        report["minimum_sessions"] = int(counts.min())
        report["maximum_sessions"] = int(counts.max())
        report["symbols_with_at_least_119_sessions"] = int((counts >= 119).sum())
        path = OUT / "tencent_100_symbol_sample.csv"
        combined.to_csv(path, index=False)
        report["artifact_file"] = str(path)
        report["artifact_bytes"] = path.stat().st_size
        report["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["corporate_action_audit"]["qualified_response_coverage"] = (
        report["corporate_action_audit"]["dividend_symbols_ok"] >= 95
        and report["corporate_action_audit"]["allotment_symbols_ok"] >= 95
    )
    report["qualified"] = (
        len(picks) == 100
        and report["passed_symbols"] >= 95
        and report.get("symbols_with_at_least_119_sessions", 0) >= 95
        and report["corporate_action_audit"]["qualified_response_coverage"]
    )
    report["qualification_scope"] = (
        "100-symbol stratified provider qualification only; corporate-action response coverage is not a verified adjustment ledger; "
        "full-market history and historical PIT remain unqualified"
    )
    (OUT / "tencent_60_symbol_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["qualified"]:
        raise SystemExit("Tencent multi-symbol history qualification failed")


if __name__ == "__main__":
    main()
