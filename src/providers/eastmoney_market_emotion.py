"""Historical Eastmoney market-emotion pool enrichment.

AkShare/Eastmoney only exposes these pool endpoints for a recent rolling window.
Rows are retrieval-time observations, never historical point-in-time inputs.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Callable, Iterable

import pandas as pd


def _valid_pool(frame, required: set[str]) -> bool:
    return isinstance(frame, pd.DataFrame) and required.issubset(set(frame.columns))


def fetch_market_emotion_pool_history(
    trade_dates: Iterable,
    *,
    limit_up_fetcher: Callable,
    broken_board_fetcher: Callable,
    strong_pool_fetcher: Callable,
    clock: Callable | None = None,
) -> pd.DataFrame:
    """Build daily limit-streak, board-failure, promotion and strong-pool metrics.

    Promotion rate is defined as the share of previous-session first-board
    stocks that appear in today's limit-up pool with a streak of at least two.
    Missing/incomplete pools stay missing; no zero-fill is allowed.
    """
    now = clock or (lambda: datetime.now(timezone.utc))
    dates = sorted(pd.to_datetime(list(trade_dates), errors="coerce").dropna().normalize().unique())
    rows = []
    previous_limit_up = None
    for day in dates:
        ymd = pd.Timestamp(day).strftime("%Y%m%d")
        errors = []
        try:
            limit_up = limit_up_fetcher(date=ymd)
        except Exception as exc:
            limit_up = None
            errors.append("limit_up:" + type(exc).__name__)
        try:
            broken = broken_board_fetcher(date=ymd)
        except Exception as exc:
            broken = None
            errors.append("broken_board:" + type(exc).__name__)
        try:
            strong = strong_pool_fetcher(date=ymd)
        except Exception as exc:
            strong = None
            errors.append("strong_pool:" + type(exc).__name__)

        limit_ok = _valid_pool(limit_up, {"代码", "连板数"})
        broken_ok = _valid_pool(broken, {"代码"})
        strong_ok = _valid_pool(strong, {"代码"})
        ready = limit_ok and broken_ok and strong_ok
        highest_streak = board_failure_rate = promotion_rate = strong_count = None
        limit_count = broken_count = None
        promotion_numerator = promotion_denominator = None
        if limit_ok:
            limit_count = int(len(limit_up))
            streaks = pd.to_numeric(limit_up["连板数"], errors="coerce").dropna()
            highest_streak = float(streaks.max()) if len(streaks) else 0.0
        if broken_ok:
            broken_count = int(len(broken))
        if limit_ok and broken_ok:
            denominator = int(len(limit_up) + len(broken))
            if denominator > 0:
                board_failure_rate = float(len(broken) / denominator)
        if strong_ok:
            strong_count = int(len(strong))
        if limit_ok and previous_limit_up is not None:
            previous_codes = previous_limit_up["代码"].astype(str).str.zfill(6)
            previous_streaks = pd.to_numeric(
                previous_limit_up["连板数"], errors="coerce"
            )
            first_board = set(previous_codes[previous_streaks.eq(1)])
            today_codes = set(limit_up["代码"].astype(str).str.zfill(6))
            today_streaks = pd.to_numeric(limit_up["连板数"], errors="coerce")
            today_second_plus = set(
                limit_up.loc[today_streaks.ge(2), "代码"].astype(str).str.zfill(6)
            )
            promotion_denominator = int(len(first_board))
            promotion_numerator = int(len(first_board & today_codes & today_second_plus))
            if promotion_denominator > 0:
                promotion_rate = float(promotion_numerator / promotion_denominator)

        captured = pd.Timestamp(now())
        if captured.tzinfo is None:
            raise ValueError("market-emotion pool retrieval clock must be timezone-aware")
        captured = captured.tz_convert("UTC")
        revision_payload = {
            "trade_date": ymd,
            "limit_up": (
                limit_up[["代码", "连板数"]].fillna("").astype(str).to_dict("records")
                if limit_ok else None
            ),
            "broken": (
                broken[["代码"]].fillna("").astype(str).to_dict("records")
                if broken_ok else None
            ),
            "strong": (
                strong[["代码"]].fillna("").astype(str).to_dict("records")
                if strong_ok else None
            ),
        }
        revision = hashlib.sha256(
            json.dumps(revision_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        rows.append({
            "trade_date": pd.Timestamp(day).normalize(),
            "highest_observed_limit_up_streak": highest_streak,
            "strong_pool_count": strong_count,
            "board_failure_rate": board_failure_rate,
            "promotion_rate": promotion_rate,
            "limit_up_pool_count": limit_count,
            "broken_board_pool_count": broken_count,
            "promotion_numerator": promotion_numerator,
            "promotion_denominator": promotion_denominator,
            "pool_history_status": "READY" if ready else "INSUFFICIENT",
            "pool_history_errors": "|".join(errors),
            "sentiment_context_source_id": "akshare.eastmoney.limit_up_broken_strong_pools",
            "sentiment_context_available_time": captured,
            "available_time": captured,
            "availability_basis": "retrieval_time_only",
            "historical_pit_eligible": False,
            "source_revision": revision,
        })
        if limit_ok:
            previous_limit_up = limit_up.copy()
        else:
            previous_limit_up = None
    return pd.DataFrame(rows)
