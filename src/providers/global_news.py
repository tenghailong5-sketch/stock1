"""Bounded discovery of international macro/geopolitical headlines.

GDELT is a discovery feed, not an authoritative event classifier. Records remain
candidate-only until publisher timestamps, industry mapping and impact are reviewed.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY_THEMES = {
    "trade_policy": '(tariffs OR trade restrictions OR export controls OR sanctions)',
    "semiconductors_ai": '(semiconductor OR chip export OR artificial intelligence regulation)',
    "energy_geopolitics": '(oil supply OR LNG OR energy sanctions OR shipping disruption)',
    "global_rates": '(Federal Reserve OR interest rates OR central bank policy)',
}


def discover_global_news_candidates(*, session=None, clock=None, timeout=12,
                                    per_theme_limit=20):
    if timeout <= 0 or per_theme_limit < 1:
        raise ValueError("timeout and per_theme_limit must be positive")
    client = session or requests.Session()
    now = clock or (lambda: datetime.now(timezone.utc))
    captured = pd.Timestamp(now())
    if captured.tzinfo is None:
        raise ValueError("news discovery clock must be timezone-aware")
    first_seen_at = captured.tz_convert("UTC").isoformat()
    candidates = {}
    failures = []
    succeeded = 0
    for theme_index, (theme, query) in enumerate(QUERY_THEMES.items()):
        try:
            # GDELT rate-limits bursts. Keep topic requests sequential with a
            # small pause and one bounded retry for HTTP 429.
            if theme_index:
                time.sleep(1.0)
            response = None
            for attempt in range(2):
                response = client.get(
                    API_URL,
                    params={
                        "query": query,
                        "mode": "ArtList",
                        "format": "json",
                        "maxrecords": per_theme_limit,
                        "sort": "HybridRel",
                    },
                    timeout=timeout,
                )
                if getattr(response, "status_code", None) == 429 and attempt == 0:
                    time.sleep(2.0)
                    continue
                break
            response.raise_for_status()
            payload = response.json()
            articles = payload.get("articles", []) if isinstance(payload, dict) else []
            if not isinstance(articles, list):
                raise ValueError("GDELT articles field is not a list")
            succeeded += 1
            for article in articles:
                if not isinstance(article, dict):
                    continue
                title = " ".join(str(article.get("title") or "").split())
                url = str(article.get("url") or "").strip()
                if not title or not url or urlparse(url).scheme != "https":
                    continue
                digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
                domain = str(article.get("domain") or urlparse(url).hostname or "")
                candidates.setdefault(digest, {
                    "candidate_id": "global_news_" + digest,
                    "title": title,
                    "source_url": url,
                    "source_id": domain,
                    "theme": theme,
                    "language": str(article.get("language") or "unknown"),
                    "source_country": str(article.get("sourcecountry") or "unknown"),
                    "gdelt_seen_at": str(article.get("seendate") or ""),
                    "first_seen_at": first_seen_at,
                    "available_at": first_seen_at,
                    "timestamp_basis": "retrieval_time; GDELT seendate is not publisher publication time",
                    "review_status": "PENDING_REVIEW",
                    "impact_direction": "unclear",
                    "industry_mapping": [],
                    "historical_pit_eligible": False,
                    "formal_stock_score_eligible": False,
                })
        except Exception as exc:
            failures.append({
                "theme": theme,
                "error": f"{type(exc).__name__}: {str(exc)[:180]}",
            })
    audit = {
        "status": "READY_CANDIDATES_ONLY" if succeeded == len(QUERY_THEMES)
                  else "PARTIAL" if succeeded else "UNAVAILABLE",
        "source": "GDELT DOC API",
        "queries_requested": len(QUERY_THEMES),
        "queries_succeeded": succeeded,
        "queries_failed": len(failures),
        "candidate_count": len(candidates),
        "theme_counts": pd.Series([x["theme"] for x in candidates.values()]).value_counts().to_dict(),
        "failed_queries": failures,
        "publisher_publication_time_verified": False,
        "industry_mapping_performed": False,
        "impact_direction_inferred": False,
        "historical_pit_eligible": False,
        "formal_stock_score_eligible": False,
    }
    return sorted(candidates.values(), key=lambda x: (x["first_seen_at"], x["candidate_id"])), audit


def append_global_news_candidates(path, candidates):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("candidate_id"):
                    existing.add(row["candidate_id"])
    added = 0
    with target.open("a", encoding="utf-8") as handle:
        for row in candidates:
            if row["candidate_id"] in existing:
                continue
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            existing.add(row["candidate_id"])
            added += 1
    return added
