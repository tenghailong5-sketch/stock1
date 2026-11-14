"""Forward-only discovery of new central-government policy documents.

Search results are candidates only. This adapter never infers industry mapping,
impact direction, publication time, or positive catalyst status from a title.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests


API_URL = "https://sousuo.www.gov.cn/search-gov/data"
POLICY_LIBRARY_URL = "https://sousuo.www.gov.cn/zcwjk/policyDocumentLibrary"
SOURCE_ID = "sousuo.www.gov.cn"
DEFAULT_KEYWORDS = (
    "人工智能", "半导体", "机器人", "新能源", "数据要素",
    "低空经济", "算力", "智能汽车", "工业软件",
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FHM100PolicyResearch/1.0)",
    "Referer": POLICY_LIBRARY_URL,
    "Accept": "application/json, text/plain, */*",
}


def _clean_text(value) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _published_at(row: dict) -> str | None:
    """Parse only an explicit publisher date/timestamp; never substitute crawl time."""
    value = row.get("pubtime") or row.get("ptime") or row.get("publishTime") or row.get("publish_time")
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)) or str(value).strip().isdigit():
            number = float(value)
            unit = "ms" if number > 10_000_000_000 else "s"
            stamp = pd.to_datetime(number, unit=unit, utc=True, errors="coerce")
        else:
            stamp = pd.to_datetime(str(value).strip(), utc=True, errors="coerce")
        if pd.isna(stamp):
            return None
        return stamp.isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _result_rows(payload: dict) -> list[dict]:
    if not isinstance(payload, dict) or payload.get("code") not in (None, 200, "200"):
        raise ValueError("government policy API returned an invalid response envelope")
    search = payload.get("searchVO") or payload.get("searchVo") or {}
    cat_map = search.get("catMap") or search.get("catmap") or {}
    rows = []
    if isinstance(cat_map, dict):
        for category in cat_map.values():
            if not isinstance(category, dict):
                continue
            items = category.get("listVO") or category.get("listvo") or []
            if isinstance(items, list):
                rows.extend(item for item in items if isinstance(item, dict))
    if not rows:
        legacy = search.get("listVO") or search.get("listvo") or payload.get("results") or []
        if isinstance(legacy, list):
            rows.extend(item for item in legacy if isinstance(item, dict))
    return rows


def _canonical_source_url(raw_url: str) -> str | None:
    value = str(raw_url or "").strip()
    if not value:
        return None
    url = urljoin("https://www.gov.cn", value)
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host != "gov.cn" and not host.endswith(".gov.cn"):
        return None
    # The official search index still emits legacy http:// links. Upgrade only
    # verified government hosts; never forward plaintext HTTP source URLs.
    return parsed._replace(scheme="https").geturl()


def _candidate(row: dict, *, keyword: str, first_seen_at: str) -> dict | None:
    title = _clean_text(row.get("title") or row.get("TITLE"))
    source_url = _canonical_source_url(row.get("url") or row.get("URL") or row.get("link"))
    if not title or not source_url:
        return None
    digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:24]
    parsed = urlparse(source_url)
    return {
        "candidate_id": "gov_policy_" + digest,
        "title": title,
        "source_url": source_url,
        "source_id": parsed.hostname,
        "catalog": _clean_text(row.get("catalog") or row.get("category") or row.get("type")),
        "document_number": _clean_text(row.get("docno") or row.get("docNo") or row.get("pcode")),
        "issuing_agency": _clean_text(row.get("puborg") or row.get("pubOrg") or row.get("author")),
        "published_at": _published_at(row),
        "first_seen_at": first_seen_at,
        "available_at": first_seen_at,
        "discovery_source": SOURCE_ID,
        "discovery_keyword": keyword,
        "review_status": "PENDING_REVIEW",
        "impact_direction": "unclear",
        "industry_codes": [],
        "historical_pit_eligible": False,
        "formal_stock_score_eligible": False,
    }


def discover_gov_policy_candidates(
    *, session=None, keywords=DEFAULT_KEYWORDS, clock=None,
    timeout: int = 15, pause_seconds: float = 0.25, sleep_fn=time.sleep,
    per_keyword_limit: int = 20,
) -> tuple[list[dict], dict]:
    """Search a small current-policy window; return auditable unclassified candidates."""
    if timeout <= 0 or per_keyword_limit < 1:
        raise ValueError("timeout and per_keyword_limit must be positive")
    client = session or requests.Session()
    now = clock or (lambda: datetime.now(timezone.utc))
    captured = pd.Timestamp(now())
    if captured.tzinfo is None:
        raise ValueError("policy discovery clock must be timezone-aware")
    first_seen_at = captured.tz_convert("UTC").isoformat()
    candidates = {}
    failures = []
    successful_queries = 0
    for index, keyword in enumerate(tuple(keywords)):
        try:
            response = client.get(
                API_URL,
                params={
                    "t": "zhengcelibrary_gw_bm_gb",
                    "q": keyword,
                    "searchfield": "title:content:summary",
                    "sort": "pubtime",
                    "sortType": 1,
                    "p": 1,
                    "n": per_keyword_limit,
                    "inpro": "",
                    "childtype": "",
                    "subchildtype": "",
                },
                headers=HEADERS,
                timeout=timeout,
            )
            response.raise_for_status()
            rows = _result_rows(response.json())
            successful_queries += 1
            for row in rows:
                item = _candidate(row, keyword=keyword, first_seen_at=first_seen_at)
                if item:
                    candidates.setdefault(item["candidate_id"], item)
        except Exception as exc:
            failures.append({
                "keyword": keyword,
                "error": f"{type(exc).__name__}: {str(exc)[:180]}",
            })
        if pause_seconds > 0 and index + 1 < len(tuple(keywords)):
            sleep_fn(pause_seconds)
    audit = {
        "status": "READY_CANDIDATES_ONLY" if successful_queries == len(tuple(keywords))
                  else "PARTIAL" if successful_queries else "UNAVAILABLE",
        "source": SOURCE_ID,
        "api_url": API_URL,
        "keywords_requested": len(tuple(keywords)),
        "queries_succeeded": successful_queries,
        "queries_failed": len(failures),
        "candidate_count": len(candidates),
        "failed_queries": failures,
        "candidate_only": True,
        "industry_mapping_performed": False,
        "impact_direction_inferred": False,
        "historical_pit_eligible": False,
        "formal_stock_score_eligible": False,
    }
    return sorted(candidates.values(), key=lambda item: (item["published_at"] or "", item["candidate_id"]), reverse=True), audit


def append_policy_candidates(path: str | Path, candidates: list[dict]) -> int:
    """Append only unseen candidate IDs; preserve the original first-seen record."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
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
