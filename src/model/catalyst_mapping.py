"""Conservative title-to-SW1 sector candidate mapping; never a score or trade signal."""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone

# Explicit, reviewable vocabulary. Broad words such as "technology" are omitted.
KEYWORD_TO_INDUSTRY = {
    "半导体": ("电子",),
    "芯片": ("电子",),
    "集成电路": ("电子",),
    "AI芯片": ("电子",),
    "人工智能": ("计算机",),
    "算力": ("计算机",),
    "数据中心": ("计算机",),
    "云计算": ("计算机",),
    "工业软件": ("计算机",),
    "机器人": ("机械设备",),
    "人形机器人": ("机械设备",),
    "工业母机": ("机械设备",),
    "光伏": ("电力设备",),
    "储能": ("电力设备",),
    "新能源": ("电力设备",),
    "风电": ("电力设备",),
    "锂电": ("电力设备",),
    "动力电池": ("电力设备",),
    "智能汽车": ("汽车",),
    "新能源汽车": ("汽车",),
    "汽车": ("汽车",),
    "油气": ("石油石化",),
    "原油": ("石油石化",),
    "天然气": ("石油石化",),
    "LNG": ("石油石化",),
    "能源制裁": ("石油石化",),
    "航运": ("交通运输",),
    "港口": ("交通运输",),
    "影视": ("传媒",),
    "游戏": ("传媒",),
    "出版": ("传媒",),
    "低空经济": ("国防军工", "机械设备"),
    "商业航天": ("国防军工",),
    "semiconductor": ("电子",),
    "chip": ("电子",),
    "integrated circuit": ("电子",),
    "artificial intelligence": ("计算机",),
    "ai infrastructure": ("计算机",),
    "computing power": ("计算机",),
    "data center": ("计算机",),
    "cloud computing": ("计算机",),
    "industrial software": ("计算机",),
    "robotics": ("机械设备",),
    "humanoid robot": ("机械设备",),
    "industrial machinery": ("机械设备",),
    "solar": ("电力设备",),
    "energy storage": ("电力设备",),
    "renewable energy": ("电力设备",),
    "wind power": ("电力设备",),
    "lithium battery": ("电力设备",),
    "electric vehicle": ("汽车",),
    "smart vehicle": ("汽车",),
    "automotive": ("汽车",),
    "crude oil": ("石油石化",),
    "oil supply": ("石油石化",),
    "natural gas": ("石油石化",),
    "lng": ("石油石化",),
    "energy sanctions": ("石油石化",),
    "shipping": ("交通运输",),
    "maritime": ("交通运输",),
    "film": ("传媒",),
    "gaming": ("传媒",),
    "publishing": ("传媒",),
    "commercial aerospace": ("国防军工",),
}

def _parse_aware_time(value):
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo is not None else None
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None


def map_candidate_sectors(candidates, available_industries, *, decision_time=None, max_age_days=30):
    """Map fresh candidates to current SW1 names; direction stays unclear."""
    cutoff = _parse_aware_time(decision_time) if decision_time is not None else None
    if decision_time is not None and cutoff is None:
        raise ValueError("decision_time must be timezone-aware")
    if isinstance(max_age_days, bool) or not isinstance(max_age_days, (int, float)) or max_age_days <= 0:
        raise ValueError("max_age_days must be positive")
    names = {str(name).strip() for name in available_industries if str(name).strip()}
    mapped = []
    unmapped = 0
    stale = 0
    duplicate_headlines = 0
    seen_headlines = set()
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        if cutoff is not None:
            # GDELT seendate is an index-observation clock, not publisher time.
            # Official policy uses its publication clock, then first_seen_at.
            raw_time = (
                candidate.get("gdelt_seen_at")
                if "gdelt_seen_at" in candidate
                else candidate.get("published_at") or candidate.get("first_seen_at")
            )
            observed = _parse_aware_time(raw_time)
            if observed is None or observed > cutoff or (cutoff - observed).total_seconds() > max_age_days * 86400:
                stale += 1
                continue
        headline = re.sub(r"[\W_]+", "", str(candidate.get("title") or "").lower(), flags=re.UNICODE)
        identity = headline or str(candidate.get("candidate_id") or candidate.get("event_id") or "")
        text = " ".join(str(candidate.get(key) or "") for key in (
            "title", "summary", "catalog", "issuing_agency",
            "discovery_keyword", "theme",
        )).lower()
        hits = []
        for keyword, industries in KEYWORD_TO_INDUSTRY.items():
            if keyword.lower() in text:
                for industry in industries:
                    if industry in names and industry not in hits:
                        hits.append(industry)
        if not hits:
            unmapped += 1
            continue
        if identity and identity in seen_headlines:
            duplicate_headlines += 1
            continue
        if identity:
            seen_headlines.add(identity)
        row = dict(candidate)
        row["mapped_industries"] = hits
        row["mapping_status"] = "KEYWORD_MATCH_PENDING_REVIEW"
        row["mapping_basis"] = "explicit_title_or_metadata_keyword_to_current_SW1_name"
        row["impact_direction"] = "unclear"
        row["formal_stock_score_eligible"] = False
        row["historical_pit_eligible"] = False
        mapped.append(row)
    counts = Counter(industry for row in mapped for industry in row["mapped_industries"])
    return mapped, {
        "input_candidates": len(candidates or []),
        "mapped_candidates": len(mapped),
        "unmapped_candidates": unmapped,
        "stale_candidates_excluded": stale,
        "freshness_window_days": float(max_age_days),
        "duplicate_headlines_suppressed": duplicate_headlines,
        "industry_candidate_counts": dict(counts),
        "mapping_is_review_only": True,
        "impact_direction_inferred": False,
        "formal_stock_score_eligible": False,
        "historical_pit_eligible": False,
    }
