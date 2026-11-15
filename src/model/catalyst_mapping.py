"""Conservative title-to-SW1 sector candidate mapping; never a score or trade signal."""
from __future__ import annotations

import re
from collections import Counter

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
}

def map_candidate_sectors(candidates, available_industries):
    """Return mapped candidate copies and coverage audit; direction stays unclear."""
    names = {str(name).strip() for name in available_industries if str(name).strip()}
    mapped = []
    unmapped = 0
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
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
        "industry_candidate_counts": dict(counts),
        "mapping_is_review_only": True,
        "impact_direction_inferred": False,
        "formal_stock_score_eligible": False,
        "historical_pit_eligible": False,
    }
