from src.model.catalyst_mapping import map_candidate_sectors


def test_maps_explicit_sector_keywords_but_keeps_direction_unclear():
    candidates, audit = map_candidate_sectors(
        [
            {"title": "半导体设备出口管制", "theme": "semiconductors_ai",
             "formal_stock_score_eligible": False},
            {"title": "人工智能算力基础设施建设"},
            {"title": "宏观经济形势分析"},
        ],
        ["电子", "计算机", "传媒"],
    )
    assert len(candidates) == 2
    assert candidates[0]["mapped_industries"] == ["电子"]
    assert candidates[1]["mapped_industries"] == ["计算机"]
    assert all(row["impact_direction"] == "unclear" for row in candidates)
    assert all(not row["formal_stock_score_eligible"] for row in candidates)
    assert audit["mapped_candidates"] == 2
    assert audit["unmapped_candidates"] == 1
    assert not audit["formal_stock_score_eligible"]


def test_ignores_keywords_without_matching_current_sw_industry():
    rows, audit = map_candidate_sectors(
        [{"title": "光伏储能政策"}],
        ["电子", "计算机"],
    )
    assert rows == []
    assert audit["unmapped_candidates"] == 1


def test_maps_english_global_headlines_to_current_sw_industries():
    rows, audit = map_candidate_sectors(
        [
            {"title": "US weighs new semiconductor export controls"},
            {"title": "New data center investment plan announced"},
            {"title": "Global oil supply disruption risk rises"},
        ],
        ["电子", "计算机", "石油石化"],
    )
    assert [row["mapped_industries"] for row in rows] == [
        ["电子"], ["计算机"], ["石油石化"]
    ]
    assert audit["mapped_candidates"] == 3
    assert all(row["impact_direction"] == "unclear" for row in rows)
    # "export" must never match the transport keyword "port".
    export_only, export_audit = map_candidate_sectors(
        [{"title": "New export controls announced"}], ["交通运输"]
    )
    assert export_only == []
    assert export_audit["unmapped_candidates"] == 1


def test_suppresses_syndicated_duplicate_headlines_before_industry_counts():
    rows, audit = map_candidate_sectors(
        [
            {"candidate_id": "source-a", "title": "半导体设备出口管制"},
            {"candidate_id": "source-b", "title": "半导体设备出口管制！"},
            {"candidate_id": "source-c", "title": "AI infrastructure investment"},
        ],
        ["电子", "计算机"],
    )
    assert len(rows) == 2
    assert audit["duplicate_headlines_suppressed"] == 1
    assert audit["industry_candidate_counts"] == {"电子": 1, "计算机": 1}


def test_freshness_filter_uses_gdelt_seen_clock_and_policy_publication_clock():
    rows, audit = map_candidate_sectors(
        [
            {
                "candidate_id": "stale-global",
                "title": "Semiconductor export controls",
                "gdelt_seen_at": "20260827T123000Z",
                "first_seen_at": "2026-11-15T14:00:00+00:00",
            },
            {
                "candidate_id": "fresh-global",
                "title": "Semiconductor export controls update",
                "gdelt_seen_at": "20261110T123000Z",
                "first_seen_at": "2026-11-15T14:00:00+00:00",
            },
            {
                "candidate_id": "fresh-policy",
                "title": "人工智能算力政策",
                "published_at": "2026-11-12T08:00:00+08:00",
                "first_seen_at": "2026-11-15T14:00:00+00:00",
            },
        ],
        ["电子", "计算机"],
        decision_time="2026-11-15T15:00:00Z",
    )
    assert [row["candidate_id"] for row in rows] == ["fresh-global", "fresh-policy"]
    assert audit["stale_candidates_excluded"] == 1
    assert audit["freshness_window_days"] == 30.0
