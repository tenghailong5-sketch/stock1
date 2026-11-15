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
