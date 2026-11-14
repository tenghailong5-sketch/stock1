import json

from src.providers.gov_policy import (
    append_policy_candidates,
    discover_gov_policy_candidates,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, *, params, headers, timeout):
        self.calls.append({
            "url": url, "params": params, "headers": headers, "timeout": timeout
        })
        return FakeResponse(self.payload)


def test_discovery_normalizes_official_policy_candidates_without_guessing():
    session = FakeSession({
        "code": 200,
        "searchVO": {
            "catMap": {
                "department": {
                    "listVO": [
                        {
                            "title": "<em>人工智能</em>产业政策通知",
                            "url": "/zhengce/content/202611/content_123.htm",
                            "pubtime": 1794000000,
                            "docno": "国发〔2026〕1号",
                        },
                        {
                            "title": "无发布时间文件",
                            "url": "https://www.miit.gov.cn/zwgk/test.html",
                        },
                        {
                            "title": "Legacy HTTP official link",
                            "url": "http://www.gov.cn/zhengce/legacy.html",
                        },
                        {
                            "title": "Reject external URL",
                            "url": "https://example.com/policy",
                        },
                    ]
                }
            }
        },
    })
    candidates, audit = discover_gov_policy_candidates(
        session=session,
        keywords=("人工智能",),
        clock=lambda: "2026-11-14T00:00:00Z",
        pause_seconds=0,
    )

    assert audit["status"] == "READY_CANDIDATES_ONLY"
    assert audit["candidate_count"] == 3
    assert len(session.calls) == 1
    assert session.calls[0]["params"]["q"] == "人工智能"
    assert all(item["review_status"] == "PENDING_REVIEW" for item in candidates)
    assert all(item["impact_direction"] == "unclear" for item in candidates)
    assert all(item["industry_codes"] == [] for item in candidates)
    assert all(item["available_at"] == "2026-11-14T00:00:00+00:00" for item in candidates)
    no_date = next(item for item in candidates if item["title"] == "无发布时间文件")
    assert no_date["published_at"] is None
    assert no_date["source_url"].startswith("https://www.miit.gov.cn/")
    legacy = next(item for item in candidates if item["title"] == "Legacy HTTP official link")
    assert legacy["source_url"] == "https://www.gov.cn/zhengce/legacy.html"


def test_candidate_archive_is_deduplicated(tmp_path):
    session = FakeSession({
        "code": 200,
        "searchVO": {
            "catMap": {
                "all": {
                    "listVO": [{
                        "title": "政策文件",
                        "url": "https://www.gov.cn/zhengce/test.html",
                        "pubtime": "2026-11-13",
                    }]
                }
            }
        },
    })
    candidates, _ = discover_gov_policy_candidates(
        session=session, keywords=("算力",), clock=lambda: "2026-11-14T00:00:00Z",
        pause_seconds=0,
    )
    target = tmp_path / "candidates.jsonl"
    assert append_policy_candidates(target, candidates) == 1
    assert append_policy_candidates(target, candidates) == 0
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == candidates[0]["candidate_id"]


def test_discovery_reports_partial_provider_failures():
    class FailingSession:
        def get(self, *args, **kwargs):
            raise TimeoutError("provider timeout")

    candidates, audit = discover_gov_policy_candidates(
        session=FailingSession(), keywords=("人工智能", "机器人"),
        clock=lambda: "2026-11-14T00:00:00Z", pause_seconds=0,
    )
    assert candidates == []
    assert audit["status"] == "UNAVAILABLE"
    assert audit["queries_failed"] == 2
