import json
from datetime import datetime, timezone

from src.providers import global_news


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = 0
        self.params = []

    def get(self, url, *, params, timeout):
        self.calls += 1
        self.params.append(dict(params))
        return FakeResponse({
            "articles": [{
                "title": "Chip export controls discussed",
                "url": "https://example.com/article",
                "domain": "example.com",
                "language": "English",
                "sourcecountry": "United States",
                "seendate": "20261115T010000Z",
            }]
        })


def test_global_news_discovery_is_bounded_auditable_and_candidate_only():
    session = FakeSession()
    rows, audit = global_news.discover_global_news_candidates(
        session=session,
        clock=lambda: datetime(2026, 11, 15, 2, tzinfo=timezone.utc),
        per_theme_limit=5,
    )
    assert session.calls == len(global_news.QUERY_THEMES)
    assert all(params["timespan"] == "30d" for params in session.params)
    assert len(rows) == 1
    assert rows[0]["theme"] in global_news.QUERY_THEMES
    assert rows[0]["timestamp_basis"].startswith("retrieval_time")
    assert rows[0]["impact_direction"] == "unclear"
    assert rows[0]["industry_mapping"] == []
    assert rows[0]["historical_pit_eligible"] is False
    assert rows[0]["formal_stock_score_eligible"] is False
    assert audit["status"] == "READY_CANDIDATES_ONLY"
    assert audit["publisher_publication_time_verified"] is False


def test_global_news_archive_deduplicates_candidate_ids(tmp_path):
    path = tmp_path / "global_news.jsonl"
    rows = [{
        "candidate_id": "global_news_abc",
        "title": "headline",
        "historical_pit_eligible": False,
    }]
    assert global_news.append_global_news_candidates(path, rows) == 1
    assert global_news.append_global_news_candidates(path, rows) == 0
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert saved == rows


def test_global_news_retries_one_rate_limit_and_throttles_topics(monkeypatch):
    monkeypatch.setattr(global_news.time, "sleep", lambda _seconds: None)

    class RateLimitResponse(FakeResponse):
        status_code = 429

    class RetrySession:
        def __init__(self):
            self.calls = 0

        def get(self, url, *, params, timeout):
            self.calls += 1
            if self.calls == 1:
                return RateLimitResponse({})
            return FakeResponse({"articles": [{
                "title": "Central bank policy update",
                "url": f"https://example.com/article-{self.calls}",
                "domain": "example.com",
            }]})

    session = RetrySession()
    rows, audit = global_news.discover_global_news_candidates(
        session=session,
        clock=lambda: datetime(2026, 11, 15, 2, tzinfo=timezone.utc),
    )
    assert session.calls == len(global_news.QUERY_THEMES) + 1
    assert audit["queries_succeeded"] == len(global_news.QUERY_THEMES)
    assert audit["queries_failed"] == 0
    assert audit["status"] == "READY_CANDIDATES_ONLY"
    assert len(rows) == len(global_news.QUERY_THEMES)



def test_global_news_retries_two_bounded_rate_limits(monkeypatch):
    sleeps = []
    monkeypatch.setattr(global_news.time, "sleep", sleeps.append)

    class RateLimitResponse(FakeResponse):
        status_code = 429

    class RetryTwiceSession:
        def __init__(self):
            self.calls = 0

        def get(self, url, *, params, timeout):
            self.calls += 1
            if self.calls <= 2:
                return RateLimitResponse({})
            return FakeResponse({"articles": [{
                "title": "Energy policy update",
                "url": f"https://example.com/retry-{self.calls}",
                "domain": "example.com",
            }]})

    session = RetryTwiceSession()
    rows, audit = global_news.discover_global_news_candidates(
        session=session,
        clock=lambda: datetime(2026, 11, 15, 2, tzinfo=timezone.utc),
    )
    assert session.calls == len(global_news.QUERY_THEMES) + 2
    assert audit["queries_succeeded"] == len(global_news.QUERY_THEMES)
    assert audit["queries_failed"] == 0
    assert audit["status"] == "READY_CANDIDATES_ONLY"
    assert len(rows) == len(global_news.QUERY_THEMES)
    assert sleeps[:2] == [4.0, 8.0]
