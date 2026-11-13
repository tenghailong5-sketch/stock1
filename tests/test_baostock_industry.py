import pandas as pd
import pytest

from src.providers.baostock_industry import (
    fetch_baostock_industry_membership_snapshot,
)


class FakeResult:
    fields = [
        "updateDate", "code", "code_name", "industry", "industryClassification"
    ]
    error_code = "0"
    error_msg = ""

    def __init__(self, rows):
        self.rows = list(rows)
        self.position = 0

    def next(self):
        self.position += 1
        return self.position <= len(self.rows)

    def get_row_data(self):
        return self.rows[self.position - 1]


class LoginResult:
    def __init__(self, code, message=""):
        self.error_code = code
        self.error_msg = message


class FakeBaostock:
    def __init__(self, login_results):
        self.login_results = list(login_results)
        self.industry_query_dates = []
        self.login_calls = 0
        self.logout_calls = 0

    def login(self):
        self.login_calls += 1
        return self.login_results.pop(0)

    def logout(self):
        self.logout_calls += 1

    def query_stock_industry(self, date=None):
        self.industry_query_dates.append(date)
        return FakeResult([
            ["2026-11-09", "sz.000001", "平安银行", "银行", "申万一级行业"],
            ["2026-11-09", "sh.600000", "浦发银行", "银行", "申万一级行业"],
        ])


def test_baostock_provider_retries_transient_login_failure():
    module = FakeBaostock([
        LoginResult("1001", "网络接收错误"),
        LoginResult("0"),
    ])
    sleeps = []
    frame = fetch_baostock_industry_membership_snapshot(
        module,
        clock=lambda: pd.Timestamp("2026-11-14T08:00:00Z"),
        sleep_fn=sleeps.append,
    )

    assert module.login_calls == 2
    assert sleeps == [1]
    assert module.logout_calls == 2  # cleanup after failed login and query
    assert set(frame["symbol"]) == {"000001", "600000"}
    assert frame["historical_pit_eligible"].eq(False).all()


def test_baostock_provider_stops_after_bounded_login_retries():
    module = FakeBaostock([
        LoginResult("1001", "network"),
        LoginResult("1002", "network"),
    ])
    sleeps = []
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        fetch_baostock_industry_membership_snapshot(
            module, login_attempts=2, sleep_fn=sleeps.append
        )
    assert module.login_calls == 2
    assert sleeps == [1]


def test_baostock_provider_passes_historical_query_date_but_keeps_pit_blocked():
    module = FakeBaostock([LoginResult("0")])
    frame = fetch_baostock_industry_membership_snapshot(
        module,
        as_of_date="2024-12-31",
        clock=lambda: pd.Timestamp("2026-11-14T08:00:00Z"),
    )
    assert module.industry_query_dates == ["2024-12-31"]
    assert frame["query_date"].eq("2024-12-31").all()
    assert frame["historical_pit_eligible"].eq(False).all()
    assert frame["snapshot_scope"].eq("historical_date_query_candidate").all()


def test_baostock_provider_rejects_non_iso_query_date():
    module = FakeBaostock([LoginResult("0")])
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        fetch_baostock_industry_membership_snapshot(
            module, as_of_date="20241231"
        )
    assert module.login_calls == 0
