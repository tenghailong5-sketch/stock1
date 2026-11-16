import pandas as pd
import pytest

from src.simple_fhm100 import rank_quotes


def sample():
    return pd.DataFrame({
        "code": ["000001", "000002", "300001", "600001", "600002"],
        "name": ["甲", "乙", "丙", "ST丁", "戊"],
        "price": [10, 20, 30, 5, 6],
        "amount": [300e6, 200e6, 100e6, 500e6, 10e6],
        "pct_chg": [2.0, 5.0, 7.0, 3.0, 4.0],
    })


def test_rank_quotes_uses_only_observed_quote_fields_and_never_authorizes_buy():
    result, audit = rank_quotes(sample())
    assert result["code"].tolist() == ["300001", "000002", "000001"]
    assert result["buy_authorization"].eq(False).all()
    assert result["entry_price"].isna().all()
    assert result["stop_price"].isna().all()
    assert audit["industry_mainline_used"] is False
    assert audit["m2_m3_m5_m8_used"] is False


def test_rank_quotes_excludes_overextended_and_illiquid_names():
    result, audit = rank_quotes(sample(), min_amount=50e6)
    assert "600001" not in set(result["code"])
    assert "600002" not in set(result["code"])
    assert audit["candidate_count"] == 3


def test_rank_quotes_empty_pool_is_explicit_not_fabricated():
    frame = sample().assign(pct_chg=12.0)
    result, audit = rank_quotes(frame)
    assert result.empty
    assert audit["status"] == "NO_ELIGIBLE_QUOTES"


def test_rank_quotes_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="missing columns"):
        rank_quotes(pd.DataFrame({"code": ["000001"]}))
