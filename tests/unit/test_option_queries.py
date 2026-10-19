"""Tests for queries/option_queries.py: read_df is faked."""

import math

import pandas as pd
import pytest

from stock_dashboard.config import get_settings
from stock_dashboard.demo.contract import UNDERLYING_VALUE_COLUMNS
from stock_dashboard.queries import option_queries as oq

COLUMNS = pd.DataFrame({"column_name": ["dxoi", "flownet", "gxoi"]})


@pytest.fixture
def whitelisted(fake_db):
    """First query answers the column whitelist lookup."""
    fake_db.returns(COLUMNS)
    return fake_db


def test_get_all_option_symbols(fake_db):
    fake_db.returns(pd.DataFrame({"symbol": ["AAA", "TSLA"]}))
    assert oq.get_all_option_symbols() == ["AAA", "TSLA"]
    assert fake_db.last.database == "options"


def test_get_option_numerical_columns(fake_db):
    fake_db.returns(COLUMNS)
    assert oq.get_option_numerical_columns() == ["dxoi", "flownet", "gxoi"]
    assert sorted(fake_db.last.params["keys"]) == sorted(oq.KEY_COLUMNS)


def test_fetch_option_last_30_days(fake_db):
    oq.fetch_option_last_30_days("AAA")
    assert "ORDER BY collection_date DESC LIMIT %(limit)s" in fake_db.last.sql
    assert fake_db.last.params == {"symbol": "AAA", "limit": oq.RECENT_ROWS}


class TestValidateColumn:
    def test_rejects_unknown_column(self, whitelisted):
        with pytest.raises(ValueError, match="Invalid column name"):
            oq.validate_column("dxoi; DROP TABLE x")

    def test_whitelist_is_fetched_once(self, whitelisted):
        assert oq.validate_column("dxoi") == "dxoi"
        assert oq.validate_column("gxoi") == "gxoi"
        assert len(whitelisted.calls) == 1

    def test_whitelist_is_reread_after_its_ttl(self, fake_db, monkeypatch):
        """A column added while the app runs is accepted once the TTL has passed."""
        clock = [1000.0]
        monkeypatch.setattr(oq.time, "monotonic", lambda: clock[0])
        fake_db.returns(COLUMNS, pd.DataFrame({"column_name": ["dxoi", "new_col"]}))
        with pytest.raises(ValueError, match="Invalid column name"):
            oq.validate_column("new_col")
        clock[0] += oq.ALLOWED_COLUMNS_TTL
        assert oq.validate_column("new_col") == "new_col"

    def test_whitelist_follows_the_option_database(self, fake_db, monkeypatch):
        fake_db.returns(COLUMNS, pd.DataFrame({"column_name": ["other"]}))
        assert oq.validate_column("dxoi") == "dxoi"
        monkeypatch.setenv("OPTION_DB_NAME", "another_db")
        get_settings.cache_clear()
        assert oq.validate_column("other") == "other"


class TestFetchOptionLatestComparison:
    def test_ratios_from_same_date(self, fake_db):
        latest = pd.Timestamp("2025-01-03")
        data = pd.DataFrame(
            {
                "collection_date": [latest],
                "symbol": ["AAA"],
                "dxoi": [200.0],
                "gxoi": [5.0],
                "flownet": [1.0],
                "extra": [3.0],
            }
        )
        avg = pd.DataFrame(
            {
                "date": [latest],
                "symbol": ["AAA"],
                "dxoi_avg": [100.0],
                "gxoi_avg": [0.0],
                "flownet_avg": [None],
            }
        )
        fake_db.returns(data, avg)
        result = oq.fetch_option_latest_comparison("AAA")
        assert result["dxoi"] == pytest.approx(2.0)
        assert math.isnan(result["gxoi"])  # zero average
        assert math.isnan(result["flownet"])  # missing average
        assert "extra" not in result
        assert "symbol" not in result
        first, second = fake_db.calls
        assert "a.date = u.collection_date" in first.sql
        assert second.params == {"symbol": "AAA", "date": latest.date()}

    def test_empty_when_no_dated_pair(self, fake_db):
        assert oq.fetch_option_latest_comparison("NONE") == {}
        assert len(fake_db.calls) == 1

    def test_empty_when_average_row_vanished(self, fake_db):
        data = pd.DataFrame({"collection_date": [pd.Timestamp("2025-01-03")], "dxoi": [1.0]})
        fake_db.returns(data, pd.DataFrame())
        assert oq.fetch_option_latest_comparison("AAA") == {}


def test_ratio_handles_non_numeric():
    assert math.isnan(oq._ratio("x", 1.0))
    assert oq._ratio(3, 2) == 1.5


AVG_COLUMNS = pd.DataFrame({"column_name": ["symbol", "date", "dxoi_avg"]})


class TestFetchOptionColumnHistory:
    def test_quotes_identifiers(self, whitelisted):
        whitelisted.returns(
            pd.DataFrame({"collection_date": [], "dxoi": []}), AVG_COLUMNS, COLUMNS, pd.DataFrame()
        )
        data, _avg = oq.fetch_option_column_history("AAA", "dxoi")
        assert "dxoi" in data.columns
        history_call, avg_call = whitelisted.calls[1], whitelisted.calls[-1]
        assert 'SELECT collection_date, "dxoi" FROM underlying_data' in history_call.sql
        assert 'SELECT date, "dxoi_avg" FROM underlying_data_avg' in avg_call.sql
        assert avg_call.params == {"symbol": "AAA"}

    def test_column_without_average_skips_the_average_query(self, whitelisted):
        """A provider column with no _avg partner once raised UndefinedColumn."""
        values = pd.DataFrame({"collection_date": [pd.Timestamp("2025-01-02")], "gxoi": [1.0]})
        whitelisted.returns(values, AVG_COLUMNS, COLUMNS)
        data, avg = oq.fetch_option_column_history("AAA", "gxoi")
        assert data["gxoi"].tolist() == [1.0]
        assert avg.empty
        assert list(avg.columns) == ["date", "gxoi_avg"]
        assert not any("gxoi_avg" in call.sql for call in whitelisted.calls)

    def test_rejects_unknown_column(self, whitelisted):
        with pytest.raises(ValueError, match="Invalid column name"):
            oq.fetch_option_column_history("AAA", "secret")

    def test_column_ending_in_avg_is_read_as_given(self, fake_db):
        """A provider's underlying_data column named iv_avg was once looked up as iv."""
        fake_db.returns(pd.DataFrame({"column_name": ["dxoi", "iv_avg"]}))
        values = pd.DataFrame({"collection_date": [pd.Timestamp("2025-01-02")], "iv_avg": [0.3]})
        fake_db.returns(values, AVG_COLUMNS, pd.DataFrame({"column_name": ["dxoi", "iv_avg"]}))
        data, avg = oq.fetch_option_column_history("AAA", "iv_avg")
        assert data["iv_avg"].tolist() == [0.3]
        assert 'SELECT collection_date, "iv_avg" FROM underlying_data' in fake_db.calls[1].sql
        assert avg.empty


class TestFetchOptionScreenerResults:
    def test_same_day_average(self, whitelisted):
        whitelisted.returns(
            AVG_COLUMNS, COLUMNS, pd.DataFrame({"symbol": ["AAA"], "multiple": [5.0]})
        )
        result = oq.fetch_option_screener_results("dxoi", 5.0)
        assert result["symbol"].tolist() == ["AAA"]
        text = whitelisted.last.sql
        assert "a.symbol = d.symbol AND a.date = d.collection_date" in text
        assert "d.collection_date = (SELECT MAX(collection_date) FROM underlying_data)" in text
        assert 'd."dxoi"::float / a."dxoi_avg" >= %(min_multiple)s' in text
        assert 'a."dxoi_avg" > 0' in text  # a negative average would flip the multiple
        assert whitelisted.last.params == {"min_multiple": 5.0}

    def test_rejects_injection(self, whitelisted):
        with pytest.raises(ValueError, match="Invalid column name"):
            oq.fetch_option_screener_results("dxoi; DROP TABLE--", 5.0)

    def test_rejects_a_column_without_an_average(self, whitelisted):
        """A numeric column with no _avg partner once raised UndefinedColumn in SQL."""
        whitelisted.returns(AVG_COLUMNS, COLUMNS)
        with pytest.raises(ValueError, match="has no gxoi_avg average"):
            oq.fetch_option_screener_results("gxoi", 5.0)
        assert not any("JOIN underlying_data_avg" in call.sql for call in whitelisted.calls)


def test_get_option_average_columns_needs_an_avg_partner(fake_db):
    fake_db.returns(
        pd.DataFrame({"column_name": ["symbol", "date", "dxoi_avg", "gxoi_avg"]}), COLUMNS
    )
    assert oq.get_option_average_columns() == ["dxoi", "gxoi"]
    assert "underlying_data_avg" in fake_db.calls[0].sql


class TestGetOptionMultipleColumns:
    def test_contract_columns_only_non_negative_ones(self, fake_db):
        averaged = ["option_volume", "flownet", "dxoi", "volatility", "put_call_ratio"]
        fake_db.returns(
            pd.DataFrame({"column_name": [f"{c}_avg" for c in averaged]}),
            pd.DataFrame({"column_name": sorted(averaged)}),
        )
        assert oq.get_option_multiple_columns() == ["option_volume", "put_call_ratio", "volatility"]
        assert len(fake_db.calls) == 2  # no extra columns: no minimum query

    def test_provider_column_needs_avg_and_no_negative_latest_value(self, fake_db):
        averaged = ["option_volume", "skew_extra", "spread_extra"]
        fake_db.returns(
            pd.DataFrame({"column_name": [f"{c}_avg" for c in averaged]}),
            pd.DataFrame({"column_name": [*averaged, "no_avg_extra"]}),
            pd.DataFrame({"skew_extra": [-0.2], "spread_extra": [0.0]}),
        )
        assert oq.get_option_multiple_columns() == ["option_volume", "spread_extra"]
        minima = fake_db.last.sql
        assert 'MIN("skew_extra")' in minima
        assert "no_avg_extra" not in minima
        assert "MAX(collection_date)" in minima

    def test_non_negative_columns_are_contract_columns(self):
        assert set(UNDERLYING_VALUE_COLUMNS) >= oq.NON_NEGATIVE_COLUMNS
        signed = {"flownet", "vflownet", "deltas", "gammas", "dxoi", "gxoi", "vannaxoi"}
        assert not signed & oq.NON_NEGATIVE_COLUMNS


@pytest.mark.parametrize("has_rows", [True, False])
def test_has_underlying_rows(fake_db, has_rows):
    fake_db.returns(pd.DataFrame({"has_rows": [has_rows]}))
    assert oq.has_underlying_rows() is has_rows
    assert "EXISTS" in fake_db.last.sql
