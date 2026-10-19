"""option_signal_analysis: the demeaned target, forward alignment and the chain query."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research import option_signal_analysis as osa


def test_returns_are_demeaned_within_each_bucket_and_date() -> None:
    day1, day2 = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")
    df = pd.DataFrame(
        {
            "bucket": ["b1", "b1", "b2", "b1"],
            "collection_date": [day1, day1, day1, day2],
            "fwd_opt_ret": [0.1, 0.3, 0.5, -0.2],
        }
    )
    resid = osa.demean_within_bucket_date(df)
    np.testing.assert_allclose(resid, [-0.1, 0.1, 0.0, 0.0])


def test_attach_spot_drops_rows_whose_forward_price_skips_sessions() -> None:
    dates = pd.bdate_range("2025-01-01", periods=4)
    closes = pd.DataFrame({"ticker": "ABC", "date": dates, "close": [10.0, 11.0, 12.0, 13.0]})
    panel = pd.DataFrame(
        {
            "symbol": "ABC",
            "collection_date": [dates[0], dates[1]],
            "fwd_date": [dates[1], dates[3]],  # the second contract was missing on dates[2]
        }
    )
    out = osa.attach_spot(panel, closes, horizon=1)
    assert out["collection_date"].tolist() == [dates[0]]
    assert out["fwd_spot"].tolist() == [11.0]


def test_contract_query_binds_values_and_filters_after_the_windows(fake_db) -> None:
    osa.load_contract_panel(["ABC"], None, None, horizon=5, min_oi=100, min_price=0.1, min_dte=5)
    call = fake_db.last
    assert "LEAD(price, %(horizon)s)" in call.sql
    assert "ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING" in call.sql
    inner, outer = call.sql.split(") contracts")
    assert "oi >= %(min_oi)s" in outer
    assert "oi >=" not in inner
    assert call.params["symbols"] == ["ABC"]
    assert call.params["horizon"] == 5


def test_output_stem_abbreviates_long_symbol_lists() -> None:
    assert osa.output_stem(["A", "B"], 5, "residual_ret") == "A_B_h5_residual_ret"
    many = [f"S{i}" for i in range(10)]
    assert osa.output_stem(many, 5, "fwd_opt_ret") == "10symbols_h5_fwd_opt_ret"
