"""Tests for queries/signal_queries.py: one history fetch, classification, RS enrichment."""

import numpy as np
import pandas as pd
import pytest

from stock_dashboard.logic.option_signals import SIGNAL_HISTORY_SESSIONS
from stock_dashboard.queries import signal_queries as sig

DATES = pd.bdate_range("2025-01-01", periods=25)


@pytest.fixture
def option_history() -> pd.DataFrame:
    """Two symbols over 25 sessions; AAA spikes on the latest one, BBB has negative GEX."""
    frames = []
    for symbol, last_volume, gxoi in [("AAA", 500.0, 10.0), ("BBB", 100.0, -5.0)]:
        n = len(DATES)
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "collection_date": DATES,
                    "vannaxoi": np.arange(n, dtype=float) * (1 if symbol == "AAA" else -1),
                    "flownet": np.arange(n, dtype=float) * (-1 if symbol == "AAA" else 1),
                    "option_volume": [100.0] * (n - 1) + [last_volume],
                    "dxoi": [100.0] * (n - 1) + [200.0 if symbol == "AAA" else 101.0],
                    "gxoi": [1.0] * (n - 1) + [gxoi],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def rs_rows():
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "etf_ticker": ["SEMX", "TCHX"],
            "relative_strength": [1.0, 1.0],
            "rs_rank_index": [0.2, 0.7],
            "rs_rank_overall": [0.6, 0.6],
            "rs_z_score": [0.1, 0.9],
        }
    )


def test_history_query_anchors_on_latest_sessions(fake_db):
    sig.fetch_option_signal_history()
    call = fake_db.last
    assert call.database == "options"
    assert (
        "SELECT DISTINCT collection_date FROM underlying_data ORDER BY collection_date DESC"
        in call.sql
    )
    assert "LIMIT %(sessions)s" in call.sql
    assert '"vannaxoi", "flownet", "option_volume", "dxoi", "gxoi"' in call.sql
    assert call.params == {"sessions": SIGNAL_HISTORY_SESSIONS}
    assert "NOW()" not in call.sql
    assert "INTERVAL" not in call.sql


def test_scan_fetches_history_and_enriches(fake_db, option_history):
    fake_db.returns(option_history, rs_rows())
    out = sig.scan_vanna_extreme(20.0)
    assert [c.database for c in fake_db.calls] == ["options", "stocks"]
    row = out.set_index("symbol").loc["AAA"]
    assert row["signal_label"] == "High Vanna"
    assert row["etf_ticker"] == "TCHX"
    assert row["etfs"] == "SEMX, TCHX"
    assert out.set_index("symbol").loc["BBB", "signal_label"] == "Low Vanna"


def test_shared_history_without_enrichment_runs_no_query(fake_db, option_history):
    results = {
        "flownet": sig.scan_flownet_extreme(10.0, history=option_history, enrich=False),
        "volume": sig.scan_volume_spike(2.0, history=option_history, enrich=False),
        "dxoi": sig.scan_dxoi_change(20.0, history=option_history, enrich=False),
        "gex": sig.scan_gex_level(history=option_history, enrich=False),
    }
    assert fake_db.calls == []
    assert results["flownet"]["symbol"].tolist() == ["AAA"]
    assert results["volume"]["symbol"].tolist() == ["AAA"]
    assert results["dxoi"]["symbol"].tolist() == ["AAA"]
    assert results["gex"]["symbol"].tolist() == ["BBB"]


def test_gex_scan_filters_to_negative_gex(fake_db, option_history):
    """The old scan returned every symbol; only the negative-GEX one should remain."""
    fake_db.returns(option_history, rs_rows())
    out = sig.scan_gex_level()
    assert out["symbol"].tolist() == ["BBB"]
    assert out["signal_label"].tolist() == ["Negative GEX"]


def test_empty_result_skips_enrichment(fake_db, option_history):
    flat = option_history.assign(option_volume=100.0)
    out = sig.scan_volume_spike(2.0, history=flat)
    assert out.empty
    assert fake_db.calls == []
