"""Tests for the backtesting engine — pure logic, no DB."""

import numpy as np
import pandas as pd
import pytest

from stock_dashboard.logic.backtest import (
    CONFLICTING_FILTERS,
    FILTER_RANGES,
    FILTER_TO_COLUMN,
    FilterSpec,
    HorizonStats,
    Strategy,
    StrategyResult,
    apply_bh_correction,
    apply_filter,
    build_backtest_dataframe,
    classify_regimes,
    compute_forward_returns,
    compute_rs_impulse,
    evaluate_strategy,
    generate_strategies,
    precompute_indicators,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def spy_ohlcv() -> pd.DataFrame:
    """Synthetic SPY data: 500 days with a clear trend shift."""
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range("2023-01-01", periods=n)

    # Bull phase (0-250), then bear phase (250-400), then recovery (400-500)
    close = np.empty(n)
    close[0] = 400.0
    for i in range(1, 250):
        close[i] = close[i - 1] * (1 + 0.0005 + np.random.randn() * 0.005)
    for i in range(250, 400):
        close[i] = close[i - 1] * (1 - 0.001 + np.random.randn() * 0.008)
    for i in range(400, n):
        close[i] = close[i - 1] * (1 + 0.0003 + np.random.randn() * 0.005)

    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": np.random.randint(50_000_000, 150_000_000, size=n),
        }
    )


@pytest.fixture
def multi_ticker_ohlcv() -> pd.DataFrame:
    """OHLCV for 3 tickers, 300 days each."""
    np.random.seed(42)
    frames = []
    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        dates = pd.bdate_range("2023-01-01", periods=300)
        close = 100 + np.cumsum(np.random.randn(300) * 0.5)
        # Ensure positive prices
        close = np.maximum(close, 10.0)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": np.random.randint(100_000, 5_000_000, size=300).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def sample_indicators_df() -> pd.DataFrame:
    """Pre-built indicator DataFrame with forward returns for testing evaluation."""
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2023-06-01", periods=n)

    # Repeat for 2 tickers
    rows = []
    for ticker in ["AAPL", "MSFT"]:
        for _i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "rsi_14": 30 + np.random.rand() * 40,  # 30-70
                    "momentum_63": -10 + np.random.rand() * 30,  # -10 to 20
                    "ratio_to_200ma": 0.9 + np.random.rand() * 0.2,  # 0.9 - 1.1
                    "volume_z": -1 + np.random.rand() * 4,  # -1 to 3
                    "weinstein_stage": np.random.choice(
                        ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
                    ),
                    "rs_rank_overall": np.random.rand(),
                    "rs_z_score": np.random.randn(),
                    "fwd_5": np.random.randn() * 0.02,
                    "fwd_10": np.random.randn() * 0.03,
                    "fwd_20": np.random.randn() * 0.04,
                    "fwd_63": np.random.randn() * 0.06,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestClassifyRegimes
# ---------------------------------------------------------------------------


class TestClassifyRegimes:
    def test_returns_series_indexed_by_date(self, spy_ohlcv):
        regimes = classify_regimes(spy_ohlcv)
        assert isinstance(regimes, pd.Series)
        assert regimes.index.name == "date"

    def test_valid_labels(self, spy_ohlcv):
        regimes = classify_regimes(spy_ohlcv)
        valid_labels = {"bull", "bear", "volatile", "correction"}
        non_null = regimes.dropna()
        assert set(non_null.unique()).issubset(valid_labels)

    def test_early_dates_are_nan(self, spy_ohlcv):
        """First ~200 days should be NaN (need 200-day SMA)."""
        regimes = classify_regimes(spy_ohlcv)
        # First 199 entries should be NaN (need 200 periods for SMA)
        assert regimes.iloc[:199].isna().all()

    def test_bull_when_above_sma200(self):
        """Steady uptrend with low vol should be 'bull'."""
        n = 300
        dates = pd.bdate_range("2022-01-01", periods=n)
        close = np.linspace(300, 500, n)  # Steady uptrend
        df = pd.DataFrame(
            {
                "date": dates,
                "open": close - 1,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": [100_000_000] * n,
            }
        )
        regimes = classify_regimes(df)
        # Late dates with steady uptrend and low vol should be bull
        late_regimes = regimes.iloc[-50:]
        assert (late_regimes == "bull").all()

    def test_correction_takes_priority(self):
        """A >10% drawdown from 63-day high should label as correction."""
        n = 300
        dates = pd.bdate_range("2022-01-01", periods=n)
        # Flat then sudden 15% drop
        close = np.full(n, 400.0)
        close[250:] = 340.0  # 15% drop
        df = pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": [100_000_000] * n,
            }
        )
        regimes = classify_regimes(df)
        # After the drop, should be "correction"
        post_drop = regimes.iloc[260:270]
        assert (post_drop == "correction").all()


# ---------------------------------------------------------------------------
# TestPrecomputeIndicators
# ---------------------------------------------------------------------------


class TestPrecomputeIndicators:
    def test_output_columns(self, multi_ticker_ohlcv):
        result = precompute_indicators(multi_ticker_ohlcv)
        expected_cols = {
            "date",
            "ticker",
            "rsi_14",
            "momentum_20",
            "momentum_63",
            "momentum_126",
            "ratio_to_50ma",
            "ratio_to_200ma",
            "macd_histogram",
            "volume_z",
            "adr_14",
            "weinstein_stage",
            "weinstein_stage_num",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_row_count_matches_input(self, multi_ticker_ohlcv):
        result = precompute_indicators(multi_ticker_ohlcv)
        assert len(result) == len(multi_ticker_ohlcv)

    def test_all_tickers_present(self, multi_ticker_ohlcv):
        result = precompute_indicators(multi_ticker_ohlcv)
        assert set(result["ticker"].unique()) == {"AAPL", "MSFT", "GOOGL"}

    def test_rsi_in_valid_range(self, multi_ticker_ohlcv):
        result = precompute_indicators(multi_ticker_ohlcv)
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_empty_input(self):
        empty = pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])
        result = precompute_indicators(empty)
        assert result.empty

    def test_weinstein_stage_values(self, multi_ticker_ohlcv):
        result = precompute_indicators(multi_ticker_ohlcv)
        valid_stages = {"Stage 1", "Stage 2", "Stage 3", "Stage 4", "Undetermined"}
        assert set(result["weinstein_stage"].unique()).issubset(valid_stages)


# ---------------------------------------------------------------------------
# TestComputeForwardReturns
# ---------------------------------------------------------------------------


class TestComputeForwardReturns:
    def test_output_columns(self, multi_ticker_ohlcv):
        result = compute_forward_returns(multi_ticker_ohlcv)
        assert {"date", "ticker", "fwd_5", "fwd_10", "fwd_20", "fwd_63"}.issubset(
            set(result.columns)
        )

    def test_row_count(self, multi_ticker_ohlcv):
        result = compute_forward_returns(multi_ticker_ohlcv)
        assert len(result) == len(multi_ticker_ohlcv)

    def test_last_rows_have_nans(self, multi_ticker_ohlcv):
        """Last N rows for each ticker should have NaN for fwd_N."""
        result = compute_forward_returns(multi_ticker_ohlcv)
        aapl = result[result["ticker"] == "AAPL"].sort_values("date")
        # Last 5 rows should have NaN for fwd_5
        assert aapl["fwd_5"].iloc[-5:].isna().all()
        # Last 63 rows should have NaN for fwd_63
        assert aapl["fwd_63"].iloc[-63:].isna().all()

    def test_forward_return_arithmetic(self):
        """Verify fwd_5 = (close[i+5] - close[i]) / close[i]."""
        dates = pd.bdate_range("2023-01-01", periods=20)
        close = np.arange(100.0, 120.0)
        df = pd.DataFrame(
            {
                "ticker": "TEST",
                "date": dates,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": [1_000_000] * 20,
            }
        )
        result = compute_forward_returns(df)
        # Row 0: close=100, close[5]=105, fwd_5 = 5/100 = 0.05
        assert result["fwd_5"].iloc[0] == pytest.approx(0.05, abs=1e-10)
        # Row 10: close=110, close[15]=115, fwd_5 = 5/110
        assert result["fwd_5"].iloc[10] == pytest.approx(5.0 / 110.0, abs=1e-10)

    def test_empty_input(self):
        empty = pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])
        result = compute_forward_returns(empty)
        assert result.empty


# ---------------------------------------------------------------------------
# TestEvaluateStrategy
# ---------------------------------------------------------------------------


class TestEvaluateStrategy:
    def test_basic_filtering(self, sample_indicators_df):
        """A filter should reduce n_signals below total rows."""
        strategy = Strategy(
            name="high_rs",
            filters=[FilterSpec(name="rs_rank_overall_min", value=0.9)],
        )
        result = evaluate_strategy(sample_indicators_df, strategy)
        assert result.n_signals < len(sample_indicators_df)
        assert result.n_signals > 0

    def test_no_matching_signals(self, sample_indicators_df):
        """An impossible filter should yield 0 signals."""
        strategy = Strategy(
            name="impossible",
            filters=[FilterSpec(name="rsi_min", value=200)],  # RSI never > 200
        )
        result = evaluate_strategy(sample_indicators_df, strategy)
        assert result.n_signals == 0
        assert len(result.horizons) == 0

    def test_regime_filter(self, sample_indicators_df):
        """With regime filter, signals should only come from matching dates."""
        # Create regime series matching indicator dates
        dates = sample_indicators_df["date"].unique()
        regime = pd.Series("bull", index=pd.DatetimeIndex(dates))
        # Make half the dates "bear"
        regime.iloc[len(regime) // 2 :] = "bear"

        strategy = Strategy(
            name="bull_only",
            filters=[FilterSpec(name="rsi_min", value=30)],
            regime="bull",
        )
        result = evaluate_strategy(sample_indicators_df, strategy, regime)
        # Should have fewer signals than without regime filter
        result_no_regime = evaluate_strategy(
            sample_indicators_df,
            Strategy(name="all", filters=[FilterSpec(name="rsi_min", value=30)]),
        )
        assert result.n_signals < result_no_regime.n_signals

    def test_regime_strategy_without_regime_series_has_no_signals(self, sample_indicators_df):
        """A regime that cannot be evaluated must not silently become unconditional."""
        filters = [FilterSpec(name="rsi_min", value=30)]
        unconditional = evaluate_strategy(sample_indicators_df, Strategy("all", filters))
        bear = evaluate_strategy(sample_indicators_df, Strategy("bear", filters, regime="bear"))
        assert unconditional.n_signals > 0
        assert bear.n_signals == 0
        assert bear.horizons == {}

    def test_horizon_stats_populated(self, sample_indicators_df):
        """With enough signals, horizons should have stats."""
        strategy = Strategy(
            name="broad",
            filters=[FilterSpec(name="rsi_min", value=30)],  # Most pass
        )
        result = evaluate_strategy(sample_indicators_df, strategy)
        assert len(result.horizons) > 0
        for _label, hs in result.horizons.items():
            assert isinstance(hs.mean_return, float)
            assert 0.0 <= hs.win_rate <= 1.0
            assert 0.0 <= hs.p_value <= 1.0

    def test_min_signal_threshold(self, sample_indicators_df):
        """Strategies with <10 signals should have empty horizons."""
        # Filter to get very few signals
        strategy = Strategy(
            name="few_signals",
            filters=[
                FilterSpec(name="rsi_min", value=69),
                FilterSpec(name="rs_rank_overall_min", value=0.99),
            ],
        )
        result = evaluate_strategy(sample_indicators_df, strategy)
        if result.n_signals < 10:
            assert len(result.horizons) == 0

    def test_unknown_filter_name(self, sample_indicators_df):
        """An unknown filter name should yield 0 signals."""
        strategy = Strategy(
            name="bad_filter",
            filters=[FilterSpec(name="nonexistent_min", value=1.0)],
        )
        result = evaluate_strategy(sample_indicators_df, strategy)
        assert result.n_signals == 0

    def test_weinstein_stage_filter(self, sample_indicators_df):
        strategy = Strategy(
            name="stage2",
            filters=[FilterSpec(name="weinstein_stage", value="Stage 2")],
        )
        result = evaluate_strategy(sample_indicators_df, strategy)
        # Should only include Stage 2 rows
        assert result.n_signals > 0
        mask = sample_indicators_df["weinstein_stage"] == "Stage 2"
        assert result.n_signals == mask.sum()


# ---------------------------------------------------------------------------
# TestBHCorrection
# ---------------------------------------------------------------------------


class TestBHCorrection:
    def test_single_pvalue(self):
        """Single p-value should be unchanged."""
        hs = HorizonStats(
            mean_return=0.01, median_return=0.01, win_rate=0.6, sharpe=0.5, p_value=0.03
        )
        result = StrategyResult(
            strategy=Strategy(name="test", filters=[]),
            n_signals=100,
            horizons={"T+5": hs},
        )
        apply_bh_correction([result])
        assert result.horizons["T+5"].p_value_adj == pytest.approx(0.03, abs=1e-10)
        assert result.horizons["T+5"].significant is True

    def test_multiple_pvalues_adjusted(self):
        """Multiple p-values should be adjusted upward."""
        results = []
        for i, p in enumerate([0.01, 0.02, 0.04, 0.10]):
            hs = HorizonStats(
                mean_return=0.01, median_return=0.01, win_rate=0.6, sharpe=0.5, p_value=p
            )
            results.append(
                StrategyResult(
                    strategy=Strategy(name=f"s{i}", filters=[]),
                    n_signals=100,
                    horizons={"T+5": hs},
                )
            )
        apply_bh_correction(results)

        # All adjusted p-values should be >= original p-values
        for i, p_orig in enumerate([0.01, 0.02, 0.04, 0.10]):
            assert results[i].horizons["T+5"].p_value_adj >= p_orig

    def test_significance_threshold(self):
        """Only p_value_adj < 0.05 should be marked significant."""
        results = []
        for p in [0.001, 0.5]:
            hs = HorizonStats(
                mean_return=0.01, median_return=0.01, win_rate=0.6, sharpe=0.5, p_value=p
            )
            results.append(
                StrategyResult(
                    strategy=Strategy(name=f"p={p}", filters=[]),
                    n_signals=100,
                    horizons={"T+5": hs},
                )
            )
        apply_bh_correction(results)
        assert results[0].horizons["T+5"].significant is True
        assert results[1].horizons["T+5"].significant is False

    def test_empty_results(self):
        """No results should not raise."""
        apply_bh_correction([])

    def test_no_horizons(self):
        """Results with no horizons should not raise."""
        result = StrategyResult(
            strategy=Strategy(name="empty", filters=[]),
            n_signals=0,
        )
        apply_bh_correction([result])

    def test_adjusted_values_capped_at_one(self):
        """Adjusted p-values should never exceed 1.0."""
        results = []
        for p in [0.3, 0.4, 0.5, 0.9]:
            hs = HorizonStats(
                mean_return=0.01, median_return=0.01, win_rate=0.6, sharpe=0.5, p_value=p
            )
            results.append(
                StrategyResult(
                    strategy=Strategy(name=f"p={p}", filters=[]),
                    n_signals=100,
                    horizons={"T+5": hs},
                )
            )
        apply_bh_correction(results)
        for r in results:
            assert r.horizons["T+5"].p_value_adj <= 1.0


# ---------------------------------------------------------------------------
# TestGenerateStrategies
# ---------------------------------------------------------------------------


class TestGenerateStrategies:
    def test_returns_list(self):
        strategies = generate_strategies(max_filters=1)
        assert isinstance(strategies, list)
        assert len(strategies) > 0

    def test_strategy_has_required_fields(self):
        strategies = generate_strategies(max_filters=1)
        s = strategies[0]
        assert isinstance(s.name, str)
        assert isinstance(s.filters, list)
        assert len(s.filters) >= 1

    def test_no_conflicting_pairs(self):
        """No strategy should have both rsi_min and rsi_max."""
        strategies = generate_strategies(max_filters=3)
        for s in strategies:
            names = {f.name for f in s.filters}
            assert not ({"rsi_min", "rsi_max"}.issubset(names)), f"Conflict in {s.name}"
            assert not ({"ratio_to_200ma_min", "ratio_to_200ma_max"}.issubset(names)), (
                f"Conflict in {s.name}"
            )

    def test_no_duplicate_filter_names(self):
        """No strategy should have two filters with the same name."""
        strategies = generate_strategies(max_filters=3)
        for s in strategies:
            names = [f.name for f in s.filters]
            assert len(names) == len(set(names)), f"Duplicate in {s.name}"

    def test_regime_coverage(self):
        """Strategies should cover all regimes including None."""
        strategies = generate_strategies(max_filters=1)
        regimes_seen = {s.regime for s in strategies}
        assert None in regimes_seen
        assert "bull" in regimes_seen
        assert "bear" in regimes_seen

    def test_count_sanity(self):
        """With max_filters=1, count should be manageable."""
        s1 = generate_strategies(max_filters=1)
        s2 = generate_strategies(max_filters=2)
        s3 = generate_strategies(max_filters=3)
        assert len(s1) < len(s2) < len(s3)
        # After events + S/R: ~260 / ~6.5K / ~100K
        assert len(s1) > 100
        assert len(s3) < 200_000

    def test_max_filters_respected(self):
        strategies = generate_strategies(max_filters=2)
        for s in strategies:
            assert len(s.filters) <= 2


# ---------------------------------------------------------------------------
# TestBuildBacktestDataframe
# ---------------------------------------------------------------------------


class TestBuildBacktestDataframe:
    def test_has_indicators_and_forward_returns(self, multi_ticker_ohlcv):
        result = build_backtest_dataframe(multi_ticker_ohlcv)
        assert "rsi_14" in result.columns
        assert "fwd_5" in result.columns
        assert "fwd_63" in result.columns

    def test_with_metrics(self, multi_ticker_ohlcv):
        metrics = pd.DataFrame(
            {
                "date": multi_ticker_ohlcv["date"].iloc[:10],
                "ticker": multi_ticker_ohlcv["ticker"].iloc[:10],
                "etf_ticker": "XLK",
                "rs_rank_overall": np.random.rand(10),
                "rs_z_score": np.random.randn(10),
            }
        )
        result = build_backtest_dataframe(multi_ticker_ohlcv, metrics)
        assert "rs_rank_overall" in result.columns
        assert "rs_z_score" in result.columns


# ---------------------------------------------------------------------------
# TestADRFilter
# ---------------------------------------------------------------------------


class TestADRFilter:
    def test_adr_in_filter_ranges(self):
        """adr_14_min should be in FILTER_RANGES."""
        assert "adr_14_min" in FILTER_RANGES
        assert len(FILTER_RANGES["adr_14_min"]) >= 1

    def test_adr_filter_evaluation(self, sample_indicators_df):
        """adr_14_min filter should work in strategy evaluation."""
        # Add adr_14 column to sample data
        sample_indicators_df["adr_14"] = np.random.uniform(1.0, 10.0, len(sample_indicators_df))
        strategy = Strategy(
            name="high_adr",
            filters=[FilterSpec(name="adr_14_min", value=5.0)],
        )
        result = evaluate_strategy(sample_indicators_df, strategy)
        assert result.n_signals > 0
        assert result.n_signals < len(sample_indicators_df)


# ---------------------------------------------------------------------------
# TestPatternFilters
# ---------------------------------------------------------------------------


class TestPatternFilters:
    def test_pattern_columns_in_precompute(self, multi_ticker_ohlcv):
        """precompute_indicators() should produce has_* pattern columns."""
        result = precompute_indicators(multi_ticker_ohlcv)
        expected_pattern_cols = {
            "has_high_tight_flag",
            "has_bull_flag",
            "has_bear_flag",
            "has_cup_and_handle",
            "has_v_bottom",
        }
        assert expected_pattern_cols.issubset(set(result.columns))

    def test_pattern_columns_are_int8(self, multi_ticker_ohlcv):
        """Pattern columns should be int8 (0/1)."""
        result = precompute_indicators(multi_ticker_ohlcv)
        for col in ["has_bull_flag", "has_bear_flag", "has_v_bottom"]:
            assert result[col].dtype == np.int8

    def test_pattern_filters_in_filter_to_column(self):
        """All has_* pattern filters should be in FILTER_TO_COLUMN."""
        for name in [
            "has_bull_flag",
            "has_bear_flag",
            "has_high_tight_flag",
            "has_cup_and_handle",
            "has_v_bottom",
        ]:
            assert name in FILTER_TO_COLUMN

    def test_pattern_filters_in_filter_ranges(self):
        """All has_* pattern filters should be in FILTER_RANGES."""
        for name in [
            "has_bull_flag",
            "has_bear_flag",
            "has_high_tight_flag",
            "has_cup_and_handle",
            "has_v_bottom",
        ]:
            assert name in FILTER_RANGES
            assert FILTER_RANGES[name] == [True]

    def test_apply_filter_has_prefix(self):
        """apply_filter should treat has_ prefix as boolean."""
        series = pd.Series([0, 1, 0, 1, 0], dtype=np.int8)
        fspec = FilterSpec(name="has_bull_flag", value=True)
        mask = apply_filter(series, fspec)
        assert mask.tolist() == [False, True, False, True, False]

    def test_pattern_filter_in_strategy(self, sample_indicators_df):
        """A has_bull_flag filter should work in evaluate_strategy."""
        # Add pattern column: ~20% of rows have pattern
        np.random.seed(42)
        sample_indicators_df["has_bull_flag"] = np.random.choice(
            [0, 1],
            size=len(sample_indicators_df),
            p=[0.8, 0.2],
        ).astype(np.int8)
        strategy = Strategy(
            name="bull_flags",
            filters=[FilterSpec(name="has_bull_flag", value=True)],
        )
        result = evaluate_strategy(sample_indicators_df, strategy)
        expected = int(sample_indicators_df["has_bull_flag"].sum())
        assert result.n_signals == expected

    def test_generate_strategies_includes_patterns(self):
        """generate_strategies should produce pattern-based strategies."""
        strategies = generate_strategies(max_filters=1)
        pattern_strategies = [
            s for s in strategies if any(f.name.startswith("has_") for f in s.filters)
        ]
        assert len(pattern_strategies) > 0


# ---------------------------------------------------------------------------
# TestRSImpulse
# ---------------------------------------------------------------------------


class TestRSImpulse:
    @pytest.fixture
    def impulse_df(self) -> pd.DataFrame:
        """DataFrame with linear rs_rank_overall for predictable impulse values."""
        frames = []
        dates = pd.bdate_range("2023-01-01", periods=50)
        for ticker in ["AAPL", "MSFT"]:
            # Linear rank: 0.0, 0.02, 0.04, ... for AAPL; reversed for MSFT
            ranks = np.linspace(0.0, 0.98, 50) if ticker == "AAPL" else np.linspace(0.98, 0.0, 50)
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "ticker": ticker,
                        "rs_rank_overall": ranks.astype(np.float32),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def test_output_columns(self, impulse_df):
        result = compute_rs_impulse(impulse_df)
        for col in ["rs_impulse_5", "rs_impulse_10", "rs_impulse_20", "rs_impulse_42"]:
            assert col in result.columns

    def test_float32_dtype(self, impulse_df):
        result = compute_rs_impulse(impulse_df)
        for col in ["rs_impulse_5", "rs_impulse_10", "rs_impulse_20", "rs_impulse_42"]:
            assert result[col].dtype == np.float32

    def test_early_rows_nan(self, impulse_df):
        """First N rows per ticker should be NaN for impulse_N."""
        result = compute_rs_impulse(impulse_df)
        aapl = result[result["ticker"] == "AAPL"].sort_values("date")
        assert aapl["rs_impulse_5"].iloc[:5].isna().all()
        assert aapl["rs_impulse_10"].iloc[:10].isna().all()
        assert aapl["rs_impulse_20"].iloc[:20].isna().all()
        assert aapl["rs_impulse_42"].iloc[:42].isna().all()

    def test_impulse_values_correct(self, impulse_df):
        """With linear ranks, impulse should equal step_size * window."""
        result = compute_rs_impulse(impulse_df)
        aapl = result[result["ticker"] == "AAPL"].sort_values("date")
        step = 0.98 / 49  # linspace step
        # impulse_5 at row 5 = rank[5] - rank[0] = 5 * step
        assert aapl["rs_impulse_5"].iloc[5] == pytest.approx(5 * step, abs=1e-3)
        assert aapl["rs_impulse_10"].iloc[10] == pytest.approx(10 * step, abs=1e-3)

    def test_falling_ticker_negative_impulse(self, impulse_df):
        """MSFT has declining ranks, so impulse should be negative."""
        result = compute_rs_impulse(impulse_df)
        msft = result[result["ticker"] == "MSFT"].sort_values("date")
        non_nan = msft["rs_impulse_10"].dropna()
        assert (non_nan < 0).all()

    def test_row_count_preserved(self, impulse_df):
        result = compute_rs_impulse(impulse_df)
        assert len(result) == len(impulse_df)

    def test_per_ticker_independence(self, impulse_df):
        """Impulse should be computed independently per ticker."""
        result = compute_rs_impulse(impulse_df)
        aapl = result[result["ticker"] == "AAPL"].sort_values("date")
        msft = result[result["ticker"] == "MSFT"].sort_values("date")
        # AAPL rising, MSFT falling — impulse_10 should have opposite signs
        aapl_val = aapl["rs_impulse_10"].iloc[15]
        msft_val = msft["rs_impulse_10"].iloc[15]
        assert aapl_val > 0
        assert msft_val < 0


class TestRSImpulseIntegration:
    def test_build_backtest_includes_impulse(self, multi_ticker_ohlcv):
        """build_backtest_dataframe should add impulse columns when metrics provided."""
        dates = multi_ticker_ohlcv["date"].unique()
        metrics_rows = []
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            for d in dates:
                metrics_rows.append(
                    {
                        "date": d,
                        "ticker": ticker,
                        "rs_rank_overall": np.random.rand(),
                    }
                )
        metrics = pd.DataFrame(metrics_rows)
        result = build_backtest_dataframe(multi_ticker_ohlcv, metrics)
        for col in ["rs_impulse_5", "rs_impulse_10", "rs_impulse_20", "rs_impulse_42"]:
            assert col in result.columns

    def test_build_backtest_no_impulse_without_metrics(self, multi_ticker_ohlcv):
        """Without metrics, impulse columns should not be present."""
        result = build_backtest_dataframe(multi_ticker_ohlcv)
        assert "rs_impulse_10" not in result.columns

    def test_impulse_filter_evaluation(self, sample_indicators_df):
        """rs_impulse filters should work in evaluate_strategy."""
        # Add impulse column
        sample_indicators_df["rs_impulse_10"] = np.random.uniform(
            -0.2, 0.2, len(sample_indicators_df)
        ).astype(np.float32)
        strategy = Strategy(
            name="rising_impulse",
            filters=[FilterSpec(name="rs_impulse_10_min", value=0.10)],
        )
        result = evaluate_strategy(sample_indicators_df, strategy)
        expected = (sample_indicators_df["rs_impulse_10"] >= 0.10).sum()
        assert result.n_signals == expected

    def test_falling_impulse_filter(self, sample_indicators_df):
        """rs_impulse_10_max should select falling stocks."""
        sample_indicators_df["rs_impulse_10"] = np.random.uniform(
            -0.2, 0.2, len(sample_indicators_df)
        ).astype(np.float32)
        strategy = Strategy(
            name="falling_impulse",
            filters=[FilterSpec(name="rs_impulse_10_max", value=-0.10)],
        )
        result = evaluate_strategy(sample_indicators_df, strategy)
        expected = (sample_indicators_df["rs_impulse_10"] <= -0.10).sum()
        assert result.n_signals == expected

    def test_impulse_filters_in_filter_to_column(self):
        for name in [
            "rs_impulse_10_min",
            "rs_impulse_10_max",
            "rs_impulse_20_min",
            "rs_impulse_20_max",
        ]:
            assert name in FILTER_TO_COLUMN

    def test_impulse_filters_in_filter_ranges(self):
        for name in [
            "rs_impulse_10_min",
            "rs_impulse_10_max",
            "rs_impulse_20_min",
            "rs_impulse_20_max",
        ]:
            assert name in FILTER_RANGES
            assert len(FILTER_RANGES[name]) == 3

    def test_impulse_conflict_detection(self):
        """Same-window min+max should conflict."""
        assert {"rs_impulse_10_min", "rs_impulse_10_max"} in CONFLICTING_FILTERS
        assert {"rs_impulse_20_min", "rs_impulse_20_max"} in CONFLICTING_FILTERS

    def test_no_impulse_cross_window_conflict(self):
        """Different windows should NOT conflict."""
        strategies = generate_strategies(max_filters=2)
        cross_window = [
            s
            for s in strategies
            if any(f.name == "rs_impulse_10_min" for f in s.filters)
            and any(f.name == "rs_impulse_20_min" for f in s.filters)
        ]
        assert len(cross_window) > 0

    def test_generate_strategies_includes_impulse(self):
        """generate_strategies should produce impulse-based strategies."""
        strategies = generate_strategies(max_filters=1)
        impulse_strategies = [
            s for s in strategies if any("rs_impulse" in f.name for f in s.filters)
        ]
        # 4 filter names x 3 values x 5 regimes = 60
        assert len(impulse_strategies) == 60


# ---------------------------------------------------------------------------
# Single source of truth for indicators and filters
# ---------------------------------------------------------------------------

from stock_dashboard.logic.backtest import (  # noqa: E402
    INDICATOR_COLUMNS,
    compute_ticker_indicators,
    signal_mask,
)


class TestComputeTickerIndicators:
    @pytest.fixture
    def one_ticker(self, multi_ticker_ohlcv):
        df = multi_ticker_ohlcv
        return df[df["ticker"] == df["ticker"].iloc[0]].sort_values("date").reset_index(drop=True)

    def test_all_columns_by_default(self, one_ticker):
        out = compute_ticker_indicators(one_ticker)
        assert set(INDICATOR_COLUMNS).issubset(out.columns)

    def test_subset_matches_full_computation(self, one_ticker):
        full = compute_ticker_indicators(one_ticker)
        subset = compute_ticker_indicators(one_ticker, ["momentum_20", "dist_recent_low_63"])
        assert set(subset.columns) == {"momentum_20", "dist_recent_high_63", "dist_recent_low_63"}
        pd.testing.assert_series_equal(subset["momentum_20"], full["momentum_20"])

    def test_events_pull_in_their_inputs(self, one_ticker):
        out = compute_ticker_indicators(one_ticker, ["evt_regain_200ma"])
        assert {"rsi_14", "macd_histogram", "volume_z", "ratio_to_200ma"}.issubset(out.columns)

    def test_keeps_input_index(self, one_ticker):
        shifted = one_ticker.set_index(one_ticker.index + 100)
        assert compute_ticker_indicators(shifted, ["rsi_14"]).index.equals(shifted.index)


class TestSignalMask:
    def test_missing_column_matches_nothing(self):
        df = pd.DataFrame({"rsi_14": [50.0, 60.0]})
        assert not signal_mask(df, [FilterSpec("volume_z_min", 1.0)]).any()

    def test_nan_pattern_flag_does_not_match(self):
        df = pd.DataFrame({"has_bull_flag": [1.0, np.nan, 0.0]})
        assert signal_mask(df, [FilterSpec("has_bull_flag", True)]).tolist() == [True, False, False]

    def test_every_filter_maps_to_a_known_column(self):
        known = set(INDICATOR_COLUMNS) | {
            "rs_rank_overall",
            "rs_z_score",
            "rs_impulse_10",
            "rs_impulse_20",
        }
        assert set(FILTER_TO_COLUMN.values()) <= known
