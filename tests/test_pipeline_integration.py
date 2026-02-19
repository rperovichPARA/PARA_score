"""Integration test: full pipeline with synthetic data for 5 stocks.

Validates end-to-end scoring logic without J-Quants API access.
Uses mock financial, price, and TOPIX data for:
    7203 (Toyota), 6758 (Sony), 8306 (MUFG), 9984 (SoftBank), 6861 (Keyence)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.scoring.composite import compute_composite_scores
from src.scoring.factors import compute_factor_metrics
from src.scoring.fundamentals import compute_fundamentals_metrics
from src.scoring.kozo import compute_kozo_metrics
from src.scoring.sector import compute_sector_metrics
from src.scoring.valuation import compute_valuation_metrics
from src.export import format_payload

# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------

TEST_CODES = ["7203", "6758", "8306", "9984", "6861"]
TEST_NAMES = [
    "Toyota Motor Corp",
    "Sony Group Corp",
    "Mitsubishi UFJ Financial Group",
    "SoftBank Group Corp",
    "Keyence Corp",
]
TEST_SECTORS = ["3050", "3650", "3150", "3650", "3600"]


def _make_financials() -> pd.DataFrame:
    """Create synthetic financials for the 5-stock universe."""
    np.random.seed(42)
    n = len(TEST_CODES)
    return pd.DataFrame({
        "Code": TEST_CODES,
        "CompanyName": TEST_NAMES,
        "CompanyNameEnglish": TEST_NAMES,
        "Sector33Code": TEST_SECTORS,
        "MarketCode": ["0111"] * n,
        # Income statement
        "NetSales": [30_000_000, 12_000_000, 8_000_000, 6_500_000, 900_000],
        "OperatingProfit": [3_000_000, 1_200_000, 1_500_000, 800_000, 400_000],
        "Profit": [2_500_000, 900_000, 1_200_000, 500_000, 350_000],
        # Balance sheet
        "Equity": [25_000_000, 5_000_000, 18_000_000, 4_000_000, 1_500_000],
        "TotalAssets": [60_000_000, 30_000_000, 380_000_000, 45_000_000, 2_000_000],
        "EquityToAssetRatio": [41.7, 16.7, 4.7, 8.9, 75.0],
        # Per-share data
        "EarningsPerShare": [180.0, 73.0, 95.0, 34.0, 620.0],
        "BookValuePerShare": [1800.0, 405.0, 1400.0, 270.0, 2500.0],
        # Forecasts
        "ForecastOperatingProfit": [3_200_000, 1_300_000, 1_600_000, 900_000, 420_000],
        "ForecastEarningsPerShare": [195.0, 80.0, 100.0, 40.0, 660.0],
        "NextYearForecastNetSales": [32_000_000, 13_000_000, 8_500_000, 7_000_000, 980_000],
        "NextYearForecastOperatingProfit": [3_400_000, 1_400_000, 1_700_000, 1_000_000, 450_000],
        "NextYearForecastEarningsPerShare": [210.0, 88.0, 108.0, 45.0, 700.0],
        # Payout
        "ResultPayoutRatioAnnual": [30.0, 15.0, 40.0, 5.0, 20.0],
        "ForecastNetSales": [31_000_000, 12_500_000, 8_200_000, 6_800_000, 950_000],
    })


def _make_prices() -> pd.DataFrame:
    """Create synthetic 6-month daily prices for the 5 stocks."""
    np.random.seed(42)
    dates = pd.bdate_range("2025-08-19", periods=120, freq="B")
    base_prices = {"7203": 2400, "6758": 3000, "8306": 1500, "9984": 9000, "6861": 65000}
    records = []
    for code, base_px in base_prices.items():
        # Simulate price path with drift
        returns = np.random.normal(0.0003, 0.015, len(dates))
        prices = base_px * np.cumprod(1 + returns)
        volumes = np.random.randint(500_000, 5_000_000, len(dates))
        for i, date in enumerate(dates):
            records.append({
                "Code": code,
                "Date": date,
                "AdjustmentClose": round(prices[i], 1),
                "Volume": int(volumes[i]),
            })
    return pd.DataFrame(records)


def _make_topix() -> pd.DataFrame:
    """Create synthetic TOPIX index data."""
    dates = pd.bdate_range("2025-08-19", periods=120, freq="B")
    np.random.seed(99)
    returns = np.random.normal(0.0002, 0.008, len(dates))
    levels = 2700 * np.cumprod(1 + returns)
    return pd.DataFrame({
        "Date": dates,
        "Close": levels.round(2),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """End-to-end pipeline integration tests with 5-stock synthetic data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create test data available to all tests."""
        self.financials = _make_financials()
        self.prices = _make_prices()
        self.topix = _make_topix()

    def _mock_gsheet_empty(self):
        """Patch Google Sheets calls to return empty DataFrames."""
        return patch(
            "src.data.gsheets.GoogleSheetsClient._fetch_csv",
            return_value=pd.DataFrame(),
        )

    def test_fundamentals_metrics(self):
        """Fundamentals: growth, profitability, and balance sheet metrics."""
        with self._mock_gsheet_empty():
            result = compute_fundamentals_metrics(self.financials, self.prices)

        # All 5 companies should be present.
        assert len(result) == 5

        # J-Quants-derived metrics should be computed.
        for col in ["opm", "roe", "equity_ratio", "f1_f0_op_growth",
                     "f2_f1_op_growth", "f2_f0_sales_growth", "f2_f0_ebit_growth"]:
            assert col in result.columns, f"Missing metric: {col}"
            assert result[col].notna().sum() == 5, f"Missing data in {col}"

        # ROE sanity: Toyota Profit/Equity = 2.5M/25M = 0.1
        toyota_roe = result.loc[result["Code"] == "7203", "roe"].iloc[0]
        assert abs(toyota_roe - 0.1) < 0.001

    def test_valuation_metrics(self):
        """Valuation: PBR, PE, PEG, ADV, peer upside, TOPIX relative."""
        with self._mock_gsheet_empty():
            result = compute_valuation_metrics(
                self.financials, self.prices, topix=self.topix,
            )

        assert len(result) == 5

        for col in ["pbr_vs_10yr", "pen_vs_10yr", "pegn",
                     "adv_liquidity", "peer_target_upside", "price_6mo_vs_tpx"]:
            assert col in result.columns, f"Missing metric: {col}"
            assert result[col].notna().sum() > 0, f"No data in {col}"

        # ADV should be positive for all stocks.
        assert (result["adv_liquidity"] > 0).all()

    def test_sector_stub(self):
        """Sector: all metrics should be NaN (stub)."""
        result = compute_sector_metrics(self.financials)
        assert len(result) == 5
        for col in ["cyclical_momentum", "sector_trend_fund",
                     "sector_trend_val", "competitive_strength"]:
            assert col in result.columns
            assert result[col].isna().all()

    def test_factors_stub(self):
        """Factors: all metrics should be NaN (stub)."""
        result = compute_factor_metrics(self.financials)
        assert len(result) == 5
        for col in ["factor_fit", "theme_fit"]:
            assert col in result.columns
            assert result[col].isna().all()

    def test_kozo_metrics(self):
        """Kozo: ROE vs peer, payout ratio, NaN stubs."""
        with self._mock_gsheet_empty():
            # First compute fundamentals to get roe column.
            fund_df = compute_fundamentals_metrics(self.financials, self.prices)
            result = compute_kozo_metrics(fund_df)

        assert len(result) == 5

        # J-Quants computed metrics.
        assert "roe_vs_peer" in result.columns
        assert result["roe_vs_peer"].notna().sum() == 5
        assert "payout_ratio" in result.columns
        assert result["payout_ratio"].notna().sum() == 5

        # Unavailable metrics should be NaN.
        assert result["underperf_segments"].isna().all()
        assert result["sga_vs_peer"].isna().all()

    def test_composite_scoring(self):
        """Composite: VI and SP scores and ranks computed from all categories."""
        with self._mock_gsheet_empty():
            fund_df = compute_fundamentals_metrics(self.financials, self.prices)
            val_df = compute_valuation_metrics(
                self.financials, self.prices, topix=self.topix,
            )
            sector_df = compute_sector_metrics(self.financials)
            factors_df = compute_factor_metrics(self.financials)
            kozo_df = compute_kozo_metrics(fund_df)

        category_dfs = {
            "fundamentals": fund_df,
            "valuation": val_df,
            "sector": sector_df,
            "factors": factors_df,
            "kozo": kozo_df,
        }

        result = compute_composite_scores(category_dfs)

        assert len(result) == 5

        # Score columns present.
        for col in ["VI_score", "SP_score", "VI_rank", "SP_rank",
                     "fundamentals_score", "valuation_score",
                     "sector_score", "factors_score", "kozo_score"]:
            assert col in result.columns, f"Missing column: {col}"

        # Ranks should be 1..5 (no ties expected with distinct data).
        assert set(result["VI_rank"].astype(int)) == {1, 2, 3, 4, 5}
        assert set(result["SP_rank"].astype(int)) == {1, 2, 3, 4, 5}

        # VI and SP scores should not all be zero (fundamentals and
        # valuation have real data).
        assert result["VI_score"].abs().sum() > 0
        assert result["SP_score"].abs().sum() > 0

    def test_export_format(self):
        """Export: JSON payload matches the expected schema."""
        with self._mock_gsheet_empty():
            fund_df = compute_fundamentals_metrics(self.financials, self.prices)
            val_df = compute_valuation_metrics(
                self.financials, self.prices, topix=self.topix,
            )
            sector_df = compute_sector_metrics(self.financials)
            factors_df = compute_factor_metrics(self.financials)
            kozo_df = compute_kozo_metrics(fund_df)

        category_dfs = {
            "fundamentals": fund_df,
            "valuation": val_df,
            "sector": sector_df,
            "factors": factors_df,
            "kozo": kozo_df,
        }
        composite = compute_composite_scores(category_dfs)

        # Add company info columns as pipeline does.
        for col in ["CompanyNameEnglish", "Sector33Code"]:
            if col in self.financials.columns and col not in composite.columns:
                mapping = self.financials.set_index(
                    self.financials["Code"].str.strip()
                )[col]
                if "Code" in composite.columns:
                    composite[col] = composite["Code"].str.strip().map(mapping)

        payload = format_payload(composite, category_dfs)

        # Top-level keys.
        assert "scores" in payload
        assert "metadata" in payload
        assert "coverage" in payload

        # Scores array.
        assert len(payload["scores"]) == 5
        first = payload["scores"][0]
        assert "code" in first
        assert "name" in first
        assert "sector" in first
        assert "VI_score" in first
        assert "SP_score" in first
        assert "VI_rank" in first
        assert "SP_rank" in first

        # Metadata.
        assert payload["metadata"]["universe_size"] == 5
        assert payload["metadata"]["pipeline_version"] == "0.1.0"
        assert "run_timestamp" in payload["metadata"]

        # Coverage should have some metrics.
        assert len(payload["coverage"]) > 0
        for metric, stats in payload["coverage"].items():
            assert "count" in stats
            assert "pct" in stats

    def test_export_dry_run_writes_file(self):
        """Export dry-run: payload written to local JSON file."""
        with self._mock_gsheet_empty():
            fund_df = compute_fundamentals_metrics(self.financials, self.prices)
            val_df = compute_valuation_metrics(
                self.financials, self.prices, topix=self.topix,
            )
            sector_df = compute_sector_metrics(self.financials)
            factors_df = compute_factor_metrics(self.financials)
            kozo_df = compute_kozo_metrics(fund_df)

        category_dfs = {
            "fundamentals": fund_df,
            "valuation": val_df,
            "sector": sector_df,
            "factors": factors_df,
            "kozo": kozo_df,
        }
        composite = compute_composite_scores(category_dfs)
        payload = format_payload(composite, category_dfs)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(payload, f, indent=2)
            tmp_path = f.name

        try:
            with open(tmp_path, "r") as f:
                loaded = json.load(f)
            assert len(loaded["scores"]) == 5
            assert loaded["metadata"]["universe_size"] == 5
        finally:
            os.unlink(tmp_path)


class TestScoringLogic:
    """Unit tests for scoring edge cases."""

    def test_winsorized_zscore_preserves_nan(self):
        """Winsorized z-score preserves NaN positions."""
        from src.scoring.utils import winsorized_zscore
        s = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10])
        result = winsorized_zscore(s)
        assert result.isna().sum() == 1
        assert pd.isna(result.iloc[2])

    def test_score_category_renormalises(self):
        """score_category renormalises when available weights < 1."""
        from src.scoring.utils import score_category
        df = pd.DataFrame({
            "metric_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            # metric_b is all NaN — should be skipped
            "metric_b": [np.nan] * 5,
        })
        metric_defs = {
            "metric_a": (0.5, True),
            "metric_b": (0.5, True),
        }
        result = score_category(df, metric_defs, min_coverage=0.05)
        # metric_b skipped; metric_a weight 0.5 < 1.0, so result is
        # renormalised by dividing by 0.5.
        assert result.abs().sum() > 0

    def test_safe_ratio_guards_zero(self):
        """_safe_ratio returns NaN for zero/negative denominators."""
        from src.scoring.valuation import _safe_ratio
        num = pd.Series([10, 20, 30])
        denom = pd.Series([5, 0, -1])
        result = _safe_ratio(num, denom)
        assert result.iloc[0] == 2.0
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])
