"""Pipeline coverage analysis.

Generates realistic mock data simulating J-Quants API + Google Sheets
supplement responses, then runs the full scoring pipeline and reports
detailed per-metric coverage statistics.

Usage::

    python scripts/coverage_analysis.py
    python scripts/coverage_analysis.py --universe-size 1800
    python scripts/coverage_analysis.py --no-sheets   # without supplement
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring.composite import CATEGORY_NAMES, compute_composite_scores
from src.scoring.factors import compute_factor_metrics
from src.scoring.fundamentals import compute_fundamentals_metrics
from src.scoring.kozo import compute_kozo_metrics
from src.scoring.sector import compute_sector_metrics
from src.scoring.valuation import compute_valuation_metrics
from src.scoring.utils import load_config

logger = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "config" / "scoring_weights.yaml"


# ---------------------------------------------------------------------------
# Mock data generators
# ---------------------------------------------------------------------------

def _generate_universe(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic listed-company universe."""
    rng = np.random.default_rng(seed)

    # Real TOPIX Sector33 codes (33 sectors)
    sector_codes = [
        "0050", "1050", "2050", "3050", "3100", "3150", "3200", "3250",
        "3300", "3350", "3400", "3450", "3500", "3550", "3600", "3650",
        "3700", "3750", "3800", "4050", "5050", "5100", "5150", "5200",
        "5250", "6050", "6100", "7050", "7100", "7150", "7200", "8050",
        "9050",
    ]

    codes = [str(10000 + i) for i in range(n)]
    return pd.DataFrame({
        "Code": codes,
        "CompanyName": [f"Company_{c}" for c in codes],
        "CompanyNameEnglish": [f"Company {c} Inc." for c in codes],
        "Sector33Code": rng.choice(sector_codes, size=n),
        "MarketCode": "0111",
    })


def _generate_financials(
    codes: list[str],
    sector_codes: pd.Series,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic J-Quants financial statement data.

    Coverage patterns mirror real J-Quants Standard-plan availability:
    - Core financials (NetSales, OperatingProfit, etc.): ~90%
    - Forecasts (F1, F2): ~80%
    - Per-share data: ~85%
    - Equity/payout: ~90%
    """
    rng = np.random.default_rng(seed)
    n = len(codes)

    # Base coverage masks (True = data present)
    core_mask = rng.random(n) < 0.90
    forecast_f1_mask = rng.random(n) < 0.80
    forecast_f2_mask = rng.random(n) < 0.75
    pershare_mask = rng.random(n) < 0.85
    equity_mask = rng.random(n) < 0.90
    payout_mask = rng.random(n) < 0.85

    def _optional(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = values.copy().astype(float)
        result[~mask] = np.nan
        return result

    net_sales = _optional(rng.lognormal(10, 1.5, n), core_mask)
    op = _optional(net_sales * rng.uniform(0.02, 0.25, n), core_mask)
    profit = _optional(op * rng.uniform(0.5, 0.9, n), core_mask)
    equity = _optional(rng.lognormal(9, 1.5, n), equity_mask)
    total_assets = _optional(equity * rng.uniform(1.5, 5.0, n), equity_mask)

    forecast_op = _optional(op * rng.uniform(0.9, 1.3, n), forecast_f1_mask)
    f2_op = _optional(op * rng.uniform(0.85, 1.5, n), forecast_f2_mask)
    f1_sales = _optional(net_sales * rng.uniform(0.95, 1.15, n), forecast_f1_mask)
    f2_sales = _optional(net_sales * rng.uniform(0.90, 1.25, n), forecast_f2_mask)

    eps = _optional(rng.lognormal(4, 1, n), pershare_mask)
    bvps = _optional(rng.lognormal(6, 1, n), pershare_mask)
    f1_eps = _optional(eps * rng.uniform(0.9, 1.3, n), forecast_f1_mask & pershare_mask)
    f2_eps = _optional(eps * rng.uniform(0.85, 1.5, n), forecast_f2_mask & pershare_mask)

    shares = _optional(rng.lognormal(16, 1, n), core_mask)

    df = pd.DataFrame({
        "Code": codes,
        "DisclosedDate": "2026-01-15",
        "NetSales": net_sales,
        "OperatingProfit": op,
        "OrdinaryProfit": _optional(op * rng.uniform(0.95, 1.05, n), core_mask),
        "Profit": profit,
        "Equity": equity,
        "TotalAssets": total_assets,
        "EquityToAssetRatio": _optional(rng.uniform(20, 80, n), equity_mask),
        "EarningsPerShare": eps,
        "BookValuePerShare": bvps,
        "ForecastNetSales": f1_sales,
        "ForecastOperatingProfit": forecast_op,
        "ForecastOrdinaryProfit": _optional(forecast_op * rng.uniform(0.95, 1.05, n), forecast_f1_mask),
        "ForecastProfit": _optional(profit * rng.uniform(0.9, 1.2, n), forecast_f1_mask),
        "ForecastEarningsPerShare": f1_eps,
        "NextYearForecastNetSales": f2_sales,
        "NextYearForecastOperatingProfit": f2_op,
        "NextYearForecastOrdinaryProfit": _optional(f2_op * rng.uniform(0.95, 1.05, n), forecast_f2_mask),
        "NextYearForecastProfit": _optional(profit * rng.uniform(0.85, 1.3, n), forecast_f2_mask),
        "NextYearForecastEarningsPerShare": f2_eps,
        "ResultDividendPerShareAnnual": _optional(rng.uniform(10, 200, n), payout_mask),
        "ResultPayoutRatioAnnual": _optional(rng.uniform(10, 80, n), payout_mask),
        "ForecastDividendPerShareAnnual": _optional(rng.uniform(10, 200, n), forecast_f1_mask),
        "ForecastPayoutRatioAnnual": _optional(rng.uniform(10, 80, n), forecast_f1_mask),
        "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock": shares,
        "Sector33Code": sector_codes.values,
    })

    df = df.set_index(df["Code"].astype(str).str.strip(), drop=False)
    return df


def _generate_prices(codes: list[str], days: int = 130, seed: int = 42) -> pd.DataFrame:
    """Generate realistic daily price data for 6 months."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-02-18", periods=days)
    rows = []

    for code in codes:
        if rng.random() < 0.03:
            continue  # 3% of stocks have no price data

        base_price = rng.lognormal(6.5, 1)
        returns = rng.normal(0.0003, 0.02, days)
        prices = base_price * np.cumprod(1 + returns)
        volumes = rng.lognormal(12, 1.5, days).astype(int)

        for i, date in enumerate(dates):
            rows.append({
                "Date": date,
                "Code": code,
                "Open": prices[i] * rng.uniform(0.99, 1.01),
                "High": prices[i] * rng.uniform(1.0, 1.03),
                "Low": prices[i] * rng.uniform(0.97, 1.0),
                "Close": prices[i],
                "AdjustmentClose": prices[i],
                "Volume": volumes[i],
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values(["Code", "Date"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def _generate_topix(days: int = 130, seed: int = 42) -> pd.DataFrame:
    """Generate realistic TOPIX index data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-02-18", periods=days)
    base = 2700.0
    returns = rng.normal(0.0002, 0.008, days)
    levels = base * np.cumprod(1 + returns)

    return pd.DataFrame({
        "Date": dates,
        "Open": levels * rng.uniform(0.999, 1.001, days),
        "High": levels * rng.uniform(1.0, 1.005, days),
        "Low": levels * rng.uniform(0.995, 1.0, days),
        "Close": levels,
    })


def _generate_gsheet_supplement(
    codes: list[str],
    include_sheets: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic Google Sheets supplement data.

    Coverage patterns mirror real PARA.FS.data availability:
    - altman_z: ~30% (Bloomberg sourced)
    - net_cash_mktcap: ~40% (Premium J-Quants)
    - broker_target_upside: ~15% (Bloomberg BEST)
    - para_target_upside: ~25% (proprietary model)
    - board_size: ~20%
    - analyst_coverage: ~15%
    - excess_cash_mktcap: ~25%
    - lti_mktcap: ~20%
    - land_mktcap: ~15%
    - ke (cost of equity): ~20%
    """
    if not include_sheets:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    n = len(codes)

    def _optional(values: np.ndarray, rate: float) -> np.ndarray:
        result = values.copy().astype(float)
        result[rng.random(n) >= rate] = np.nan
        return result

    df = pd.DataFrame({
        "Code": codes,
        # Fundamentals supplement
        "altman_z": _optional(rng.uniform(0.5, 5.0, n), 0.30),
        "net_cash_mktcap": _optional(rng.uniform(-0.3, 0.5, n), 0.40),
        # Valuation supplement
        "pbr_vs_10yr": _optional(rng.uniform(0.5, 2.5, n), 0.20),
        "pen_vs_10yr": _optional(rng.uniform(0.5, 3.0, n), 0.20),
        "pegn": _optional(rng.uniform(0.3, 3.0, n), 0.15),
        "broker_target_upside": _optional(rng.uniform(-0.2, 0.5, n), 0.15),
        "para_target_upside": _optional(rng.uniform(-0.3, 0.8, n), 0.25),
        "peer_target_upside": _optional(rng.uniform(-0.3, 0.5, n), 0.15),
        # Kozo supplement
        "excess_cash_mktcap": _optional(rng.uniform(0, 0.4, n), 0.25),
        "lti_mktcap": _optional(rng.uniform(0, 0.3, n), 0.20),
        "land_mktcap": _optional(rng.uniform(0, 0.2, n), 0.15),
        "board_size": _optional(rng.integers(5, 20, n).astype(float), 0.20),
        "analyst_coverage": _optional(rng.integers(0, 25, n).astype(float), 0.15),
        "ke": _optional(rng.uniform(0.04, 0.12, n), 0.20),
    })

    df["Code"] = df["Code"].astype(str).str.strip()
    df = df.set_index("Code")
    return df


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------

def print_coverage_report(
    category_dfs: dict[str, pd.DataFrame],
    config: dict[str, Any],
    results: pd.DataFrame,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Print detailed coverage statistics for every metric in every category.

    Returns a nested dict: {category: {metric: {count, pct, weight, direction}}}.
    """
    categories = ["fundamentals", "valuation", "sector", "factors", "kozo"]
    all_stats: dict[str, dict[str, dict[str, Any]]] = {}

    total_metrics = 0
    total_active = 0
    total_above_gate = 0

    print("\n" + "=" * 90)
    print("PARA SCORE PIPELINE — METRIC COVERAGE ANALYSIS")
    print("=" * 90)

    for cat in categories:
        cat_df = category_dfs.get(cat, pd.DataFrame())
        cat_config = config.get(cat, {})
        n_companies = len(cat_df)

        print(f"\n{'─' * 90}")
        print(f"  {cat.upper()}  ({n_companies} companies)")
        print(f"{'─' * 90}")
        print(f"  {'Metric':<28} {'Count':>7} {'/ Total':>8} {'Coverage':>10} {'Weight':>8} {'Direction':>12} {'Gate?':>7}")
        print(f"  {'─'*28} {'─'*7} {'─'*8} {'─'*10} {'─'*8} {'─'*12} {'─'*7}")

        cat_stats: dict[str, dict[str, Any]] = {}
        cat_weight_sum = 0.0
        cat_active_weight = 0.0

        for metric_name, spec in cat_config.items():
            weight = float(spec[0])
            higher_is_better = bool(spec[1])
            direction = "higher" if higher_is_better else "lower"
            total_metrics += 1

            if metric_name in cat_df.columns:
                non_null = int(cat_df[metric_name].notna().sum())
                pct = non_null / n_companies * 100 if n_companies > 0 else 0.0
                passes_gate = pct >= 5.0
            else:
                non_null = 0
                pct = 0.0
                passes_gate = False

            if pct > 0:
                total_active += 1
            if passes_gate:
                total_above_gate += 1
                cat_active_weight += weight

            cat_weight_sum += weight
            gate_str = "PASS" if passes_gate else "SKIP"

            cat_stats[metric_name] = {
                "count": non_null,
                "total": n_companies,
                "pct": pct,
                "weight": weight,
                "direction": direction,
                "passes_gate": passes_gate,
            }

            # Color-code coverage
            if pct >= 80:
                bar = "█" * int(pct / 5)
            elif pct >= 30:
                bar = "▓" * int(pct / 5)
            elif pct > 0:
                bar = "░" * max(1, int(pct / 5))
            else:
                bar = "·"

            print(
                f"  {metric_name:<28} {non_null:>7,} {f'/ {n_companies}':>8} "
                f"{pct:>8.1f}%  {weight:>6.0%} {direction:>12}  {gate_str:>5}"
            )

        # Category summary
        print(f"  {'─'*28} {'─'*7} {'─'*8} {'─'*10} {'─'*8} {'─'*12} {'─'*7}")
        active_pct = cat_active_weight / cat_weight_sum * 100 if cat_weight_sum > 0 else 0
        print(f"  Category weight active: {cat_active_weight:.0%} / {cat_weight_sum:.0%}  ({active_pct:.0f}% of category weight has data)")

        all_stats[cat] = cat_stats

    # ── Composite score summary ──
    print(f"\n{'=' * 90}")
    print("COMPOSITE SCORE SUMMARY")
    print(f"{'=' * 90}")

    cat_weights = config.get("category_weights", {})
    for composite in ("VI", "SP"):
        weights = cat_weights.get(composite, {})
        print(f"\n  {composite} Score:")
        total_effective = 0.0
        for cat, w in weights.items():
            w = float(w)
            if w == 0:
                print(f"    {cat:<20} weight={w:.0%}  (excluded)")
                continue
            cat_stats = all_stats.get(cat, {})
            active_metrics = sum(1 for s in cat_stats.values() if s["passes_gate"])
            total_cat_metrics = len(cat_stats)
            active_w = sum(s["weight"] for s in cat_stats.values() if s["passes_gate"])
            total_w = sum(s["weight"] for s in cat_stats.values())
            eff_rate = active_w / total_w if total_w > 0 else 0
            effective_composite_w = w * eff_rate
            total_effective += effective_composite_w
            print(
                f"    {cat:<20} weight={w:.0%}  "
                f"active_metrics={active_metrics}/{total_cat_metrics}  "
                f"effective={effective_composite_w:.1%}"
            )
        print(f"    {'Total effective weight:':<41} {total_effective:.1%}")

    # ── Global summary ──
    print(f"\n{'=' * 90}")
    print("GLOBAL SUMMARY")
    print(f"{'=' * 90}")
    print(f"  Total metrics defined:        {total_metrics}")
    print(f"  Metrics with any data:        {total_active} ({total_active/total_metrics*100:.0f}%)")
    print(f"  Metrics passing 5% gate:      {total_above_gate} ({total_above_gate/total_metrics*100:.0f}%)")
    print(f"  Metrics with zero coverage:   {total_metrics - total_active}")

    # ── Score distribution ──
    if not results.empty:
        print(f"\n{'─' * 90}")
        print("SCORE DISTRIBUTIONS")
        print(f"{'─' * 90}")
        for col in ["fundamentals_score", "valuation_score", "sector_score",
                     "factors_score", "kozo_score", "VI_score", "SP_score"]:
            if col in results.columns:
                s = results[col]
                non_zero = (s != 0).sum()
                print(
                    f"  {col:<25}  "
                    f"mean={s.mean():>7.3f}  std={s.std():>7.3f}  "
                    f"min={s.min():>7.3f}  max={s.max():>7.3f}  "
                    f"non-zero={non_zero}/{len(s)}"
                )

    print(f"\n{'=' * 90}\n")
    return all_stats


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

# Expected baseline coverage without supplement data
# (J-Quants only, no Google Sheets)
BASELINE_NO_SHEETS: dict[str, dict[str, float]] = {
    "fundamentals": {
        "f2_f0_sales_growth": 67.5,
        "f2_f0_ebit_growth": 67.5,
        "f1_f0_op_growth": 72.0,
        "f2_f1_op_growth": 60.0,
        "opm": 81.0,
        "roe": 81.0,
        "equity_ratio": 81.0,
        "net_cash_mktcap": 0.0,
        "altman_z": 0.0,
    },
    "valuation": {
        "pbr_vs_10yr": 79.0,
        "pen_vs_10yr": 60.0,
        "pegn": 40.0,
        "adv_liquidity": 93.0,
        "broker_target_upside": 0.0,
        "peer_target_upside": 55.0,
        "price_6mo_vs_tpx": 93.0,
        "para_target_upside": 0.0,
    },
    "sector": {m: 0.0 for m in ["cyclical_momentum", "sector_trend_fund", "sector_trend_val", "competitive_strength"]},
    "factors": {"factor_fit": 0.0, "theme_fit": 0.0},
    "kozo": {
        "underperf_segments": 0.0,
        "sga_vs_peer": 0.0,
        "roe_vs_peer": 81.0,
        "excess_cash_mktcap": 0.0,
        "lti_mktcap": 0.0,
        "land_mktcap": 0.0,
        "payout_ratio": 76.5,
        "analyst_coverage": 0.0,
        "ke_vs_peer": 0.0,
        "board_size": 0.0,
    },
}


def print_baseline_comparison(
    stats: dict[str, dict[str, dict[str, Any]]],
    baseline: dict[str, dict[str, float]],
) -> None:
    """Compare actual coverage against baseline expectations."""
    print(f"\n{'=' * 90}")
    print("BASELINE COMPARISON  (actual vs expected J-Quants-only baseline)")
    print(f"{'=' * 90}")
    print(f"  {'Category':<15} {'Metric':<28} {'Actual':>8} {'Baseline':>10} {'Delta':>8} {'Status'}")
    print(f"  {'─'*15} {'─'*28} {'─'*8} {'─'*10} {'─'*8} {'─'*20}")

    significant_divergences: list[tuple[str, str, float, float]] = []

    for cat in ["fundamentals", "valuation", "sector", "factors", "kozo"]:
        cat_stats = stats.get(cat, {})
        cat_baseline = baseline.get(cat, {})

        for metric, base_pct in cat_baseline.items():
            actual = cat_stats.get(metric, {}).get("pct", 0.0)
            delta = actual - base_pct

            if abs(delta) > 15:
                status = "** SIGNIFICANT **"
                significant_divergences.append((cat, metric, actual, base_pct))
            elif abs(delta) > 5:
                status = "* notable"
            elif actual > 0 and base_pct == 0:
                status = "+ NEW DATA"
                significant_divergences.append((cat, metric, actual, base_pct))
            else:
                status = "ok"

            print(
                f"  {cat:<15} {metric:<28} {actual:>7.1f}% {base_pct:>9.1f}% {delta:>+7.1f}%  {status}"
            )

    if significant_divergences:
        print(f"\n{'─' * 90}")
        print("SIGNIFICANT DIVERGENCES (>15pp or new data source):")
        print(f"{'─' * 90}")
        for cat, metric, actual, base in significant_divergences:
            direction = "IMPROVED" if actual > base else "DEGRADED"
            print(f"  [{cat}] {metric}: {base:.1f}% -> {actual:.1f}%  ({direction})")
            if base == 0 and actual > 0:
                print(f"    -> New data source detected (likely Google Sheets supplement)")
            elif actual > base + 15:
                print(f"    -> Check if supplement is filling gaps beyond expected rate")
            elif actual < base - 15:
                print(f"    -> Possible scoring bug: fewer valid values than expected")
    else:
        print("\n  No significant divergences from baseline.")

    print(f"\n{'=' * 90}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_coverage_analysis(
    universe_size: int = 1800,
    include_sheets: bool = True,
    seed: int = 42,
) -> None:
    """Run the full pipeline with mock data and report coverage."""

    config = load_config(str(CONFIG_PATH))

    print(f"\nGenerating mock data: {universe_size} companies, sheets={'ON' if include_sheets else 'OFF'}")

    # 1. Generate mock data
    universe = _generate_universe(universe_size, seed=seed)
    codes = universe["Code"].tolist()
    sector_codes = universe["Sector33Code"]

    financials = _generate_financials(codes, sector_codes, seed=seed)
    prices = _generate_prices(codes, days=130, seed=seed)
    topix = _generate_topix(days=130, seed=seed)
    supplement = _generate_gsheet_supplement(codes, include_sheets=include_sheets, seed=seed)

    print(f"  Universe:    {len(universe)} companies")
    print(f"  Financials:  {len(financials)} rows")
    print(f"  Prices:      {len(prices)} rows ({prices['Code'].nunique() if not prices.empty else 0} stocks)")
    print(f"  TOPIX:       {len(topix)} rows")
    print(f"  Supplement:  {len(supplement)} rows, {len(supplement.columns)} columns")

    # 2. Patch supplement loaders so they return our mock data
    def _mock_fund_supplement(url=None):
        if not include_sheets or supplement.empty:
            return pd.DataFrame()
        from src.scoring.fundamentals import _SUPPLEMENT_METRIC_COLS
        keep = [c for c in supplement.columns if c in _SUPPLEMENT_METRIC_COLS]
        return supplement[keep] if keep else pd.DataFrame()

    def _mock_val_supplement(url=None):
        if not include_sheets or supplement.empty:
            return pd.DataFrame()
        from src.scoring.valuation import _SUPPLEMENT_METRIC_COLS
        keep = [c for c in supplement.columns if c in _SUPPLEMENT_METRIC_COLS]
        return supplement[keep] if keep else pd.DataFrame()

    def _mock_kozo_supplement():
        if not include_sheets or supplement.empty:
            return pd.DataFrame()
        # Return the full supplement (kozo needs raw 'ke' which is non-metric)
        # Add Ke as 'ke' column name (already present from generation)
        return supplement

    # 3. Run metric computations with patched supplements
    with (
        patch("src.scoring.fundamentals.load_gsheet_supplement", _mock_fund_supplement),
        patch("src.scoring.valuation.load_gsheet_supplement", _mock_val_supplement),
        patch("src.scoring.kozo._load_kozo_supplement", _mock_kozo_supplement),
    ):
        print("\nComputing metrics...")
        fundamentals_df = compute_fundamentals_metrics(financials, prices)
        valuation_df = compute_valuation_metrics(financials, prices, topix=topix)
        sector_df = compute_sector_metrics(financials)
        factors_df = compute_factor_metrics(financials)
        kozo_df = compute_kozo_metrics(fundamentals_df)

    category_dfs = {
        "fundamentals": fundamentals_df,
        "valuation": valuation_df,
        "sector": sector_df,
        "factors": factors_df,
        "kozo": kozo_df,
    }

    # 4. Composite scoring
    print("Computing composite scores...")
    results = compute_composite_scores(category_dfs, str(CONFIG_PATH))

    # 5. Coverage report
    stats = print_coverage_report(category_dfs, config, results)

    # 6. Baseline comparison
    print_baseline_comparison(stats, BASELINE_NO_SHEETS)

    # 7. Run again without sheets for a direct comparison
    if include_sheets:
        print("\n" + "#" * 90)
        print("RE-RUNNING WITHOUT SHEETS (J-Quants only baseline)")
        print("#" * 90)

        with (
            patch("src.scoring.fundamentals.load_gsheet_supplement", lambda url=None: pd.DataFrame()),
            patch("src.scoring.valuation.load_gsheet_supplement", lambda url=None: pd.DataFrame()),
            patch("src.scoring.kozo._load_kozo_supplement", lambda: pd.DataFrame()),
        ):
            fund_no_sheets = compute_fundamentals_metrics(financials, prices)
            val_no_sheets = compute_valuation_metrics(financials, prices, topix=topix)
            sector_no_sheets = compute_sector_metrics(financials)
            factors_no_sheets = compute_factor_metrics(financials)
            kozo_no_sheets = compute_kozo_metrics(fund_no_sheets)

        cat_dfs_no = {
            "fundamentals": fund_no_sheets,
            "valuation": val_no_sheets,
            "sector": sector_no_sheets,
            "factors": factors_no_sheets,
            "kozo": kozo_no_sheets,
        }

        results_no = compute_composite_scores(cat_dfs_no, str(CONFIG_PATH))
        stats_no = print_coverage_report(cat_dfs_no, config, results_no)

        # Print sheets impact summary
        print(f"\n{'=' * 90}")
        print("SHEETS IMPACT: Coverage deltas (with sheets - without sheets)")
        print(f"{'=' * 90}")
        print(f"  {'Category':<15} {'Metric':<28} {'No Sheets':>10} {'With Sheets':>12} {'Delta':>8}")
        print(f"  {'─'*15} {'─'*28} {'─'*10} {'─'*12} {'─'*8}")

        for cat in ["fundamentals", "valuation", "kozo"]:
            cat_with = stats.get(cat, {})
            cat_without = stats_no.get(cat, {})
            for metric in cat_with:
                pct_with = cat_with[metric]["pct"]
                pct_without = cat_without.get(metric, {}).get("pct", 0.0)
                delta = pct_with - pct_without
                if abs(delta) > 0.1:
                    print(
                        f"  {cat:<15} {metric:<28} {pct_without:>9.1f}% {pct_with:>11.1f}% {delta:>+7.1f}%"
                    )

        # Score divergence check
        print(f"\n{'─' * 90}")
        print("SCORE DIVERGENCE: With sheets vs without sheets")
        print(f"{'─' * 90}")
        for col in ["VI_score", "SP_score"]:
            if col in results.columns and col in results_no.columns:
                corr = results[col].corr(results_no[col])
                diff = (results[col] - results_no[col]).abs()
                print(
                    f"  {col}: correlation={corr:.4f}  "
                    f"mean_abs_diff={diff.mean():.4f}  max_abs_diff={diff.max():.4f}"
                )

                # Rank changes
                rank_col = col.replace("_score", "_rank")
                if rank_col in results.columns and rank_col in results_no.columns:
                    rank_diff = (results[rank_col] - results_no[rank_col]).abs()
                    big_moves = (rank_diff > 100).sum()
                    print(
                        f"  {rank_col}: mean_rank_change={rank_diff.mean():.1f}  "
                        f"max_rank_change={rank_diff.max():.0f}  "
                        f"stocks_moving_>100_ranks={big_moves}"
                    )

        print(f"\n{'=' * 90}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="PARA Score pipeline coverage analysis")
    parser.add_argument("--universe-size", type=int, default=1800,
                        help="Number of companies in mock universe (default: 1800)")
    parser.add_argument("--no-sheets", action="store_true",
                        help="Run without Google Sheets supplement data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", type=str, default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: WARNING)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_coverage_analysis(
        universe_size=args.universe_size,
        include_sheets=not args.no_sheets,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
