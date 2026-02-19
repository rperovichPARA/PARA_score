# CLAUDE.md — PARA Score Model

## Project Overview

This is **Paradaim Capital's** proprietary stock scoring system for Japanese equities. It produces two composite scores for each company in the TOPIX universe (~3,700+ stocks):

- **VI (Value Impact) Score** — Measures the magnitude of potential value creation through alliance engagement. Emphasizes fundamentals, valuation upside, and structural improvement ("kozo") opportunities.
- **SP (Stock Pick) Score** — Measures overall attractiveness as a portfolio holding. Balances fundamentals, valuation, sector dynamics, and factor/theme alignment.

The system is built on the **PARA Cycle** framework (Partner → Architect → Reform → Amplify) that underpins Paradaim's alliance investing methodology in Japan.

---

## Scoring Architecture

### Five Scoring Categories

Each company is scored across five categories. Within each category, individual metrics are winsorized z-scored cross-sectionally, direction-adjusted (higher_better / lower_better), and weighted to produce a category score. Category scores are then weighted differently for VI vs SP.

#### Category Weights

| Category              | VI Weight | SP Weight |
|-----------------------|-----------|-----------|
| Fundamentals          | 0.25      | 0.25      |
| Valuation             | 0.25      | 0.25      |
| Sector Attractiveness | 0.00      | 0.15      |
| Factors / Themes      | 0.20      | 0.20      |
| Value Impact (Kozo)   | 0.30      | 0.15      |

> ⚠️ **IMPORTANT:** These weights should be verified against Ryan's latest preferred values. The key structural points are: Sector gets 0% in VI (it's irrelevant to value creation potential), and Kozo gets the highest VI weight (structural improvement = the core of alliance investing).

---

### 1. Fundamentals (strongest data coverage)

| Metric              | Weight | Direction    | Notes |
|---------------------|--------|-------------|-------|
| F2/F0 Sales growth  | 5%     | higher_better | `NextYearForecastNetSales` ÷ `NetSales` |
| F2/F0 EBIT growth   | 10%    | higher_better | `NextYearForecastOperatingProfit` ÷ `OperatingProfit` (OP ≈ EBIT for Japan) |
| F1/F0 OP growth     | 10%    | higher_better | `ForecastOperatingProfit` ÷ actual `OperatingProfit` |
| F2/F1 OP growth     | 15%    | higher_better | `NextYearForecastOperatingProfit` ÷ `ForecastOperatingProfit` |
| OPM                 | 15%    | higher_better | `OperatingProfit` ÷ `NetSales` |
| ROE                 | 20%    | higher_better | `Profit` ÷ `Equity` |
| Equity Ratio        | 5%     | higher_better | `EquityToAssetRatio` (provided directly) |
| Net Cash / Mkt Cap  | 10%    | higher_better | ⚠️ Partial — need Premium `/fins/fs_details` for proper net cash |
| Altman Z-Score      | 10%    | higher_better | ❌ Not available from J-Quants summary; needs external source |

**J-Quants coverage: ~80%**

### 2. Valuation

| Metric              | Weight | Direction    | Notes |
|---------------------|--------|-------------|-------|
| PBR vs 10yr avg     | 10%    | lower_better  | Price ÷ `BookValuePerShare` vs trailing average |
| PEn vs 10yr avg     | 10%    | lower_better  | Price ÷ normalized EPS vs trailing average |
| PEGn                | 10%    | lower_better  | PE ÷ forecast earnings growth |
| ADV liquidity       | 15%    | higher_better | `AdjustmentClose` × `Volume`, trailing average |
| Broker target price | 5%     | higher_better | ❌ Needs Bloomberg BEST / IBES / Nikkei consensus |
| Peer target price   | 5%     | higher_better | ⚠️ Partial — can compute avg peer PE × company F2 EPS |
| Price 6mo vs TPX    | 5%     | higher_better | Daily bars + `/indices/topix` |
| PARA target price   | 40%    | higher_better | Proprietary model — not a data sourcing question |

**J-Quants coverage: ~55%** (PARA target is proprietary; broker targets need Bloomberg)

### 3. Sector Attractiveness (SP only — 0% VI weight)

| Metric                     | Weight | Direction    | Notes |
|----------------------------|--------|-------------|-------|
| Cyclical momentum          | 20%    | higher_better | ❌ Needs external macro data |
| Sector Trend - Fundamentals| 30%    | higher_better | ⚠️ Aggregated OP growth/ROE/margins by TOPIX sector |
| Sector Trend - Valuation   | 20%    | higher_better | ⚠️ Sector-level PE/PBR aggregates |
| Competitive strength       | 30%    | higher_better | ❌ Qualitative |

**J-Quants coverage: ~25%**

A separate **sector rotation framework** exists using multi-window z-scored relative strength against TOPIX with TOPIX-17 sectors. This produces RRG-style quadrant classifications (Leading, Weakening, Lagging, Improving) and factor-adjusted alpha signals. It should feed into the sector scoring here.

### 4. Factors / Themes

| Metric     | Weight | Direction    | Notes |
|------------|--------|-------------|-------|
| Factor fit | 50%    | higher_better | ⚠️ Size, value/growth, dividend yield factor exposures derivable |
| Theme fit  | 50%    | higher_better | ❌ AI chain, ESG, takeover candidates — qualitative tagging |

**J-Quants coverage: ~25%**. Factor fit is a placeholder (0.0) in current code.

### 5. Value Impact Drivers (Kozo) — core of alliance investing

| Metric                    | Weight | Direction    | Notes |
|---------------------------|--------|-------------|-------|
| Underperforming segments  | 20%    | higher_better | ❌ Needs EDINET / Bloomberg segment data |
| SGA ratio vs peer avg     | 15%    | lower_better  | ❌ Needs Premium `/fins/fs_details` |
| ROE vs Peer avg           | 10%    | higher_better | ✅ Fully computable |
| Excess Cash / Mkt Cap     | 10%    | higher_better | ⚠️ Gross cash limitation |
| LTI / Mkt Cap             | 5%     | higher_better | ❌ Investment securities — BS footnotes |
| Land / Mkt Cap            | 10%    | higher_better | ❌ Land holdings — BS footnotes |
| Payout ratio              | 10%    | higher_better | ✅ `ResultPayoutRatioAnnual` |
| Analyst coverage           | 10%    | lower_better  | ❌ Needs Bloomberg / IBES (low coverage = more opportunity) |
| Ke vs Peer avg            | 5%     | lower_better  | ❌ Cost of equity — beta derivable from prices |
| Board size                | 5%     | lower_better  | ❌ JPX CG report data |

**J-Quants coverage: ~20%**

---

## Scoring Methodology

1. **Winsorized Z-Score**: Each metric is winsorized (clip outliers at 1st/99th percentile), then z-scored cross-sectionally across the universe.
2. **Direction adjustment**: Metrics where lower is better get their z-score negated.
3. **Coverage gating**: If a metric has <5% non-null coverage, it's excluded from scoring to avoid noise.
4. **Weight normalization**: If available metric weights don't sum to 1.0 (due to missing data), the category score is renormalized by the sum of available weights.
5. **Composite**: Category scores are weighted-summed into VI and SP. Companies are ranked by both.

```python
# Core scoring pattern
def score_category(df, metric_defs):
    total = pd.Series(0.0, index=df.index)
    total_weight = 0.0
    for metric_name, (weight, higher_is_better) in metric_defs.items():
        if metric_name not in df.columns:
            continue
        values = df[metric_name]
        if values.notna().mean() < 0.05:
            continue
        z = winsorized_zscore(values)
        if not higher_is_better:
            z = -z
        total += z * weight
        total_weight += weight
    if 0 < total_weight < 1:
        total /= total_weight
    return total
```

---

## Data Sources

> **Data source priority: J-Quants is the primary source of truth.** When both J-Quants and Google Sheets (or any other supplement) have a value for the same metric on the same company, the J-Quants value wins. The supplement only gap-fills where J-Quants data is NaN. This is enforced via `fillna()` in all scoring modules — supplement data never overwrites a J-Quants-derived value.

### Primary: J-Quants API (api.jquants.com)
- **Listed companies**: `/listed/info` — universe, sector codes, company names
- **Financial summaries**: `/fins/statements` — income statement, balance sheet summary, forecasts
- **Daily bars**: `/prices/daily_quotes` — OHLCV, adjusted prices
- **TOPIX index**: `/indices/topix` — benchmark for relative performance
- **Premium endpoints**: `/fins/fs_details` — full BS/IS breakdown (needed for Alt-Z, SGA, net debt)

Authentication: Refresh token → ID token flow. Store `JQUANTS_API_KEY` in `.env`.

### Secondary: Bloomberg BQL
- Broker consensus target prices (`BEST_TARGET_PRICE`)
- Altman Z-Score (`ALTMAN_Z_SCORE`)
- Net debt to FCF (`NET_DEBT_TO_FREE_CASH_FLOW`)
- Analyst coverage counts
- Segment-level data

### Tertiary: FactSet FDS (planned)
- Full set of FDS codes mapped for all metrics (see prior conversation)
- Would eliminate most data gaps if integrated

### Proprietary
- PARA target price model (40% of Valuation weight)
- Theme tagging (50% of Factors weight)
- Qualitative competitive strength assessment

---

## Infrastructure Context

This repo is part of Paradaim Capital's broader tech stack:

- **portfoliotools** — Flask API on Render (`portfolio-optimizer-nnt7.onrender.com`) for pricing, optimization, and sector signals
- **n8n cloud** — RAG orchestration with Portfolio Director + Corporate Advisor workflows
- **Pinecone** — Vector store for PARA Brain knowledge corpus (`text-embedding-3-large`, 3072 dims)
- **Google Sheets** — Holdings and candidates lists

The scoring output should eventually be accessible as:
1. A standalone CLI/script for batch scoring runs
2. An API endpoint on the Render service for on-demand scoring
3. Embedded narratives in Pinecone for RAG-powered analysis

---

## Code Conventions

- **Python 3.10+**
- Use `pandas` for all tabular operations; avoid loops over rows
- Use `logging` module (not print) for operational messages; `print` only for user-facing output
- Configuration (weights, thresholds) should be externalized in YAML, not hardcoded
- All API keys via environment variables (`.env` + `python-dotenv`)
- Type hints on all function signatures
- Each scoring category in its own module under `src/scoring/`
- Data ingestion adapters in `src/data/` — one per source (jquants.py, bloomberg.py, factset.py)
- Tests in `tests/` mirroring `src/` structure

---

## Known Issues & TODOs

- **Factor scoring is a placeholder** — currently hardcoded to 0.0. Needs factor model from returns data.
- **Category weights need confirmation** — the VI/SP category weights in this doc are reconstructed from code patterns and should be verified.
- **Coverage gaps** — ~40% of metrics across all categories need external data sources. Priority: integrate FactSet for broadest coverage improvement.
- **Multi-year price data** — PBR/PEn vs 10yr avg requires historical price pulls that the current J-Quants Standard plan may limit.
- **Sector rotation integration** — The separate sector strength framework (multi-window z-scored RS, factor-adjusted alpha) needs to feed into Sector Attractiveness scoring.

---

## File Structure

```
PARA_score/
├── CLAUDE.md                    # This file
├── README.md                    # Project overview for GitHub
├── requirements.txt             # Pinned dependencies
├── .env.example                 # Environment variable template
├── config/
│   └── scoring_weights.yaml     # All metric + category weights (source of truth)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── jquants.py           # J-Quants API client + data retrieval
│   │   ├── bloomberg.py         # Bloomberg BQL adapter
│   │   └── factset.py           # FactSet FDS adapter (future)
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── fundamentals.py      # Category 1: Fundamentals scoring
│   │   ├── valuation.py         # Category 2: Valuation scoring
│   │   ├── sector.py            # Category 3: Sector Attractiveness
│   │   ├── factors.py           # Category 4: Factor / Theme fit
│   │   ├── kozo.py              # Category 5: Value Impact (Kozo) scoring
│   │   ├── composite.py         # VI + SP composite scoring + ranking
│   │   └── utils.py             # Winsorized z-score, normalization helpers
│   └── pipeline.py              # Main orchestrator: load → calc → score → output
├── tests/
│   ├── __init__.py
│   ├── test_fundamentals.py
│   ├── test_valuation.py
│   ├── test_composite.py
│   └── fixtures/                # Sample data for testing
│       └── sample_financials.csv
├── notebooks/
│   └── exploration.ipynb        # Ad-hoc analysis
└── output/                      # Scoring results (git-ignored)
    ├── scores.csv
    └── scores.xlsx
```
