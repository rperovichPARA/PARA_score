# PARA Score

Paradaim Capital's proprietary stock scoring system for Japanese equities. Produces VI (Value Impact) and SP (Stock Pick) composite scores for the TOPIX Prime universe (~1,600 stocks).

## Scores

* **VI Score** — Measures potential value creation through alliance engagement. Emphasizes fundamentals, valuation upside, and structural improvement (kozo) opportunities.
* **SP Score** — Measures overall attractiveness as a portfolio holding. Balances fundamentals, valuation, sector dynamics, and factor/theme alignment.

## Scoring Categories

| Category | VI Weight | SP Weight | Data Source |
|---|---|---|---|
| Fundamentals | 0.25 | 0.25 | J-Quants financials |
| Valuation | 0.25 | 0.25 | J-Quants financials + Google Sheets |
| Sector Attractiveness | 0.00 | 0.15 | Sector rotation signals (portfoliotools API) |
| Factors / Themes | 0.20 | 0.20 | J-Quants daily quotes + financials |
| Value Impact (Kozo) | 0.30 | 0.15 | J-Quants financials + Google Sheets |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  GitHub Actions (weekly cron, Saturday 01:00 UTC)       │
│                                                         │
│  1. Fetch listed companies from J-Quants V2 API         │
│  2. Fetch financials + daily quotes per code            │
│  3. Pull supplementary metrics from Google Sheets       │
│  4. Fetch sector signals from portfoliotools API        │
│  5. Score all five categories → composite VI/SP         │
│  6. POST results to /para-score/upload on Render        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  portfoliotools (Flask on Render)                       │
│                                                         │
│  para_score_blueprint.py — pure serving layer           │
│  GET /para-score/top/20?sort=vi    Top stocks           │
│  GET /para-score/batch?codes=7203  Batch lookup         │
│  GET /para-score/screen?min_vi=0.2 Screening            │
│  GET /para-score/summary           Coverage stats       │
│  GET /para-score/status            Cache freshness      │
│  POST /para-score/upload           Receive scores       │
│                                                         │
│  Cache: Render persistent disk (/var/data/cache)        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  n8n AI Agents                                          │
│                                                         │
│  Portfolio Director — stock rankings, screening         │
│  Corporate Advisor  — engagement candidate evaluation   │
│                                                         │
│  HTTP Request tools query /para-score endpoints         │
└─────────────────────────────────────────────────────────┘
```

## Setup

```bash
# Clone and install
git clone <repo-url>
cd PARA_score
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your J-Quants API key
```

## Usage

```bash
# Run full universe
python -m src.pipeline --universe all --output-dir ./output

# Run specific stocks
python -m src.pipeline --universe 7203,6758,8306 --output-dir ./output

# Export and upload to Render
python -m src.export --input ./output/scores.csv

# Export dry run (write JSON locally, don't upload)
python -m src.export --input ./output/scores.csv --dry-run

# Run tests
pytest tests/
```

## GitHub Actions

The pipeline runs automatically via `.github/workflows/score_pipeline.yml`:

* **Scheduled:** Every Saturday 01:00 UTC (Saturday morning JST, after Friday close)
* **Manual:** Actions tab → PARA Score Pipeline → Run workflow
  * `universe`: comma-separated codes or `all` (default: `all`)
  * `dry_run`: write payload to artifact instead of uploading

### Required Secrets

| Secret | Description |
|---|---|
| `JQUANTS_API_KEY` | J-Quants V2 API key |
| `GOOGLE_SHEETS_CREDENTIALS` | Service account JSON for supplementary metrics |
| `PARA_SCORE_UPLOAD_KEY` | Bearer token matching Render env var |
| `RENDER_API_URL` | `https://portfolio-optimizer-nnt7.onrender.com` |

## Project Structure

```
PARA_score/
├── .github/workflows/
│   └── score_pipeline.yml        # Weekly cron + manual dispatch
├── config/
│   └── scoring_weights.yaml      # Metric + category weights (source of truth)
├── src/
│   ├── data/                     # Data ingestion adapters
│   │   ├── jquants.py            # J-Quants V2 API client
│   │   ├── gsheets.py            # Google Sheets adapter (supplementary metrics)
│   │   ├── bloomberg.py          # Bloomberg BQL adapter (future)
│   │   └── factset.py            # FactSet adapter (planned)
│   ├── scoring/                  # Scoring modules
│   │   ├── fundamentals.py       # ROE, margins, growth, Altman Z
│   │   ├── valuation.py          # PBR, PEn, PEG, target upside
│   │   ├── sector.py             # Sector alpha from rotation signals
│   │   ├── factors.py            # Price/earnings momentum, volatility
│   │   ├── kozo.py               # Board size, analyst coverage, balance sheet
│   │   ├── composite.py          # VI + SP composite scoring
│   │   └── utils.py              # Winsorized z-score helpers
│   ├── pipeline.py               # Main orchestrator with CLI
│   └── export.py                 # Format + upload to Render
├── tests/                        # Test suite
└── output/                       # Scoring results (git-ignored)
```

## Data Sources

* **J-Quants V2 API** (primary) — listed companies, financial statements, daily quotes, TOPIX index
* **Google Sheets** (supplementary) — broker targets, Altman Z-Score, board size, analyst coverage, net cash/mktcap, Ke, land/mktcap
* **Sector Rotation Signals** (portfoliotools API) — factor-adjusted TOPIX-17 sector alpha z-scores
* **Bloomberg BQL** (future) — additional broker/analyst data
* **FactSet FDS** (planned) — institutional ownership, ESG
* **Proprietary** — PARA target price model, theme tagging
