# PARA Score

Paradaim Capital's proprietary stock scoring system for Japanese equities. Produces **VI (Value Impact)** and **SP (Stock Pick)** composite scores for the TOPIX universe (~3,700+ stocks).

## Scores

- **VI Score** — Measures potential value creation through alliance engagement. Emphasizes fundamentals, valuation upside, and structural improvement (kozo) opportunities.
- **SP Score** — Measures overall attractiveness as a portfolio holding. Balances fundamentals, valuation, sector dynamics, and factor/theme alignment.

## Scoring Categories

| Category              | VI Weight | SP Weight |
|-----------------------|-----------|-----------|
| Fundamentals          | 0.25      | 0.25      |
| Valuation             | 0.25      | 0.25      |
| Sector Attractiveness | 0.00      | 0.15      |
| Factors / Themes      | 0.20      | 0.20      |
| Value Impact (Kozo)   | 0.30      | 0.15      |

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
# Run the full scoring pipeline
python -m src.pipeline

# Run tests
pytest tests/
```

## Project Structure

```
PARA_score/
├── config/scoring_weights.yaml   # Metric + category weights (source of truth)
├── src/
│   ├── data/                     # Data ingestion adapters
│   │   ├── jquants.py            # J-Quants API client
│   │   ├── bloomberg.py          # Bloomberg BQL adapter
│   │   └── factset.py            # FactSet adapter (planned)
│   ├── scoring/                  # Scoring modules
│   │   ├── fundamentals.py       # Category 1: Fundamentals
│   │   ├── valuation.py          # Category 2: Valuation
│   │   ├── sector.py             # Category 3: Sector Attractiveness
│   │   ├── factors.py            # Category 4: Factors / Themes
│   │   ├── kozo.py               # Category 5: Value Impact (Kozo)
│   │   ├── composite.py          # VI + SP composite scoring
│   │   └── utils.py              # Winsorized z-score helpers
│   └── pipeline.py               # Main orchestrator
├── tests/                        # Test suite
└── output/                       # Scoring results (git-ignored)
```

## Data Sources

- **Primary**: J-Quants API — listed companies, financials, daily prices, TOPIX index
- **Secondary**: Bloomberg BQL — broker targets, Altman Z, analyst coverage
- **Tertiary**: FactSet FDS (planned)
- **Proprietary**: PARA target price model, theme tagging
