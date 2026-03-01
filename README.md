# Equity Factor Analysis: Value vs Growth in the STOXX 600

Analysis of the value factor in European equities using STOXX 600 constituents,
built in Python with Bloomberg data (January 2010 – December 2025).

## Overview

This project constructs value and growth portfolios by sorting STOXX 600 stocks
on price-to-book ratio each month and examines their performance against the
cap-weighted index. It covers the full factor research workflow: data cleaning,
portfolio construction, performance measurement, risk attribution, and
statistical testing.

## Methodology

- **Universe:** STOXX 600 (600 constituents, 192 months)
- **Signal:** Price-to-book ratio, lagged one month to avoid look-ahead bias
- **Value portfolio:** Bottom 30% of constituents by P/B each month
- **Growth portfolio:** Top 30% of constituents by P/B each month
- **Weighting:** Cap-weighted within each bucket, rebalanced monthly
- **Benchmark:** Cap-weighted STOXX 600 index

## Key Findings

| Metric | Value | Growth | Index |
|---|---|---|---|
| Annualised Return | 17.3% | 16.3% | 16.2% |
| Annualised Volatility | 17.0% | 12.6% | 12.6% |
| Sharpe Ratio | 1.016 | 1.301 | 1.289 |
| Max Drawdown | -28.5% | -20.3% | -19.2% |

- Value generated higher cumulative returns (+1,177%) than growth (+1,027%) and
  the index (+1,001%), but with meaningfully higher volatility and a lower
  Sharpe ratio
- The value premium (1.43% annualised) was **not statistically significant**
  (t=0.440, p=0.660)
- After beta adjustment, neither portfolio generated significant alpha: value
  -1.97% p.a. (p=0.303), growth +1.75% p.a. (p=0.238)
- Outperformance was almost entirely regime-driven — value lagged growth for
  most of 2010–2021, then dominated from 2022 as ECB rate normalisation
  repriced long-duration assets

## Contents

| File | Description |
|---|---|
| `equity_factors.ipynb` | Full analysis — cleaning, construction, metrics, charts, analysis |
| `prices_monthly.csv` | Month-end adjusted prices, STOXX 600 constituents |
| `market_cap_monthly.csv` | Month-end market capitalisations |
| `price_to_book_monthly.csv` | Month-end price-to-book ratios |

## Requirements
```
pandas
numpy
scipy
statsmodels
matplotlib
plotly
seaborn
```

## Data

Data was downloaded from Bloomberg via BQL. The pre-downloaded CSVs covering January 2010 to
December 2025 are included in the repository.
