# Statistical Arbitrage System

> **Equity pairs trading system for S&P 500 stocks using Kalman Filter hedge ratios, HMM regime detection, and Almgren-Chriss transaction cost modeling — deployed for live paper trading via Alpaca.**

---

## Table of Contents

- [Overview](#overview)
- [Performance](#performance)
- [Architecture](#architecture)
- [Key Algorithms](#key-algorithms)
- [Universe](#universe)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Modules](#modules)
- [Safety Rules](#safety-rules)
- [Results](#results)

---

## Overview

This system implements a quantitative equity pairs trading strategy that:

1. **Selects cointegrated pairs** from 119 S&P 500 stocks using Engle-Granger and Johansen tests
2. **Estimates dynamic hedge ratios** via Kalman Filter (auto-tuned delta per pair)
3. **Detects market regimes** using a 3-state Hidden Markov Model
4. **Generates entry/exit signals** from rolling 10-day z-score on OLS spread
5. **Gates trading** using VIX levels, HMM market regime, and smoothed volatility filter
6. **Executes paper trades** automatically via Alpaca API with full audit logging

The strategy is **market-neutral** (long one stock, short another within the same sector) and designed to capture mean-reversion of spread relationships while avoiding trending and high-volatility market regimes.

---

## Performance

| Metric | Value |
|---|---|
| Backtest Period | 2014 – 2019 (walk-forward) |
| Annualised Return | +1.35% |
| **Sharpe Ratio** | **+0.732** |
| Max Drawdown | -2.30% |
| Total Trades | 249 |
| Win Rate (folds) | 6 / 7 positive |
| IS Portfolio Filter | Sharpe ≥ 1.28 |
| Transaction Cost | 2 bps / side (realistic A-C model) |

> Best individual fold: **Sharpe +3.13** (2018 H2) and **+2.91** (2016 H1)

---

## Architecture

```
stat_arb/
├── data/
│   ├── fetch.py              # yfinance downloader (1d + 5m)
│   ├── vix.py                # VIX data (CBOE Volatility Index)
│   ├── fetch_etfs.py         # Sector ETF data (XLK, XLF, XLV, ...)
│   └── universe.py           # 119-stock S&P 500 universe + sector map
│
├── pairs/
│   ├── select.py             # Engle-Granger + Johansen + half-life screen
│   ├── monitor.py            # Ongoing cointegration health checks
│   └── run.py                # CLI: pair selection report
│
├── kalman/
│   ├── filter.py             # Dynamic hedge ratio (MLE delta tuning, warmup)
│   └── smoother.py           # RTS backward smoother (offline analysis)
│
├── hmm/
│   ├── regime.py             # 3-state HMM (MR / Trending / Volatile)
│   └── train.py              # Rolling 252-day re-training
│
├── costs/
│   └── almgren_chriss.py     # Permanent + temporary market impact model
│
├── risk/
│   ├── limits.py             # Hard drawdown / position / daily-loss limits
│   └── sizing.py             # Kelly criterion + vol-adjusted sizing
│
├── backtesting/
│   ├── engine.py             # Walk-forward backtest orchestrator
│   ├── metrics.py            # Sharpe, max-DD, Calmar, win-rate
│   └── run.py                # CLI entrypoint
│
└── execution/
    ├── alpaca_client.py      # Alpaca paper-trading API wrapper
    ├── signal_generator.py   # Daily signal computation
    ├── portfolio_manager.py  # Position state + trade reconciliation
    ├── order_manager.py      # Order submission (fractional + whole shares)
    └── run.py                # Main live trading loop

data/
├── ohlcv/1d/                 # Daily price parquets (119 stocks, 2010-2026)
├── ohlcv/5m/                 # 5-minute price parquets (60-day rolling)
├── ohlcv/etf/                # Sector ETF parquets (XLK, XLF, ...)
├── market/vix.parquet        # VIX history (2008-2026)
├── execution/
│   ├── state.json            # Current open positions
│   └── scheduler.log         # Daily run log
└── trades/                   # Permanent audit logs (YYYYMMDD.json)
```

---

## Key Algorithms

### 1. Kalman Filter — Dynamic Hedge Ratios

Models the pair relationship as:

```
log(P_y[t]) = α[t] + β[t] · log(P_x[t]) + ε[t]
```

- **State**: [α_t, β_t] evolves as a random walk
- **Delta**: MLE-tuned per pair via innovation log-likelihood grid search
- **Warmup**: First 60 bars masked to prevent unreliable early estimates
- **Spread**: Innovation z-score computed BEFORE state update (no look-ahead)

### 2. HMM Regime Detection

3-state Gaussian HMM trained on spread features:

| State | Features | Trading Rule |
|---|---|---|
| Mean-Reverting | Negative autocorr, low vol | Allow entry |
| Trending | Positive autocorr, directional drift | Avoid entry |
| Volatile | High realized vol, stress | Block entry |

State labelling is **automatic** and consistent across re-training windows using spread autocorrelation as the primary criterion.

### 3. Rolling Z-Score Signal

```
spread_t  = log(P_y_t) − α_IS − β_IS · log(P_x_t)
z_t       = (spread_t − mean_{t-10:t}) / std_{t-10:t}
```

- **10-day rolling window** calibrated to actual spread std (~1.3%/day)
- Entry at |z| > 1.5 · exit at |z| < 0.3 · stop-loss at |z| > 3.5
- Rolling window (not Kalman innovation) avoids filter-adaptation artifact

### 4. Multi-Layer Market Gate

```
Layer 1 — VIX gate:      block when VIX > 25 (stress) or < 11 (complacency)
Layer 2 — HMM gate:      block when market HMM = VOLATILE (smoothed, 10d window)
Layer 3 — IS Sharpe:     skip fold when portfolio IS Sharpe < 1.28
Layer 4 — Cooldown:      5-day re-entry ban after volatile period ends
```

### 5. Almgren-Chriss Transaction Costs

```
Permanent impact  = γ · σ · (V / ADV)
Temporary impact  = η · σ · (V / ADV)^0.6
```

For S&P 500 large-caps at $50K notional: effective cost ≈ **1–4 bps/side** (far lower than the 10 bps flat-rate assumption).

---

## Universe

**119 S&P 500 stocks** across 10 GICS sectors + **9 sector ETFs**:

| Sector | Stocks | ETF |
|---|---|---|
| Technology | 22 | XLK |
| Health Care | 17 | XLV |
| Financials | 15 | XLF |
| Consumer Discretionary | 13 | XLY |
| Consumer Staples | 11 | XLP |
| Industrials | 12 | XLI |
| Communication Services | 10 | — |
| Energy | 7 | XLE |
| Materials | 6 | XLB |
| Utilities | 6 | XLU |

Same-sector constraint reduces candidate pairs from 7,875 to **~880** with strong economic rationale.

---

## Installation

```bash
# Clone and set up environment
git clone <repo>
cd "Statistical Arbitrage System"
python -m venv .venv && .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API credentials
cp .env.example .env
# Edit .env with Alpaca paper trading API key
```

**Requirements:**
- Python 3.10+
- yfinance, pandas, numpy, statsmodels
- hmmlearn, scikit-learn
- alpaca-py, python-dotenv
- loguru, pyarrow

---

## Quick Start

```bash
# 1. Download daily price data (119 stocks, 2010–present)
python -m stat_arb.data.run --interval 1d --start 2010-01-01 --force

# 2. Download VIX data
python -m stat_arb.data.vix

# 3. Screen cointegrated pairs
python -m stat_arb.pairs.run --n-jobs -1

# 4. Run walk-forward backtest
python -m stat_arb.backtesting.run \
    --start 2014-01-01 --end 2020-01-01 \
    --min-is-sharpe 0.3 --vix-threshold 25 \
    --min-portfolio-sharpe 1.28 --tc-bps 2

# 5. Paper trading (dry-run preview)
python -m stat_arb.execution.run

# 6. Paper trading (live — submits real paper orders)
python -m stat_arb.execution.run --live
```

---

## Configuration

All runtime parameters are in `config/settings.yaml`:

```yaml
pairs:
  min_half_life_days: 3
  max_half_life_days: 120
  same_sector_only: true
  fdr_correction: false

kalman:
  delta: 1.0e-5        # process noise (auto-tuned per pair)
  vt: 1.0e-3           # observation noise

hmm:
  n_states: 3
  train_window_days: 252

signals:
  entry_zscore: 1.5
  exit_zscore: 0.3
  stop_zscore: 3.5

risk:
  max_notional_per_pair: 50_000
  max_pairs_open: 15
  max_portfolio_drawdown: 0.15
  max_daily_loss: 0.03
```

---

## Modules

### Data Pipeline

```bash
# Daily data (yfinance, unlimited history)
python -m stat_arb.data.run --interval 1d

# 5-minute data (yfinance, last 60 days)
python -m stat_arb.data.run --interval 5m

# Sector ETFs
python -m stat_arb.data.fetch_etfs

# VIX
python -m stat_arb.data.vix
```

### Cointegration Screening

```bash
# Full universe screening with report
python -m stat_arb.pairs.run --n-jobs -1 --out data/pairs.csv

# Key filters applied:
# 1. Engle-Granger (p < 0.05)
# 2. Johansen trace test (95% confidence)
# 3. Half-life filter [3, 120] days
# 4. Same-sector constraint
# 5. BH-FDR correction (optional)
```

### Backtesting

```bash
# Walk-forward backtest (2014–2019, optimal config)
python -m stat_arb.backtesting.run \
    --start 2014-01-01 --end 2020-01-01 \
    --eg-pvalue 0.05 --max-halflife 120 \
    --min-is-sharpe 0.3 --vix-threshold 25 \
    --vol-threshold 0.20 --entry-z 1.5 \
    --exit-z 0.3 --stop-z 3.5 --tc-bps 2 \
    --min-portfolio-sharpe 1.28
```

### Live Paper Trading

```bash
# Check account status
python -m stat_arb.execution.run --status

# Daily execution (dry-run)
python -m stat_arb.execution.run

# Daily execution with data refresh (live)
python -m stat_arb.execution.run --live --update-data

# Emergency: close all positions
python -m stat_arb.execution.run --live --close-all
```

**Scheduled daily execution** (Windows Task Scheduler):
```
run_daily.bat  →  runs at 4:30 AM (Vietnam time = 4:30 PM US EDT)
```

---

## Safety Rules

> These rules protect real capital and audit trails. Do not bypass.

| Rule | Detail |
|---|---|
| **Paper mode only** | `ALPACA_BASE_URL` must always be `paper-api.alpaca.markets` |
| **Dry-run default** | `--live` flag required to submit real orders |
| **Audit logs** | `data/trades/YYYYMMDD.json` — never delete or overwrite |
| **Position limits** | `risk/limits.py` — never change without explicit instruction |
| **`.env` file** | Never commit, never log its contents |
| **Execution files** | Any change to `execution/` must be stated explicitly before making |

---

## Results

### Backtest Summary (2014–2019, IS Portfolio Filter = 1.28)

| Fold | OOS Period | Pairs | Trades | Sharpe |
|---|---|---|---|---|
| 1 | 2016 H1 | 10 | 9 | **+2.91** |
| 2 | 2016 H2 | 8 | 37 | +0.49 |
| 3 | 2017 H1 | 18 | 95 | +0.93 |
| 4 | 2017 H2 | 17 | 72 | +0.45 |
| 5 | 2018 H1 | 0 | 0 | *skipped* |
| 6 | 2018 H2 | 9 | 19 | **+3.13** |
| 7 | 2019 H1 | 6 | 17 | **+1.05** |
| | **Aggregate** | | **249** | **+0.732** |

### Live Paper Trading (from June 2026)

| Date | Event | Equity |
|---|---|---|
| 2026-06-02 | Started — DIS/META LONG | $100,000 |
| 2026-06-02 | First P&L update | $102,433 |
| 2026-06-03 | DIS/META KEEP | $103,950 |
| 2026-06-04 | Spread widening | $99,560 |

---

## Testing

```bash
# Full test suite (339 tests across all modules)
pytest tests/ -v

# Module-specific
pytest tests/kalman/     # Kalman filter (44 tests)
pytest tests/hmm/        # HMM regime (64 tests)
pytest tests/pairs/      # Cointegration (65 tests)
pytest tests/backtesting/ # Engine + metrics (76 tests)
pytest tests/risk/       # Risk limits + sizing (90 tests)
```

---

## License

Private research project. All rights reserved.

---

*Built with Python 3.13 · yfinance · alpaca-py · statsmodels · hmmlearn · loguru*
