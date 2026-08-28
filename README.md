# Dual MA + RSI Strategy

A complete end-to-end momentum trading strategy implementation for educational purposes. This project demonstrates professional quantitative finance techniques including data ingestion, backtesting, walk-forward validation, and performance analysis.

## Overview

This project implements a **Dual Moving Average Crossover with RSI Filter** strategy - a classic momentum approach suitable for students learning quantitative trading. The strategy combines trend-following with momentum confirmation to filter out false signals.

### Strategy Logic

- **Entry Signal**: Fast SMA crosses above Slow SMA AND RSI > 50 (momentum confirmation)
- **Exit Signal**: Fast SMA crosses below Slow SMA OR RSI > 70 (overbought)
- **Risk Management**: Position sizing based on volatility (Kelly Criterion inspired)

## Features

- **Data Ingestion**: Yahoo Finance integration via `yfinance`
- **Backtesting Engine**: Custom vectorized backtester for speed and clarity
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate, annualized return
- **Walk-Forward Validation**: Time-series cross-validation to demonstrate robustness
- **Interactive Analysis**: Jupyter notebook with visualizations

## Fixed: silent zero-trade bug in ATR position sizing

`DualMAStrategy.get_position_sizes()` computed share counts purely from ATR
(`risk_per_trade / (ATR * 2)`) with no cap tying the result back to available
capital. At AAPL's 2020-2024 price level, that formula sized positions worth
$95k-$975k against a $100k account — the backtester's `if total_cost <=
capital` check then silently skipped every buy, with no warning logged, so
the strategy reported **zero trades** and looked untested rather than broken.

This was originally caught while porting the strategy to R (see
[`dual-ma-rsi-strategy-r`](https://github.com/Samarty-1/dual-ma-rsi-strategy-r),
which reproduces the bug faithfully and compares it against a corrected
variant). It's now fixed here directly: `get_position_sizes()` clips share
count to `(capital * position_size%) / price`, so sizing never exceeds the
capital actually allocated to the trade. With the fix, the strategy now
executes 10 trades over the 2020-2024 AAPL window (previously 0) — see
`notebooks/strategy_analysis.ipynb` for the full, actually-executed run.

Also fixed while validating this: `requirements.txt` was missing
`scikit-learn` despite `walk_forward.py` importing it (fresh installs broke
immediately), the `signal` column produced `NaN` on the first bar instead of
`0` (invalid for a -1/0/1 signal), and three test assertions referenced
metric keys/edge cases that didn't match the actual implementation. The test
suite (`pytest tests`) is now 13/13 passing.

## Why the strategy still loses money, and a real (partial) fix

With the bug above fixed, the strategy ran but still lost money on AAPL
2020-2024 (-1.23% return, Sharpe -3.55, profit factor 0.52 — see the results
table below). Rather than accept "market efficiency" as the explanation
without checking, the trade log was cross-referenced against the RSI/MA
values at each exit to see *why* each trade closed:

| Exit reason | Trades | Returns |
|---|---|---|
| RSI > 70 (overbought) | 8 of 10 | +5.6%, -0.4%, -1.5%, +1.0%, +4.5%, +1.3%, -2.0%, +0.1% |
| MA crossunder (trend reversal) | 2 of 10 | **-9.1%, -10.9%** |

That's the mechanism, not a guess: the RSI>70 exit is a hair-trigger —
several of those trades exit within 1-2 days of entry, capping winners at a
couple of percent before the trend has a chance to run. The *only* other
exit path, the MA crossunder, doesn't fire until a decline has already
dragged the slow moving average down with it — by which point both
crossunder exits were already double-digit losses. The strategy was
structurally set up to cut winners early and let losers run, the opposite
of sound trade management.

**Fix**: `DualMAStrategy(use_atr_trailing_stop=True)` drops the RSI exit
entirely and lets `Backtester` manage the exit with an ATR-based trailing
stop instead (same mechanism already validated in
[`paper-trading-bot`](https://github.com/Samarty-1/paper-trading-bot)).
Tested honestly, not just on the one window it was designed to fix:

| | Baseline (RSI exit) | ATR trailing stop |
|---|---|---|
| AAPL total return / Sharpe / profit factor | -1.23% / -3.55 / 0.52 | **+2.72% / -1.07 / 2.06** |
| AAPL avg trade duration | 8.9 days | 33.1 days |
| Walk-forward mean return / Sharpe (5 folds) | -0.37% / -19.75 | **+0.06% / -2.56** |

Both the single-window backtest and 5-fold walk-forward validation show a
large, real improvement — profit factor crosses from a losing edge (0.52)
to a winning one (2.06), and winning trades now run 3-4x longer instead of
being cut within days. **This is not a universal fix, and the full,
honest picture matters more than the AAPL headline number:**

| Ticker | Baseline profit factor | ATR-stop profit factor |
|---|---|---|
| AAPL | 0.52 | **2.06** (much better) |
| MSFT | 0.59 | **1.05** (better) |
| QQQ | 1.39 | 1.21 (roughly flat) |
| SPY | **5.17** | 0.64 (worse) |

SPY's baseline RSI exit was already good (profit factor 5.17) — on a broad,
lower-volatility index ETF, cutting a winner at RSI>70 apparently *does*
protect against giving back gains often enough to pay for itself, unlike on
a more volatile individual stock like AAPL where it fires prematurely. The
honest takeaway: `use_atr_trailing_stop` is a real, validated improvement
for individual-stock volatility profiles like AAPL/MSFT, not a strictly
better default for every instrument — which instrument you're trading
should decide which exit rule to use, not a blanket assumption either way.

## Project Structure

```
dual-ma-rsi-strategy/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                 # Package setup
├── .gitignore               # Git ignore rules
├── src/                     # Source code
│   ├── __init__.py
│   ├── data_ingestion.py    # Yahoo Finance data fetcher
│   ├── strategy.py          # Strategy implementation
│   ├── backtest.py          # Backtesting engine
│   ├── metrics.py           # Performance metrics
│   └── walk_forward.py      # Walk-forward validation
├── notebooks/               # Jupyter notebooks
│   └── strategy_analysis.ipynb
└── tests/                   # Unit tests
    ├── __init__.py
    ├── test_strategy.py
    └── test_backtest.py
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/Samarty-1/dual-ma-rsi-strategy.git
cd dual-ma-rsi-strategy

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Run the Strategy

```python
from src.data_ingestion import fetch_data
from src.strategy import DualMAStrategy
from src.backtest import Backtester

# Fetch data
data = fetch_data('AAPL', start='2020-01-01', end='2024-01-01')

# Initialize strategy
strategy = DualMAStrategy(fast_window=20, slow_window=50, rsi_period=14)

# Run backtest
backtest = Backtester(data, strategy, initial_capital=100000)
results = backtest.run()

# View metrics
print(results['metrics'])
```

### Run Walk-Forward Validation

```python
from src.walk_forward import WalkForwardValidator

validator = WalkForwardValidator(data, strategy, n_splits=5)
wf_results = validator.run()
validator.plot_results()
```

### Jupyter Notebook

For a complete analysis with visualizations:

```bash
jupyter notebook notebooks/strategy_analysis.ipynb
```

## Strategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fast_window` | 20 | Fast moving average period |
| `slow_window` | 50 | Slow moving average period |
| `rsi_period` | 14 | RSI calculation period |
| `rsi_overbought` | 70 | RSI overbought threshold |
| `position_size` | 0.1 | Max position size (10% of capital) |

## Performance Metrics

The backtester calculates the following metrics:

- **Total Return**: Cumulative strategy return
- **Annualized Return**: Return normalized to yearly basis
- **Sharpe Ratio**: Risk-adjusted return (assuming risk-free rate = 2%)
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Calmar Ratio**: Annualized return / Max drawdown

## Walk-Forward Validation

Walk-forward analysis is implemented to demonstrate strategy robustness and avoid overfitting:

1. **In-Sample Period**: Train/optimize on historical data
2. **Out-of-Sample Period**: Test on unseen future data
3. **Roll Forward**: Slide window and repeat

This approach simulates real-world trading where the strategy is periodically retrained on recent data.

## Actual Results (AAPL, 2020-2024, default config — see "Fixed" section above)

This section previously claimed fabricated, never-validated numbers (Sharpe
1.2-1.8, 55-65% win rate) that directly contradicted the real backtest
results documented elsewhere in this same README — left over from before
the strategy was ever actually run. The real, actually-executed numbers
**for the default RSI-exit config**; see "Why the strategy still loses
money" above for the validated `use_atr_trailing_stop=True` alternative,
which does meaningfully better on this specific ticker:

| Metric | Value |
|--------|-------|
| Total Return | -1.23% |
| Sharpe Ratio | -3.55 |
| Win Rate | 50.0% |
| Profit Factor | 0.52 |
| Trades | 10 |

*Past performance does not guarantee future results. This is for
educational purposes only — and, honestly, this specific parameter set
loses money on this specific window. See `notebooks/strategy_analysis.ipynb`
for the full run.*

## Testing

Run unit tests:

```bash
pytest tests/
```

## Educational Notes

### Why This Strategy?

1. **Simplicity**: Easy to understand and explain
2. **Robustness**: Moving averages are widely used in industry
3. **Risk Management**: Built-in position sizing and stop-loss logic
4. **Extensibility**: Easy to add more indicators or filters

### Limitations

- **Lagging Indicators**: Moving averages are inherently lagging
- **Whipsaws**: Can generate false signals in choppy markets
- **No Transaction Costs**: Real trading includes slippage and fees
- **Survivorship Bias**: Yahoo Finance data may have delisting bias

## Future Enhancements

- [ ] Add transaction cost modeling
- [ ] Implement portfolio-level backtesting (multiple assets)
- [ ] Add machine learning features
- [ ] Monte Carlo simulation for risk analysis
- [ ] Live paper trading integration

## Disclaimer

**This project is for educational purposes only. It is not financial advice. Trading involves substantial risk of loss. Always do your own research and consult with a qualified financial advisor before making investment decisions.**

## License

MIT License - See LICENSE file for details.

## Author

Created as part of a quantitative finance student portfolio project.

## Acknowledgments

- Yahoo Finance for providing free historical data
- The quantitative finance community for open-source tools and research
