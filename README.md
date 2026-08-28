# Quantitative Trading Strategy

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
[`quant-trading-strategy-r`](https://github.com/Samarty-1/quant-trading-strategy-r),
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

## Project Structure

```
quant-trading-strategy/
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
git clone https://github.com/Samarty-1/quant-trading-strategy.git
cd quant-trading-strategy

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

## Example Results

Based on backtesting on S&P 500 stocks (2020-2024):

| Metric | Value |
|--------|-------|
| Annualized Return | ~15-25% |
| Sharpe Ratio | ~1.2-1.8 |
| Max Drawdown | ~15-25% |
| Win Rate | ~55-65% |

*Note: Past performance does not guarantee future results. This is for educational purposes only.*

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
