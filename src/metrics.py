"""
Performance Metrics Module

Calculates and reports portfolio performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> Dict:
    """
    Calculate comprehensive performance metrics.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns series
    equity_curve : pd.Series
        Equity curve (portfolio value over time)
    risk_free_rate : float
        Annual risk-free rate (default: 2%)
    periods_per_year : int
        Number of periods in a year (252 for daily)
        
    Returns
    -------
    Dict
        Dictionary of performance metrics
    """
    if len(returns) == 0 or len(equity_curve) == 0:
        return {}
    
    # Remove NaN values
    returns = returns.dropna()
    
    # Basic return metrics
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_years = len(returns) / periods_per_year
    
    # Annualized metrics
    if n_years > 0:
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
    else:
        annualized_return = 0
    
    # Volatility
    daily_vol = returns.std()
    annualized_vol = daily_vol * np.sqrt(periods_per_year)
    
    # Sharpe ratio
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / annualized_vol if annualized_vol > 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(periods_per_year)
    sortino_ratio = excess_return / downside_vol if downside_vol > 0 else 0
    
    # Maximum drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Skewness and kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # VaR (Value at Risk) - 95% confidence
    var_95 = np.percentile(returns, 5)
    
    # CVaR (Conditional VaR)
    cvar_95 = returns[returns <= var_95].mean()
    
    # Win rate (positive returns)
    win_rate = (returns > 0).sum() / len(returns)
    
    # Profit factor (gross profit / gross loss)
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Expectancy
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
    }
    
    return metrics


def calculate_rolling_metrics(
    returns: pd.Series,
    window: int = 63,  # ~3 months
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
    window : int
        Rolling window size
    risk_free_rate : float
        Annual risk-free rate
        
    Returns
    -------
    pd.DataFrame
        DataFrame with rolling metrics
    """
    # Rolling return
    rolling_return = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
    
    # Rolling volatility
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    
    # Rolling Sharpe
    excess_return = rolling_return * (252 / window) - risk_free_rate
    rolling_sharpe = excess_return / rolling_vol
    
    # Rolling win rate
    rolling_win_rate = returns.rolling(window).apply(lambda x: (x > 0).sum() / len(x))
    
    return pd.DataFrame({
        'return': rolling_return,
        'volatility': rolling_vol,
        'sharpe': rolling_sharpe,
        'win_rate': rolling_win_rate,
    })


def calculate_drawdown_series(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Calculate drawdown series and statistics.
    
    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio equity over time
        
    Returns
    -------
    pd.DataFrame
        Drawdown data with recovery information
    """
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    
    # Find drawdown periods
    is_drawdown = drawdown < 0
    
    # Group consecutive drawdown periods
    drawdown_groups = (is_drawdown != is_drawdown.shift()).cumsum()
    
    drawdown_periods = []
    for group_id in drawdown_groups.unique():
        mask = drawdown_groups == group_id
        if is_drawdown[mask].any():
            period_drawdown = drawdown[mask]
            drawdown_periods.append({
                'start': period_drawdown.index[0],
                'end': period_drawdown.index[-1],
                'max_drawdown': period_drawdown.min(),
                'duration_days': (period_drawdown.index[-1] - period_drawdown.index[0]).days,
            })
    
    return pd.DataFrame(drawdown_periods)


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    
    strategy_name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    final_equity: float
    metrics: Dict
    trade_log: Optional[pd.DataFrame] = None
    
    def __str__(self) -> str:
        """Generate formatted report string."""
        lines = [
            "=" * 60,
            f"PERFORMANCE REPORT: {self.strategy_name}",
            "=" * 60,
            f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
            f"Initial Capital: ${self.initial_capital:,.2f}",
            f"Final Equity: ${self.final_equity:,.2f}",
            "",
            "RETURNS",
            "-" * 40,
            f"Total Return: {self.metrics.get('total_return', 0) * 100:.2f}%",
            f"Annualized Return: {self.metrics.get('annualized_return', 0) * 100:.2f}%",
            f"Annualized Volatility: {self.metrics.get('annualized_volatility', 0) * 100:.2f}%",
            "",
            "RISK METRICS",
            "-" * 40,
            f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}",
            f"Sortino Ratio: {self.metrics.get('sortino_ratio', 0):.2f}",
            f"Max Drawdown: {self.metrics.get('max_drawdown', 0) * 100:.2f}%",
            f"Calmar Ratio: {self.metrics.get('calmar_ratio', 0):.2f}",
            f"VaR (95%): {self.metrics.get('var_95', 0) * 100:.2f}%",
            "",
            "TRADE STATISTICS",
            "-" * 40,
        ]
        
        if self.trade_log is not None and not self.trade_log.empty:
            lines.extend([
                f"Total Trades: {len(self.trade_log)}",
                f"Win Rate: {(self.trade_log['pnl'] > 0).mean() * 100:.1f}%",
                f"Avg Trade Return: {self.trade_log['return_pct'].mean():.2f}%",
                f"Profit Factor: {self.metrics.get('profit_factor', 0):.2f}",
            ])
        else:
            lines.append("No trades executed")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            'strategy_name': self.strategy_name,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'final_equity': self.final_equity,
            'metrics': self.metrics,
            'trade_summary': {
                'total_trades': len(self.trade_log) if self.trade_log is not None else 0,
                'winning_trades': (self.trade_log['pnl'] > 0).sum() if self.trade_log is not None else 0,
                'losing_trades': (self.trade_log['pnl'] <= 0).sum() if self.trade_log is not None else 0,
            }
        }
    
    def save(self, filepath: str):
        """Save report to file."""
        with open(filepath, 'w') as f:
            f.write(str(self))
        logger.info(f"Report saved to {filepath}")


def compare_strategies(
    results: Dict[str, Dict],
    benchmark_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compare multiple strategy results.
    
    Parameters
    ----------
    results : Dict[str, Dict]
        Dictionary mapping strategy names to their backtest results
    benchmark_returns : pd.Series, optional
        Benchmark returns for comparison
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    comparison = []
    
    for name, result in results.items():
        metrics = result.get('metrics', {})
        row = {
            'Strategy': name,
            'Total Return (%)': metrics.get('total_return_pct', 0),
            'Ann. Return (%)': metrics.get('annualized_return_pct', 0),
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Max DD (%)': metrics.get('max_drawdown_pct', 0),
            'Win Rate (%)': metrics.get('win_rate_pct', 0),
            'Trades': metrics.get('num_trades', 0),
            'Profit Factor': metrics.get('profit_factor', 0),
        }
        comparison.append(row)
    
    return pd.DataFrame(comparison)


def calculate_beta_alpha(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.02,
) -> Tuple[float, float]:
    """
    Calculate strategy beta and alpha relative to market.
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Strategy daily returns
    market_returns : pd.Series
        Market (benchmark) daily returns
    risk_free_rate : float
        Annual risk-free rate
        
    Returns
    -------
    Tuple[float, float]
        (beta, alpha)
    """
    # Align series
    aligned = pd.concat([strategy_returns, market_returns], axis=1).dropna()
    if len(aligned) == 0:
        return 0, 0
    
    strat_ret = aligned.iloc[:, 0]
    mkt_ret = aligned.iloc[:, 1]
    
    # Calculate beta
    covariance = strat_ret.cov(mkt_ret)
    market_variance = mkt_ret.var()
    beta = covariance / market_variance if market_variance > 0 else 0
    
    # Calculate alpha (Jensen's alpha)
    strat_ann_return = strat_ret.mean() * 252
    mkt_ann_return = mkt_ret.mean() * 252
    alpha = strat_ann_return - (risk_free_rate + beta * (mkt_ann_return - risk_free_rate))
    
    return beta, alpha


if __name__ == "__main__":
    # Test metrics calculation
    print("Testing metrics module...")
    
    # Generate sample returns
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns = np.random.randn(252) * 0.02
    equity = 100000 * (1 + returns).cumprod()
    
    returns_series = pd.Series(returns, index=dates)
    equity_series = pd.Series(equity, index=dates)
    
    # Calculate metrics
    metrics = calculate_metrics(returns_series, equity_series)
    
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test report generation
    report = PerformanceReport(
        strategy_name="Test Strategy",
        start_date=dates[0],
        end_date=dates[-1],
        initial_capital=100000,
        final_equity=equity[-1],
        metrics=metrics,
    )
    
    print("\n" + str(report))