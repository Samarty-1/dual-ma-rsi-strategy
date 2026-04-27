"""
Walk-Forward Validation Module

Implements walk-forward analysis to validate strategy robustness
and avoid overfitting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import logging
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

from .strategy import BaseStrategy
from .backtest import Backtester
from .metrics import calculate_metrics, compare_strategies

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation for trading strategies.
    
    Walk-forward analysis helps prevent overfitting by:
    1. Training/optimizing on in-sample data
    2. Testing on out-of-sample data
    3. Rolling the window forward and repeating
    
    This simulates how a strategy would be used in practice,
    where it's periodically retrained on recent data.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategy_class,
        strategy_params: Optional[Dict] = None,
        n_splits: int = 5,
        train_size: Optional[float] = None,
        test_size: Optional[float] = None,
        expanding_window: bool = True,
        optimization_func: Optional[Callable] = None,
    ):
        """
        Initialize walk-forward validator.
        
        Parameters
        ----------
        data : pd.DataFrame
            Full historical data
        strategy_class : class
            Strategy class to instantiate
        strategy_params : Dict, optional
            Base strategy parameters
        n_splits : int
            Number of walk-forward splits
        train_size : float, optional
            Fraction of data for training (if None, uses TimeSeriesSplit)
        test_size : float, optional
            Fraction of data for testing
        expanding_window : bool
            If True, training window expands; if False, fixed window
        optimization_func : Callable, optional
            Function to optimize strategy parameters on training data
        """
        self.data = data.copy()
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params or {}
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.expanding_window = expanding_window
        self.optimization_func = optimization_func
        
        self.results: List[Dict] = []
        self.fold_metrics: List[Dict] = []
        
    def run(self) -> Dict:
        """
        Run walk-forward validation.
        
        Returns
        -------
        Dict
            Aggregated results across all folds
        """
        logger.info(f"Starting walk-forward validation with {self.n_splits} splits")
        
        if self.train_size is not None:
            # Custom split logic
            splits = self._create_custom_splits()
        else:
            # Use sklearn's TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            splits = list(tscv.split(self.data))
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"\n--- Fold {fold + 1}/{self.n_splits} ---")
            
            # Split data
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]
            
            logger.info(f"Train: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} bars)")
            logger.info(f"Test: {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} bars)")
            
            # Optimize parameters if function provided
            if self.optimization_func:
                best_params = self.optimization_func(train_data, self.strategy_class)
                strategy_params = {**self.strategy_params, **best_params}
                logger.info(f"Optimized params: {best_params}")
            else:
                strategy_params = self.strategy_params
            
            # Train strategy on in-sample data (optional - for stateful strategies)
            strategy = self.strategy_class(**strategy_params)
            
            # Test on out-of-sample data
            backtest = Backtester(
                test_data,
                strategy,
                initial_capital=100000,
            )
            result = backtest.run()
            
            # Store results
            fold_result = {
                'fold': fold + 1,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'metrics': result['metrics'],
                'equity_curve': result['equity_curve'],
                'trades': result['trades'],
                'params': strategy_params,
            }
            self.results.append(fold_result)
            self.fold_metrics.append(result['metrics'])
        
        # Aggregate results
        aggregated = self._aggregate_results()
        
        logger.info("\n" + "=" * 60)
        logger.info("WALK-FORWARD VALIDATION COMPLETE")
        logger.info("=" * 60)
        
        return aggregated
    
    def _create_custom_splits(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create custom train/test splits."""
        n_samples = len(self.data)
        splits = []
        
        if self.expanding_window:
            # Expanding window - train set grows
            test_size = int(n_samples * self.test_size) if self.test_size else int(n_samples / (self.n_splits + 1))
            train_start = 0
            
            for i in range(self.n_splits):
                train_end = n_samples - (self.n_splits - i) * test_size
                test_start = train_end
                test_end = min(test_start + test_size, n_samples)
                
                if train_end > train_start and test_end > test_start:
                    splits.append((
                        np.arange(train_start, train_end),
                        np.arange(test_start, test_end)
                    ))
        else:
            # Fixed window - sliding window
            train_size = int(n_samples * self.train_size) if self.train_size else int(n_samples * 0.6)
            test_size = int(n_samples * self.test_size) if self.test_size else int(n_samples * 0.2)
            
            step = (n_samples - train_size - test_size) // self.n_splits
            
            for i in range(self.n_splits):
                train_start = i * step
                train_end = train_start + train_size
                test_start = train_end
                test_end = min(test_start + test_size, n_samples)
                
                splits.append((
                    np.arange(train_start, train_end),
                    np.arange(test_start, test_end)
                ))
        
        return splits
    
    def _aggregate_results(self) -> Dict:
        """Aggregate results across all folds."""
        if not self.fold_metrics:
            return {}
        
        # Convert to DataFrame for easier aggregation
        metrics_df = pd.DataFrame(self.fold_metrics)
        
        aggregated = {
            'n_folds': len(self.results),
            'mean_metrics': metrics_df.mean().to_dict(),
            'std_metrics': metrics_df.std().to_dict(),
            'min_metrics': metrics_df.min().to_dict(),
            'max_metrics': metrics_df.max().to_dict(),
            'fold_results': self.results,
            'consistency_score': self._calculate_consistency(),
        }
        
        return aggregated
    
    def _calculate_consistency(self) -> float:
        """
        Calculate consistency score - how often the strategy was profitable
        across folds.
        """
        if not self.fold_metrics:
            return 0.0
        
        profitable_folds = sum(1 for m in self.fold_metrics if m.get('total_return', 0) > 0)
        return profitable_folds / len(self.fold_metrics)
    
    def get_fold_summary(self) -> pd.DataFrame:
        """Get summary of all folds as DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        summary = []
        for result in self.results:
            row = {
                'Fold': result['fold'],
                'Test Period': f"{result['test_start'].strftime('%Y-%m-%d')} to {result['test_end'].strftime('%Y-%m-%d')}",
                'Total Return (%)': result['metrics'].get('total_return_pct', 0),
                'Sharpe Ratio': result['metrics'].get('sharpe_ratio', 0),
                'Max Drawdown (%)': result['metrics'].get('max_drawdown_pct', 0),
                'Win Rate (%)': result['metrics'].get('win_rate_pct', 0),
                'Trades': result['metrics'].get('num_trades', 0),
            }
            summary.append(row)
        
        return pd.DataFrame(summary)
    
    def plot_results(self, figsize: Tuple[int, int] = (14, 10)):
        """
        Plot walk-forward validation results.
        
        Parameters
        ----------
        figsize : Tuple[int, int]
            Figure size
        """
        if not self.results:
            logger.error("No results to plot. Run validation first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Equity curves for each fold
        ax1 = axes[0, 0]
        for result in self.results:
            equity = result['equity_curve']
            # Normalize to start at same value
            normalized = equity / equity.iloc[0] * 100000
            ax1.plot(normalized.index, normalized, alpha=0.7, label=f"Fold {result['fold']}")
        ax1.set_title('Out-of-Sample Equity Curves by Fold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Returns distribution across folds
        ax2 = axes[0, 1]
        returns = [r['metrics'].get('total_return_pct', 0) for r in self.results]
        ax2.bar(range(1, len(returns) + 1), returns, color=['green' if r > 0 else 'red' for r in returns])
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('Total Return by Fold')
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Sharpe ratios
        ax3 = axes[1, 0]
        sharpe_ratios = [r['metrics'].get('sharpe_ratio', 0) for r in self.results]
        ax3.bar(range(1, len(sharpe_ratios) + 1), sharpe_ratios, color='blue', alpha=0.7)
        ax3.axhline(y=1, color='green', linestyle='--', label='Good (1.0)')
        ax3.axhline(y=0, color='red', linestyle='--', label='Breakeven (0)')
        ax3.set_title('Sharpe Ratio by Fold')
        ax3.set_xlabel('Fold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Combined equity curve (concatenated)
        ax4 = axes[1, 1]
        combined_equity = pd.concat([r['equity_curve'] for r in self.results])
        ax4.plot(combined_equity.index, combined_equity, label='Combined OOS', linewidth=2)
        ax4.set_title('Combined Out-of-Sample Equity Curve')
        ax4.set_ylabel('Portfolio Value ($)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def print_summary(self):
        """Print detailed summary of walk-forward results."""
        if not self.results:
            print("No results available. Run validation first.")
            return
        
        print("\n" + "=" * 70)
        print("WALK-FORWARD VALIDATION SUMMARY")
        print("=" * 70)
        
        # Fold details
        print("\nFOLD RESULTS:")
        print("-" * 70)
        summary_df = self.get_fold_summary()
        print(summary_df.to_string(index=False))
        
        # Aggregate statistics
        print("\n" + "=" * 70)
        print("AGGREGATE STATISTICS")
        print("=" * 70)
        
        metrics_df = pd.DataFrame(self.fold_metrics)
        
        stats = pd.DataFrame({
            'Mean': metrics_df.mean(),
            'Std': metrics_df.std(),
            'Min': metrics_df.min(),
            'Max': metrics_df.max(),
        })
        
        print(stats.round(3).to_string())
        
        # Consistency
        consistency = self._calculate_consistency()
        print(f"\nConsistency Score: {consistency:.1%} ({int(consistency * len(self.results))}/{len(self.results)} folds profitable)")
        
        # Interpretation
        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)
        
        mean_sharpe = metrics_df['sharpe_ratio'].mean()
        mean_return = metrics_df['total_return_pct'].mean()
        
        if mean_sharpe > 1.0 and consistency > 0.6:
            print("✓ Strategy shows strong robustness")
        elif mean_sharpe > 0.5 and consistency > 0.5:
            print("~ Strategy shows moderate robustness")
        else:
            print("✗ Strategy may be overfit or not robust")
        
        print(f"\nMean Sharpe Ratio: {mean_sharpe:.2f}")
        print(f"Mean Total Return: {mean_return:.2f}%")
        print(f"Return Std Dev: {metrics_df['total_return_pct'].std():.2f}%")


def simple_parameter_optimization(
    train_data: pd.DataFrame,
    strategy_class,
    param_grid: Dict[str, List],
    metric: str = 'sharpe_ratio',
) -> Dict:
    """
    Simple grid search for parameter optimization.
    
    Parameters
    ----------
    train_data : pd.DataFrame
        Training data
    strategy_class : class
        Strategy class
    param_grid : Dict[str, List]
        Parameter grid to search
    metric : str
        Metric to optimize
        
    Returns
    -------
    Dict
        Best parameters
    """
    from itertools import product
    
    best_score = -np.inf
    best_params = {}
    
    # Generate all parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for combination in product(*values):
        params = dict(zip(keys, combination))
        
        try:
            strategy = strategy_class(**params)
            backtest = Backtester(train_data, strategy, initial_capital=100000)
            result = backtest.run()
            
            score = result['metrics'].get(metric, -np.inf)
            
            if score > best_score:
                best_score = score
                best_params = params
                
        except Exception as e:
            logger.warning(f"Error with params {params}: {e}")
            continue
    
    logger.info(f"Best params: {best_params} (score: {best_score:.3f})")
    return best_params


if __name__ == "__main__":
    # Test walk-forward validation
    print("Testing walk-forward validation...")
    
    from data_ingestion import fetch_data
    from strategy import DualMAStrategy
    
    # Fetch data
    data = fetch_data('AAPL', period='3y')
    
    # Run walk-forward validation
    validator = WalkForwardValidator(
        data=data,
        strategy_class=DualMAStrategy,
        strategy_params={'fast_window': 20, 'slow_window': 50},
        n_splits=3,
    )
    
    results = validator.run()
    validator.print_summary()