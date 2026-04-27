"""
Backtesting Engine Module

Vectorized backtesting engine for strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    shares: float = 0.0
    pnl: float = 0.0
    return_pct: float = 0.0
    side: str = "long"
    status: str = "open"


class Backtester:
    """
    Vectorized backtesting engine.
    
    Simulates trading based on strategy signals and calculates
    portfolio performance over time.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategy,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        use_position_sizing: bool = True,
    ):
        """
        Initialize backtester.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV price data
        strategy : BaseStrategy
            Trading strategy instance
        initial_capital : float
            Starting capital
        commission : float
            Commission per trade (as decimal, e.g., 0.001 = 0.1%)
        slippage : float
            Slippage per trade (as decimal)
        use_position_sizing : bool
            Whether to use strategy's position sizing
        """
        self.data = data.copy()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.use_position_sizing = use_position_sizing
        
        self.trades: List[Trade] = []
        self.equity_curve: Optional[pd.Series] = None
        self.positions: Optional[pd.Series] = None
        self.returns: Optional[pd.Series] = None
        
    def run(self) -> Dict:
        """
        Run the backtest.
        
        Returns
        -------
        Dict
            Dictionary containing backtest results and metrics
        """
        logger.info(f"Starting backtest with {self.strategy.name}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        
        # Generate signals
        signals = self.strategy.generate_signals(self.data)
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0.0  # Number of shares held
        shares = 0.0
        equity = []
        positions = []
        current_trade = None
        
        # Get position sizes if enabled
        if self.use_position_sizing:
            position_sizes = self.strategy.get_position_sizes(signals, self.initial_capital)
        else:
            position_sizes = pd.Series(1, index=signals.index)
        
        # Iterate through each bar
        for i, (date, row) in enumerate(signals.iterrows()):
            signal = row['signal']
            price = row['Close']
            
            # Apply slippage to execution price
            if signal == 1:  # Buy
                exec_price = price * (1 + self.slippage)
            elif signal == -1:  # Sell
                exec_price = price * (1 - self.slippage)
            else:
                exec_price = price
            
            # Process buy signal
            if signal == 1 and position == 0:
                # Calculate shares to buy
                max_shares = position_sizes.iloc[i]
                cost = max_shares * exec_price
                commission_cost = cost * self.commission
                total_cost = cost + commission_cost
                
                if total_cost <= capital:
                    shares = max_shares
                    position = shares
                    capital -= total_cost
                    
                    # Record trade entry
                    current_trade = Trade(
                        entry_date=date,
                        entry_price=exec_price,
                        shares=shares,
                        side="long",
                        status="open"
                    )
                    logger.debug(f"BUY: {shares:.2f} shares @ ${exec_price:.2f} on {date}")
            
            # Process sell signal
            elif signal == -1 and position > 0:
                # Close position
                proceeds = position * exec_price
                commission_cost = proceeds * self.commission
                net_proceeds = proceeds - commission_cost
                
                # Calculate P&L
                entry_cost = current_trade.shares * current_trade.entry_price
                pnl = net_proceeds - entry_cost
                return_pct = (pnl / entry_cost) * 100
                
                # Update trade record
                current_trade.exit_date = date
                current_trade.exit_price = exec_price
                current_trade.pnl = pnl
                current_trade.return_pct = return_pct
                current_trade.status = "closed"
                
                self.trades.append(current_trade)
                
                # Update capital
                capital += net_proceeds
                position = 0
                shares = 0
                current_trade = None
                
                logger.debug(f"SELL: @ ${exec_price:.2f}, P&L: ${pnl:.2f} ({return_pct:.2f}%) on {date}")
            
            # Calculate current equity (mark-to-market)
            current_equity = capital + (position * price)
            equity.append(current_equity)
            positions.append(position)
        
        # Close any open position at the end
        if position > 0:
            final_price = self.data['Close'].iloc[-1]
            proceeds = position * final_price
            commission_cost = proceeds * self.commission
            net_proceeds = proceeds - commission_cost
            
            entry_cost = current_trade.shares * current_trade.entry_price
            pnl = net_proceeds - entry_cost
            return_pct = (pnl / entry_cost) * 100
            
            current_trade.exit_date = self.data.index[-1]
            current_trade.exit_price = final_price
            current_trade.pnl = pnl
            current_trade.return_pct = return_pct
            current_trade.status = "closed"
            
            self.trades.append(current_trade)
            capital += net_proceeds
            
            logger.info(f"Final position closed at ${final_price:.2f}, P&L: ${pnl:.2f}")
        
        # Store results
        self.equity_curve = pd.Series(equity, index=signals.index)
        self.positions = pd.Series(positions, index=signals.index)
        
        # Calculate returns
        self.returns = self.equity_curve.pct_change().fillna(0)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        logger.info(f"Backtest complete. Final equity: ${self.equity_curve.iloc[-1]:,.2f}")
        logger.info(f"Total return: {metrics['total_return_pct']:.2f}%")
        
        return {
            'equity_curve': self.equity_curve,
            'positions': self.positions,
            'returns': self.returns,
            'trades': self.trades,
            'signals': signals,
            'metrics': metrics
        }
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if self.equity_curve is None or len(self.equity_curve) == 0:
            return {}
        
        # Basic metrics
        final_equity = self.equity_curve.iloc[-1]
        total_return = final_equity - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Annualized return
        n_years = len(self.equity_curve) / 252  # Assuming daily data
        annualized_return = ((final_equity / self.initial_capital) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
        
        # Volatility (annualized)
        daily_returns = self.returns
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = daily_returns - (risk_free_rate / 252)
        sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (daily_returns.mean() * 252 - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        rolling_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        max_drawdown_duration = self._calculate_drawdown_duration(drawdown)
        
        # Trade metrics
        if self.trades:
            trades_df = pd.DataFrame([
                {
                    'pnl': t.pnl,
                    'return_pct': t.return_pct,
                    'duration': (t.exit_date - t.entry_date).days if t.exit_date else 0
                }
                for t in self.trades
            ])
            
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            
            num_trades = len(self.trades)
            win_rate = (len(winning_trades) / num_trades * 100) if num_trades > 0 else 0
            
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            profit_factor = (
                winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum())
                if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0
                else float('inf')
            )
            
            avg_trade_return = trades_df['return_pct'].mean()
            avg_trade_duration = trades_df['duration'].mean()
        else:
            num_trades = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_trade_return = 0
            avg_trade_duration = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'annualized_return_pct': annualized_return,
            'volatility_pct': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown,
            'max_drawdown_duration_days': max_drawdown_duration,
            'calmar_ratio': calmar_ratio,
            'num_trades': num_trades,
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade_return_pct': avg_trade_return,
            'avg_trade_duration_days': avg_trade_duration,
            'final_equity': final_equity,
        }
        
        return metrics
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        is_drawdown = drawdown < 0
        if not is_drawdown.any():
            return 0
        
        # Find drawdown periods
        drawdown_starts = is_drawdown & (~is_drawdown.shift(1).fillna(False))
        drawdown_ends = (~is_drawdown) & (is_drawdown.shift(1).fillna(False))
        
        start_indices = drawdown_starts[drawdown_starts].index
        end_indices = drawdown_ends[drawdown_ends].index
        
        if len(start_indices) == 0:
            return 0
        
        # Match starts with ends
        max_duration = 0
        for start in start_indices:
            # Find the next end after this start
            future_ends = end_indices[end_indices > start]
            if len(future_ends) > 0:
                end = future_ends[0]
                duration = (end - start).days
                max_duration = max(max_duration, duration)
            else:
                # Drawdown continues to end of data
                duration = (drawdown.index[-1] - start).days
                max_duration = max(max_duration, duration)
        
        return max_duration
    
    def get_trade_log(self) -> pd.DataFrame:
        """Get trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'shares': t.shares,
                'pnl': t.pnl,
                'return_pct': t.return_pct,
                'duration_days': (t.exit_date - t.entry_date).days if t.exit_date else 0
            }
            for t in self.trades
        ])
    
    def plot_results(self, figsize: Tuple[int, int] = (14, 10)):
        """
        Plot backtest results.
        
        Parameters
        ----------
        figsize : Tuple[int, int]
            Figure size for matplotlib
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib is required for plotting")
            return
        
        if self.equity_curve is None:
            logger.error("No backtest results to plot. Run backtest first.")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot 1: Equity curve and buy-and-hold
        ax1 = axes[0]
        ax1.plot(self.equity_curve.index, self.equity_curve, label='Strategy', linewidth=2)
        
        # Buy and hold benchmark
        bh_returns = self.data['Close'] / self.data['Close'].iloc[0]
        bh_equity = self.initial_capital * bh_returns
        ax1.plot(bh_equity.index, bh_equity, label='Buy & Hold', alpha=0.7, linestyle='--')
        
        ax1.set_title(f'Backtest Results: {self.strategy.name}')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        ax2 = axes[1]
        rolling_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - rolling_max) / rolling_max * 100
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Positions
        ax3 = axes[2]
        ax3.fill_between(self.positions.index, self.positions, 0, alpha=0.3)
        ax3.set_ylabel('Position (Shares)')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Test the backtester
    print("Testing backtester...")
    
    from strategy import DualMAStrategy
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    np.random.seed(42)
    returns = np.random.randn(500) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 500)
    }, index=dates)
    
    # Run backtest
    strategy = DualMAStrategy(fast_window=20, slow_window=50)
    backtest = Backtester(sample_data, strategy, initial_capital=100000)
    results = backtest.run()
    
    print("\nBacktest Results:")
    for key, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")