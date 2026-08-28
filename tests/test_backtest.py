"""
Unit tests for backtest module.
"""

import unittest
import pandas as pd
import numpy as np
from src.backtest import Backtester, Trade
from src.strategy import DualMAStrategy


class TestBacktester(unittest.TestCase):
    """Test cases for Backtester."""
    
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Create trending price series
        returns = np.random.randn(252) * 0.02 + 0.0005
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.sample_data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 252)
        }, index=dates)
        
        self.strategy = DualMAStrategy(fast_window=20, slow_window=50)
    
    def test_initialization(self):
        """Test backtester initialization."""
        backtest = Backtester(
            self.sample_data,
            self.strategy,
            initial_capital=100000,
            commission=0.001,
        )
        
        self.assertEqual(backtest.initial_capital, 100000)
        self.assertEqual(backtest.commission, 0.001)
        self.assertEqual(len(backtest.trades), 0)
    
    def test_backtest_run(self):
        """Test running a backtest."""
        backtest = Backtester(
            self.sample_data,
            self.strategy,
            initial_capital=100000,
        )
        
        results = backtest.run()
        
        # Check results structure
        self.assertIn('equity_curve', results)
        self.assertIn('positions', results)
        self.assertIn('returns', results)
        self.assertIn('trades', results)
        self.assertIn('metrics', results)
        
        # Check equity curve
        self.assertEqual(len(results['equity_curve']), len(self.sample_data))
        self.assertEqual(results['equity_curve'].iloc[0], 100000)
        
        # Check metrics
        metrics = results['metrics']
        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown_pct', metrics)
    
    def test_trade_recording(self):
        """Test that trades are properly recorded."""
        backtest = Backtester(
            self.sample_data,
            self.strategy,
            initial_capital=100000,
        )
        
        results = backtest.run()
        
        # Check trades
        trades = results['trades']
        self.assertIsInstance(trades, list)
        
        if trades:  # If any trades were made
            trade = trades[0]
            self.assertIsInstance(trade, Trade)
            self.assertIn(trade.side, ['long', 'short'])
            self.assertIsNotNone(trade.entry_date)
            self.assertIsNotNone(trade.entry_price)
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        backtest = Backtester(
            self.sample_data,
            self.strategy,
            initial_capital=100000,
        )
        
        results = backtest.run()
        metrics = results['metrics']
        
        # Check key metrics exist and are valid
        self.assertIsInstance(metrics['total_return'], (int, float))
        self.assertIsInstance(metrics['sharpe_ratio'], (int, float))
        self.assertIsInstance(metrics['max_drawdown_pct'], (int, float))
        
        # Max drawdown should be <= 0
        self.assertLessEqual(metrics['max_drawdown_pct'], 0)
        
        # Win rate should be between 0 and 100 if trades exist
        if metrics['num_trades'] > 0:
            self.assertGreaterEqual(metrics['win_rate_pct'], 0)
            self.assertLessEqual(metrics['win_rate_pct'], 100)
    
    def test_trade_log(self):
        """Test trade log generation."""
        backtest = Backtester(
            self.sample_data,
            self.strategy,
            initial_capital=100000,
        )
        
        backtest.run()
        trade_log = backtest.get_trade_log()
        
        if not trade_log.empty:
            self.assertIn('entry_date', trade_log.columns)
            self.assertIn('exit_date', trade_log.columns)
            self.assertIn('pnl', trade_log.columns)
            self.assertIn('return_pct', trade_log.columns)


class TestTrade(unittest.TestCase):
    """Test cases for Trade dataclass."""
    
    def test_trade_creation(self):
        """Test creating a Trade object."""
        trade = Trade(
            entry_date=pd.Timestamp('2023-01-01'),
            entry_price=100.0,
            shares=10.0,
            side='long'
        )
        
        self.assertEqual(trade.entry_price, 100.0)
        self.assertEqual(trade.shares, 10.0)
        self.assertEqual(trade.side, 'long')
        self.assertEqual(trade.status, 'open')
    
    def test_trade_close(self):
        """Test closing a trade."""
        trade = Trade(
            entry_date=pd.Timestamp('2023-01-01'),
            entry_price=100.0,
            shares=10.0,
            side='long'
        )
        
        trade.exit_date = pd.Timestamp('2023-01-10')
        trade.exit_price = 110.0
        trade.pnl = 100.0
        trade.return_pct = 10.0
        trade.status = 'closed'
        
        self.assertEqual(trade.exit_price, 110.0)
        self.assertEqual(trade.pnl, 100.0)
        self.assertEqual(trade.status, 'closed')


if __name__ == '__main__':
    unittest.main()