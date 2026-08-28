"""
Unit tests for strategy module.
"""

import unittest
import pandas as pd
import numpy as np
from src.strategy import DualMAStrategy, MeanReversionStrategy, MomentumStrategy


class TestDualMAStrategy(unittest.TestCase):
    """Test cases for DualMAStrategy."""
    
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        self.sample_data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = DualMAStrategy(fast_window=10, slow_window=30)
        self.assertEqual(strategy.fast_window, 10)
        self.assertEqual(strategy.slow_window, 30)
        self.assertEqual(strategy.name, "DualMA_Strategy")
    
    def test_signal_generation(self):
        """Test signal generation."""
        strategy = DualMAStrategy(fast_window=10, slow_window=20)
        signals = strategy.generate_signals(self.sample_data)
        
        # Check that signals DataFrame has expected columns
        self.assertIn('fast_ma', signals.columns)
        self.assertIn('slow_ma', signals.columns)
        self.assertIn('rsi', signals.columns)
        self.assertIn('signal', signals.columns)
        self.assertIn('position', signals.columns)
        
        # Check that signal values are valid (-1, 0, 1)
        self.assertTrue(signals['signal'].isin([-1, 0, 1]).all())
        
        # Check that position values are valid (0 or 1)
        self.assertTrue(signals['position'].isin([0, 1]).all())
    
    def test_position_sizing(self):
        """Test position sizing calculation."""
        strategy = DualMAStrategy(fast_window=10, slow_window=20)
        signals = strategy.generate_signals(self.sample_data)

        position_sizes = strategy.get_position_sizes(signals, capital=100000)

        # Position sizes should be non-negative
        self.assertTrue((position_sizes >= 0).all())

        # Position sizes should be a Series
        self.assertIsInstance(position_sizes, pd.Series)

    def test_atr_column_always_present(self):
        """atr is needed by Backtester's trailing stop regardless of whether
        use_atr_trailing_stop is on, so it must always be computed."""
        strategy = DualMAStrategy(fast_window=10, slow_window=20)
        signals = strategy.generate_signals(self.sample_data)
        self.assertIn('atr', signals.columns)

    def test_rsi_overbought_exit_disabled_when_using_atr_trailing_stop(self):
        """With use_atr_trailing_stop=True, RSI > rsi_overbought must NOT by
        itself flip an open position to flat -- only an MA crossunder
        should. This is the fix for the diagnosed problem: RSI>70 was
        cutting winners within 1-2 days of entry in the real AAPL backtest
        (see README), well before the trend actually reversed."""
        strategy_with_rsi_exit = DualMAStrategy(fast_window=10, slow_window=20, use_atr_trailing_stop=False)
        strategy_without_rsi_exit = DualMAStrategy(fast_window=10, slow_window=20, use_atr_trailing_stop=True)

        signals_a = strategy_with_rsi_exit.generate_signals(self.sample_data)
        signals_b = strategy_without_rsi_exit.generate_signals(self.sample_data)

        # Any bar where RSI is overbought AND the MA hasn't crossed under yet:
        # the RSI-exit variant may show position 0 there (RSI alone caused
        # exit), the trailing-stop variant must never exit from RSI alone.
        still_uptrend = signals_b['fast_ma'] >= signals_b['slow_ma']
        overbought = signals_b['rsi'] > strategy_without_rsi_exit.rsi_overbought
        rsi_only_zone = still_uptrend & overbought
        if rsi_only_zone.any():
            # In this zone, a position entered before it must stay open in
            # the no-RSI-exit variant (still trending, no crossunder).
            in_position_a = signals_a.loc[rsi_only_zone, 'position']
            in_position_b = signals_b.loc[rsi_only_zone, 'position']
            # The trailing-stop variant should never show MORE flat bars
            # than the RSI-exit variant in this zone (it only exits on
            # crossunder here, which is a strict subset of RSI-exit's
            # reasons to be flat).
            self.assertGreaterEqual(in_position_b.sum(), in_position_a.sum())

    def test_use_atr_trailing_stop_default_is_false(self):
        """Backward compatibility: existing callers that don't pass this
        argument must get the original RSI-exit behavior unchanged."""
        strategy = DualMAStrategy(fast_window=10, slow_window=20)
        self.assertFalse(strategy.use_atr_trailing_stop)


class TestMeanReversionStrategy(unittest.TestCase):
    """Test cases for MeanReversionStrategy."""
    
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        self.sample_data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = MeanReversionStrategy(window=20, num_std=2.5)
        self.assertEqual(strategy.window, 20)
        self.assertEqual(strategy.num_std, 2.5)
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        strategy = MeanReversionStrategy()
        signals = strategy.generate_signals(self.sample_data)
        
        # Check Bollinger Bands columns
        self.assertIn('middle_band', signals.columns)
        self.assertIn('upper_band', signals.columns)
        self.assertIn('lower_band', signals.columns)
        
        # Upper band should be above lower band (skip the rolling-window
        # warmup rows, which are legitimately NaN until `window` bars exist)
        valid = signals[['upper_band', 'lower_band']].dropna()
        self.assertTrue((valid['upper_band'] > valid['lower_band']).all())


class TestMomentumStrategy(unittest.TestCase):
    """Test cases for MomentumStrategy."""
    
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        self.sample_data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_roc_calculation(self):
        """Test Rate of Change calculation."""
        strategy = MomentumStrategy(roc_period=20)
        signals = strategy.generate_signals(self.sample_data)
        
        # Check ROC column exists
        self.assertIn('roc', signals.columns)
        
        # ROC should be calculated
        self.assertFalse(signals['roc'].isna().all())


if __name__ == '__main__':
    unittest.main()