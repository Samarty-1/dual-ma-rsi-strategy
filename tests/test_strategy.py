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
        
        # Upper band should be above lower band
        self.assertTrue((signals['upper_band'] > signals['lower_band']).all())


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