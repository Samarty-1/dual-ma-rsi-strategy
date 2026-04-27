"""
Quantitative Trading Strategy Package

A complete end-to-end momentum trading strategy implementation.
"""

__version__ = "1.0.0"
__author__ = "Quantitative Finance Student"

from .data_ingestion import fetch_data, fetch_multiple
from .strategy import DualMAStrategy, MeanReversionStrategy
from .backtest import Backtester
from .metrics import calculate_metrics, PerformanceReport
from .walk_forward import WalkForwardValidator

__all__ = [
    "fetch_data",
    "fetch_multiple", 
    "DualMAStrategy",
    "MeanReversionStrategy",
    "Backtester",
    "calculate_metrics",
    "PerformanceReport",
    "WalkForwardValidator",
]