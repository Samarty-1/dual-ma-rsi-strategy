"""
Strategy Module

Implements trading strategies for backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.signals = None
        self.positions = None
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from data.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data
            
        Returns
        -------
        pd.DataFrame
            Data with added signal columns
        """
        pass
    
    def get_position_sizes(self, data: pd.DataFrame, capital: float) -> pd.Series:
        """
        Calculate position sizes based on available capital.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with signals
        capital : float
            Available capital
            
        Returns
        -------
        pd.Series
            Position sizes (number of shares/contracts)
        """
        # Default: fixed position size
        return pd.Series(1, index=data.index)


class DualMAStrategy(BaseStrategy):
    """
    Dual Moving Average Crossover Strategy with RSI Filter.
    
    Entry: Fast MA crosses above Slow MA AND RSI > 50
    Exit: Fast MA crosses below Slow MA OR RSI > 70 (overbought)
    
    This is a trend-following momentum strategy suitable for students.
    """
    
    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 50,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        position_size: float = 0.1,
    ):
        super().__init__(name="DualMA_Strategy")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.position_size = position_size
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        """
        delta = prices.diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on MA crossover and RSI.
        
        Signal values:
        - 1: Long position (buy)
        - 0: No position (flat)
        - -1: Short position (sell) - optional, set to 0 for long-only
        """
        df = data.copy()
        
        # Calculate moving averages
        df['fast_ma'] = df['Close'].rolling(window=self.fast_window).mean()
        df['slow_ma'] = df['Close'].rolling(window=self.slow_window).mean()
        
        # Calculate RSI
        df['rsi'] = self._calculate_rsi(df['Close'], self.rsi_period)
        
        # Calculate MA crossover
        df['ma_crossover'] = df['fast_ma'] - df['slow_ma']
        df['ma_signal'] = np.where(df['ma_crossover'] > 0, 1, 0)
        
        # Generate entry/exit signals
        # Entry: MA crossover up AND RSI > 50 (momentum confirmation)
        # Exit: MA crossover down OR RSI > 70 (overbought)
        
        df['signal'] = 0
        
        # Long entry conditions
        long_entry = (
            (df['fast_ma'] > df['slow_ma']) &  # Fast above slow
            (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1)) &  # Crossover happened
            (df['rsi'] > 50)  # Momentum confirmation
        )
        
        # Long exit conditions
        long_exit = (
            (df['fast_ma'] < df['slow_ma']) |  # Trend reversal
            (df['rsi'] > self.rsi_overbought)  # Overbought
        )
        
        # Generate positions (1 = long, 0 = flat)
        position = 0
        positions = []
        
        for i in range(len(df)):
            if long_entry.iloc[i] and position == 0:
                position = 1
            elif long_exit.iloc[i] and position == 1:
                position = 0
            positions.append(position)
        
        df['position'] = positions
        
        # Signal for backtester (1 = buy, -1 = sell, 0 = hold)
        df['signal'] = df['position'].diff()
        df.loc[df['position'].diff() > 0, 'signal'] = 1
        df.loc[df['position'].diff() < 0, 'signal'] = -1
        
        # Store for later use
        self.signals = df
        
        logger.info(f"Generated {df['signal'].abs().sum():.0f} signals")
        logger.info(f"Total long entries: {(df['signal'] == 1).sum()}")
        logger.info(f"Total exits: {(df['signal'] == -1).sum()}")
        
        return df
    
    def get_position_sizes(self, data: pd.DataFrame, capital: float) -> pd.Series:
        """
        Calculate position sizes based on volatility (ATR-based sizing).
        
        Uses Average True Range to adjust position size - smaller positions
        for more volatile periods.
        """
        df = data.copy()
        
        # Calculate ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean()
        
        # Position size = (Capital * Position%) / (ATR * Multiplier)
        # This gives more shares when volatility is low, fewer when high
        risk_per_trade = capital * self.position_size
        dollar_volatility = atr * 2  # 2x ATR as risk per share
        
        position_sizes = risk_per_trade / dollar_volatility
        position_sizes = position_sizes.fillna(0)
        
        return position_sizes


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using Bollinger Bands.
    
    Entry: Price touches lower band AND RSI < 30 (oversold)
    Exit: Price touches upper band OR RSI > 50
    
    This strategy bets on price returning to its mean after extreme moves.
    """
    
    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        position_size: float = 0.1,
    ):
        super().__init__(name="MeanReversion_Strategy")
        self.window = window
        self.num_std = num_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.position_size = position_size
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals using Bollinger Bands."""
        df = data.copy()
        
        # Calculate Bollinger Bands
        df['middle_band'] = df['Close'].rolling(window=self.window).mean()
        df['std'] = df['Close'].rolling(window=self.window).std()
        df['upper_band'] = df['middle_band'] + (df['std'] * self.num_std)
        df['lower_band'] = df['middle_band'] - (df['std'] * self.num_std)
        
        # Calculate RSI
        df['rsi'] = self._calculate_rsi(df['Close'], self.rsi_period)
        
        # Generate signals
        # Entry: Price below lower band AND RSI oversold
        # Exit: Price above upper band OR RSI > 50
        
        long_entry = (
            (df['Close'] < df['lower_band']) &
            (df['rsi'] < self.rsi_oversold)
        )
        
        long_exit = (
            (df['Close'] > df['upper_band']) |
            (df['rsi'] > 50)
        )
        
        # Generate positions
        position = 0
        positions = []
        
        for i in range(len(df)):
            if long_entry.iloc[i] and position == 0:
                position = 1
            elif long_exit.iloc[i] and position == 1:
                position = 0
            positions.append(position)
        
        df['position'] = positions
        
        # Signal for backtester
        df['signal'] = 0
        df.loc[df['position'].diff() > 0, 'signal'] = 1
        df.loc[df['position'].diff() < 0, 'signal'] = -1
        
        self.signals = df
        
        logger.info(f"Generated {df['signal'].abs().sum():.0f} signals")
        
        return df
    
    def get_position_sizes(self, data: pd.DataFrame, capital: float) -> pd.Series:
        """Fixed position sizing for mean reversion."""
        current_price = data['Close']
        max_shares = (capital * self.position_size) / current_price
        return max_shares.fillna(0)


class MomentumStrategy(BaseStrategy):
    """
    Price Momentum Strategy using Rate of Change (ROC).
    
    Entry: ROC > threshold (strong momentum) AND volume > average
    Exit: ROC turns negative OR price falls below entry
    
    This strategy rides trends once they establish momentum.
    """
    
    def __init__(
        self,
        roc_period: int = 20,
        roc_threshold: float = 5.0,
        volume_window: int = 20,
        position_size: float = 0.1,
    ):
        super().__init__(name="Momentum_Strategy")
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold
        self.volume_window = volume_window
        self.position_size = position_size
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum signals based on Rate of Change."""
        df = data.copy()
        
        # Calculate Rate of Change
        df['roc'] = ((df['Close'] - df['Close'].shift(self.roc_period)) / 
                     df['Close'].shift(self.roc_period)) * 100
        
        # Calculate volume moving average
        df['volume_ma'] = df['Volume'].rolling(window=self.volume_window).mean()
        
        # Entry: Strong positive momentum with volume confirmation
        long_entry = (
            (df['roc'] > self.roc_threshold) &
            (df['Volume'] > df['volume_ma'])
        )
        
        # Exit: Momentum turns negative
        long_exit = df['roc'] < 0
        
        # Generate positions
        position = 0
        positions = []
        
        for i in range(len(df)):
            if long_entry.iloc[i] and position == 0:
                position = 1
            elif long_exit.iloc[i] and position == 1:
                position = 0
            positions.append(position)
        
        df['position'] = positions
        
        # Signal for backtester
        df['signal'] = 0
        df.loc[df['position'].diff() > 0, 'signal'] = 1
        df.loc[df['position'].diff() < 0, 'signal'] = -1
        
        self.signals = df
        
        logger.info(f"Generated {df['signal'].abs().sum():.0f} signals")
        
        return df
    
    def get_position_sizes(self, data: pd.DataFrame, capital: float) -> pd.Series:
        """Position sizing based on momentum strength."""
        current_price = data['Close']
        base_shares = (capital * self.position_size) / current_price
        
        # Scale position by momentum (stronger momentum = larger position)
        roc = data['roc'].fillna(0)
        momentum_multiplier = 1 + (roc / 100).clip(-0.5, 0.5)
        
        return (base_shares * momentum_multiplier).fillna(0)


if __name__ == "__main__":
    # Test the strategies
    print("Testing strategies...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    sample_data = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # Test Dual MA Strategy
    ma_strategy = DualMAStrategy(fast_window=10, slow_window=20)
    ma_signals = ma_strategy.generate_signals(sample_data)
    print(f"\nDual MA Strategy signals: {(ma_signals['signal'] != 0).sum()}")
    
    # Test Mean Reversion Strategy
    mr_strategy = MeanReversionStrategy()
    mr_signals = mr_strategy.generate_signals(sample_data)
    print(f"Mean Reversion Strategy signals: {(mr_signals['signal'] != 0).sum()}")