"""
Data Ingestion Module

Handles fetching historical stock data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
from typing import Optional, List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_data(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "5y",
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'SPY')
    start : str, optional
        Start date in 'YYYY-MM-DD' format
    end : str, optional
        End date in 'YYYY-MM-DD' format
    period : str, default '5y'
        Period to download if start/end not specified
        ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    interval : str, default '1d'
        Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
    auto_adjust : bool, default True
        Adjust all OHLC automatically
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume, (Adj Close if not auto_adjust)
        
    Raises
    ------
    ValueError
        If ticker is invalid or data cannot be fetched
    """
    try:
        logger.info(f"Fetching data for {ticker}...")
        
        stock = yf.Ticker(ticker)
        
        # Download data
        if start and end:
            data = stock.history(start=start, end=end, interval=interval, auto_adjust=auto_adjust)
        else:
            data = stock.history(period=period, interval=interval, auto_adjust=auto_adjust)
            
        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
            
        # Clean up column names
        data.columns = [col.replace(" ", "_") for col in data.columns]
        
        # Remove timezone info for cleaner handling
        data.index = data.index.tz_localize(None)
        
        logger.info(f"Successfully fetched {len(data)} rows for {ticker}")
        logger.info(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        raise ValueError(f"Failed to fetch data for {ticker}: {str(e)}")


def fetch_multiple(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "5y",
    interval: str = "1d",
    column: str = "Close",
) -> pd.DataFrame:
    """
    Fetch data for multiple tickers and combine into single DataFrame.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols
    start : str, optional
        Start date in 'YYYY-MM-DD' format
    end : str, optional
        End date in 'YYYY-MM-DD' format
    period : str, default '5y'
        Period to download
    interval : str, default '1d'
        Data interval
    column : str, default 'Close'
        Column to extract for each ticker
        
    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and tickers as columns
    """
    data_dict = {}
    
    for ticker in tickers:
        try:
            df = fetch_data(ticker, start=start, end=end, period=period, interval=interval)
            data_dict[ticker] = df[column]
        except Exception as e:
            logger.warning(f"Skipping {ticker}: {str(e)}")
            continue
    
    if not data_dict:
        raise ValueError("No data could be fetched for any ticker")
        
    combined = pd.DataFrame(data_dict)
    logger.info(f"Combined data shape: {combined.shape}")
    
    return combined


def get_sp500_tickers() -> List[str]:
    """
    Get list of S&P 500 tickers from Wikipedia.
    
    Returns
    -------
    List[str]
        List of S&P 500 ticker symbols
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        tickers = tables[0]['Symbol'].tolist()
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        logger.error(f"Error fetching S&P 500 tickers: {str(e)}")
        # Return a subset of major tickers as fallback
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']


def validate_data(data: pd.DataFrame) -> bool:
    """
    Validate that data is suitable for backtesting.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to validate
        
    Returns
    -------
    bool
        True if data is valid
    """
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Check required columns
    for col in required_columns:
        if col not in data.columns:
            logger.error(f"Missing required column: {col}")
            return False
    
    # Check for missing values
    if data[required_columns].isnull().any().any():
        logger.warning("Data contains missing values")
        # Fill missing values
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
    
    # Check for sufficient data
    if len(data) < 100:
        logger.error("Insufficient data (minimum 100 rows required)")
        return False
    
    # Check for duplicate indices
    if data.index.duplicated().any():
        logger.warning("Duplicate dates found, removing duplicates")
        data = data[~data.index.duplicated(keep='first')]
    
    logger.info("Data validation passed")
    return True


if __name__ == "__main__":
    # Example usage
    print("Testing data ingestion...")
    
    # Fetch single stock
    aapl = fetch_data('AAPL', period='1y')
    print(f"\nAAPL data sample:")
    print(aapl.head())
    
    # Fetch multiple stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    multi_data = fetch_multiple(tickers, period='1y')
    print(f"\nMultiple stocks data sample:")
    print(multi_data.head())