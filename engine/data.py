"""
Data loading module for price and fundamental data.

Design decisions:
- Core logic works with local files only (no network calls)
- Network calls (yfinance) are isolated to optional fetch functions
- All functions are explicit about data paths and dates
"""

from datetime import date
from pathlib import Path
from typing import List, Optional
import pandas as pd


def load_prices_from_csv(
    filepath: str,
    symbols: Optional[List[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Load price data from a CSV file.
    
    Expected CSV format:
        date,SYMBOL1,SYMBOL2,...
        2024-01-01,100.0,200.0,...
    
    Args:
        filepath: Path to CSV file with price data
        symbols: Optional list of symbols to load (loads all if None)
        start_date: Optional start date filter
        end_date: Optional end date filter
    
    Returns:
        DataFrame with date index and symbol columns
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Filter symbols if specified
    if symbols is not None:
        available = [s for s in symbols if s in df.columns]
        df = df[available]
    
    # Filter date range
    if start_date is not None:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.Timestamp(end_date)]
    
    return df


def save_prices_to_csv(
    prices: pd.DataFrame,
    filepath: str,
) -> None:
    """
    Save price data to a CSV file.
    """
    prices.to_csv(filepath)


def fetch_prices_yfinance(
    symbols: List[str],
    start_date: date,
    end_date: date,
    suffix: str = ".NS",
) -> pd.DataFrame:
    """
    Fetch price data from Yahoo Finance.
    
    This function makes network calls and should only be used
    in scripts/utilities, not in core engine logic.
    
    Args:
        symbols: List of stock symbols
        start_date: Start date for data
        end_date: End date for data
        suffix: Exchange suffix (default ".NS" for NSE)
    
    Returns:
        DataFrame with date index and symbol columns (adjusted close prices)
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance not installed. Run: pip install yfinance"
        )
    
    # Add suffix for NSE symbols
    tickers = [f"{s}{suffix}" for s in symbols]
    
    # Download data
    data = yf.download(
        tickers,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        progress=False,
        auto_adjust=True,
    )
    
    # Extract close prices
    if len(symbols) == 1:
        prices = data["Close"].to_frame(symbols[0])
    else:
        prices = data["Close"]
        # Remove suffix from column names
        prices.columns = [c.replace(suffix, "") for c in prices.columns]
    
    return prices


def get_default_prices_path() -> Path:
    """Get default path for price data."""
    project_root = Path(__file__).parent.parent
    return project_root / "data" / "prices.csv"
