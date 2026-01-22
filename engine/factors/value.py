"""
Value factor computation.

Design decisions:
- Uses Earnings Yield (E/P) and Book-to-Price (B/P)
- Fundamental data fetched via yfinance
- Factor scores are cross-sectional z-scores within the universe
- Higher score = more value (cheaper stock)
- Missing data handled conservatively (excluded from ranking)
"""

from datetime import date
from typing import List, Optional
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None


def fetch_fundamentals(
    symbols: List[str],
    as_of_date: date,
) -> pd.DataFrame:
    """
    Fetch fundamental data for symbols using yfinance.
    
    Args:
        symbols: List of stock symbols
        as_of_date: Date for data (uses latest available trailing data)
    
    Returns:
        DataFrame with columns: symbol, earnings_yield, book_to_price, pe_ratio, pb_ratio
    """
    if yf is None:
        raise ImportError("yfinance required for fundamental data")
    
    rows = []
    
    # Add .NS suffix for NSE stocks
    yf_symbols = [f"{s}.NS" for s in symbols]
    
    # Fetch in batches for efficiency
    batch_size = 50
    for i in range(0, len(yf_symbols), batch_size):
        batch = yf_symbols[i:i + batch_size]
        batch_original = symbols[i:i + batch_size]
        
        try:
            tickers = yf.Tickers(" ".join(batch))
            
            for yf_sym, orig_sym in zip(batch, batch_original):
                try:
                    ticker = tickers.tickers.get(yf_sym)
                    if ticker is None:
                        continue
                    
                    info = ticker.info
                    
                    # Get P/E ratio (trailing)
                    pe_ratio = info.get("trailingPE")
                    
                    # Get P/B ratio
                    pb_ratio = info.get("priceToBook")
                    
                    # Compute inverse metrics (higher = more value)
                    earnings_yield = 1 / pe_ratio if pe_ratio and pe_ratio > 0 else None
                    book_to_price = 1 / pb_ratio if pb_ratio and pb_ratio > 0 else None
                    
                    rows.append({
                        "symbol": orig_sym,
                        "pe_ratio": pe_ratio,
                        "pb_ratio": pb_ratio,
                        "earnings_yield": earnings_yield,
                        "book_to_price": book_to_price,
                    })
                except Exception:
                    # Skip symbols with errors
                    continue
        except Exception:
            continue
    
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "pe_ratio", "pb_ratio", "earnings_yield", "book_to_price"])
    
    return df


def zscore_normalize(series: pd.Series) -> pd.Series:
    """
    Cross-sectional z-score normalization.
    
    Args:
        series: Raw factor values
    
    Returns:
        Z-scored values (mean=0, std=1)
    """
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return series * 0  # All same value, return zeros
    return (series - mean) / std


def compute_value(
    fundamentals: pd.DataFrame,
    as_of_date: date,
    weights: Optional[dict] = None,
) -> pd.Series:
    """
    Compute composite value factor score.
    
    Combines Earnings Yield and Book-to-Price using z-score normalization
    and equal weighting (or custom weights).
    
    Args:
        fundamentals: DataFrame with symbol, earnings_yield, book_to_price columns
        as_of_date: Computation date
        weights: Dict with weights for each metric (default: equal weight)
    
    Returns:
        Series with symbol index and composite value scores
        Higher score = more "value" (cheaper stock)
    """
    if weights is None:
        weights = {"earnings_yield": 0.5, "book_to_price": 0.5}
    
    df = fundamentals.copy()
    
    # Filter to rows with valid data
    required_cols = ["earnings_yield", "book_to_price"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Drop rows with missing values (conservative handling)
    df = df.dropna(subset=required_cols)
    
    if df.empty:
        return pd.Series(dtype=float, name="value")
    
    # Z-score normalize each metric
    df["ey_zscore"] = zscore_normalize(df["earnings_yield"])
    df["bp_zscore"] = zscore_normalize(df["book_to_price"])
    
    # Compute composite score
    df["value_score"] = (
        weights.get("earnings_yield", 0.5) * df["ey_zscore"] +
        weights.get("book_to_price", 0.5) * df["bp_zscore"]
    )
    
    # Return as Series with symbol index
    result = df.set_index("symbol")["value_score"]
    result.name = "value"
    
    return result


def compute_value_from_symbols(
    symbols: List[str],
    as_of_date: date,
    weights: Optional[dict] = None,
) -> pd.Series:
    """
    Convenience function: fetch fundamentals and compute value in one call.
    
    Args:
        symbols: List of stock symbols
        as_of_date: Computation date
        weights: Optional custom weights
    
    Returns:
        Series with symbol index and value scores
    """
    print(f"      Fetching fundamentals for {len(symbols)} symbols...")
    fundamentals = fetch_fundamentals(symbols, as_of_date)
    print(f"      Received fundamentals for {len(fundamentals)} symbols")
    
    if fundamentals.empty:
        return pd.Series(dtype=float, name="value")
    
    return compute_value(fundamentals, as_of_date, weights)


def compute_earnings_yield(
    fundamentals: pd.DataFrame,
    as_of_date: date,
) -> pd.Series:
    """
    Compute earnings yield (E/P) factor only.
    
    Args:
        fundamentals: DataFrame with symbol and earnings_yield columns
        as_of_date: Computation date
    
    Returns:
        Series with symbol index and earnings yield z-scores
    """
    df = fundamentals.copy()
    df = df.dropna(subset=["earnings_yield"])
    
    if df.empty:
        return pd.Series(dtype=float, name="earnings_yield")
    
    df["ey_zscore"] = zscore_normalize(df["earnings_yield"])
    
    result = df.set_index("symbol")["ey_zscore"]
    result.name = "earnings_yield"
    
    return result


def compute_book_to_price(
    fundamentals: pd.DataFrame,
    as_of_date: date,
) -> pd.Series:
    """
    Compute book-to-price (B/P) factor only.
    
    Args:
        fundamentals: DataFrame with symbol and book_to_price columns
        as_of_date: Computation date
    
    Returns:
        Series with symbol index and book-to-price z-scores
    """
    df = fundamentals.copy()
    df = df.dropna(subset=["book_to_price"])
    
    if df.empty:
        return pd.Series(dtype=float, name="book_to_price")
    
    df["bp_zscore"] = zscore_normalize(df["book_to_price"])
    
    result = df.set_index("symbol")["bp_zscore"]
    result.name = "book_to_price"
    
    return result
