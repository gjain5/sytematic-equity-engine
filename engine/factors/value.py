"""
Value factor computation.

Design decisions:
- Multiple value metrics supported (P/E, P/B, EV/EBITDA, etc.)
- Fundamental data is passed in, not fetched (separation of concerns)
- Factor scores are cross-sectional z-scores within the universe
"""

from datetime import date
from typing import List, Optional
import pandas as pd


def compute_value(
    fundamentals: pd.DataFrame,
    as_of_date: date,
    metrics: Optional[List[str]] = None,
) -> pd.Series:
    """
    Compute composite value factor score.
    
    Args:
        fundamentals: DataFrame with symbol index and fundamental metric columns
                     Expected columns: pe_ratio, pb_ratio, ev_ebitda, dividend_yield
        as_of_date: Computation date (for point-in-time fundamental data)
        metrics: List of metrics to include (default: all available)
    
    Returns:
        Series with symbol index and composite value scores
        Higher score = more "value" (lower valuation multiples)
    
    Raises:
        NotImplementedError: Placeholder - actual computation logic TBD
    """
    if metrics is None:
        metrics = ["pe_ratio", "pb_ratio", "ev_ebitda"]
    
    raise NotImplementedError(
        f"Value computation not yet implemented. "
        f"Called with as_of_date={as_of_date}, metrics={metrics}"
    )


def compute_earnings_yield(
    fundamentals: pd.DataFrame,
    as_of_date: date,
) -> pd.Series:
    """
    Compute earnings yield (inverse P/E) factor.
    
    Raises:
        NotImplementedError: Placeholder - actual computation logic TBD
    """
    raise NotImplementedError("Earnings yield computation not yet implemented")


def compute_book_to_market(
    fundamentals: pd.DataFrame,
    as_of_date: date,
) -> pd.Series:
    """
    Compute book-to-market ratio factor.
    
    Raises:
        NotImplementedError: Placeholder - actual computation logic TBD
    """
    raise NotImplementedError("Book-to-market computation not yet implemented")
