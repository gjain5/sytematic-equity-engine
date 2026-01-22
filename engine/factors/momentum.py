"""
Momentum factor computation.

Design decisions:
- All dates are passed explicitly for reproducibility
- Lookback periods are configurable
- Returns raw factor scores, not ranks (ranking happens in portfolio construction)
"""

from datetime import date
from typing import Optional
import pandas as pd


def compute_momentum(
    prices: pd.DataFrame,
    as_of_date: date,
    lookback_days: int = 252,
    skip_recent_days: int = 21,
) -> pd.Series:
    """
    Compute momentum factor scores for all symbols.
    
    Standard momentum = return over lookback period, skipping most recent month
    to avoid short-term reversal effects.
    
    Args:
        prices: DataFrame with date index and symbol columns, containing adjusted close prices
        as_of_date: Computation date (only data up to this date is used)
        lookback_days: Number of trading days for momentum calculation (default: 12 months)
        skip_recent_days: Number of recent days to skip (default: 1 month)
    
    Returns:
        Series with symbol index and momentum scores
    
    Raises:
        NotImplementedError: Placeholder - actual computation logic TBD
    """
    raise NotImplementedError(
        f"Momentum computation not yet implemented. "
        f"Called with as_of_date={as_of_date}, lookback={lookback_days}, skip={skip_recent_days}"
    )


def compute_momentum_with_volatility_adjustment(
    prices: pd.DataFrame,
    as_of_date: date,
    lookback_days: int = 252,
) -> pd.Series:
    """
    Compute volatility-adjusted momentum (Sharpe-like).
    
    Raises:
        NotImplementedError: Placeholder - actual computation logic TBD
    """
    raise NotImplementedError("Volatility-adjusted momentum not yet implemented")
