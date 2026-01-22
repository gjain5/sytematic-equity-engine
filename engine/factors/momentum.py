"""
Momentum factor computation.

Design decisions:
- All dates are passed explicitly for reproducibility
- Lookback periods are configurable
- Returns raw factor scores, not ranks (ranking happens in portfolio construction)
"""

from datetime import date
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
        Series with symbol index and momentum scores (returns as decimals)
    """
    # Ensure index is datetime
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    
    # Filter to data up to as_of_date
    as_of_datetime = pd.Timestamp(as_of_date)
    prices = prices[prices.index <= as_of_datetime]
    
    if len(prices) < lookback_days:
        raise ValueError(
            f"Insufficient price history: need {lookback_days} days, have {len(prices)}"
        )
    
    # Get the end date (skip recent days to avoid reversal)
    # and start date (lookback from end date)
    end_idx = len(prices) - skip_recent_days
    start_idx = end_idx - lookback_days
    
    if start_idx < 0:
        raise ValueError("Insufficient price history for lookback calculation")
    
    end_prices = prices.iloc[end_idx]
    start_prices = prices.iloc[start_idx]
    
    # Compute momentum as simple return
    # momentum = (end_price / start_price) - 1
    momentum = (end_prices / start_prices) - 1
    
    # Drop any NaN values (stocks without sufficient history)
    momentum = momentum.dropna()
    
    # Name the series
    momentum.name = "momentum"
    
    return momentum


def compute_momentum_with_volatility_adjustment(
    prices: pd.DataFrame,
    as_of_date: date,
    lookback_days: int = 252,
    skip_recent_days: int = 21,
) -> pd.Series:
    """
    Compute volatility-adjusted momentum (Sharpe-like).
    
    Divides momentum by volatility to favor consistent performers.
    
    Args:
        prices: DataFrame with date index and symbol columns
        as_of_date: Computation date
        lookback_days: Number of trading days for calculation
        skip_recent_days: Number of recent days to skip
    
    Returns:
        Series with symbol index and volatility-adjusted momentum scores
    """
    # Ensure index is datetime
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    
    # Filter to data up to as_of_date
    as_of_datetime = pd.Timestamp(as_of_date)
    prices = prices[prices.index <= as_of_datetime]
    
    # Get the relevant window (excluding recent days)
    end_idx = len(prices) - skip_recent_days
    start_idx = end_idx - lookback_days
    
    if start_idx < 0:
        raise ValueError("Insufficient price history")
    
    window_prices = prices.iloc[start_idx:end_idx]
    
    # Compute daily returns
    returns = window_prices.pct_change().dropna()
    
    # Compute momentum (total return)
    momentum = (window_prices.iloc[-1] / window_prices.iloc[0]) - 1
    
    # Compute volatility (annualized)
    volatility = returns.std() * (252 ** 0.5)
    
    # Volatility-adjusted momentum
    vol_adj_momentum = momentum / volatility.replace(0, float('nan'))
    vol_adj_momentum = vol_adj_momentum.dropna()
    vol_adj_momentum.name = "vol_adj_momentum"
    
    return vol_adj_momentum
