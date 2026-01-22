"""
Portfolio module - handles portfolio construction and management.

Design decisions:
- Portfolio is a point-in-time snapshot, not a stateful object
- All operations are pure functions with explicit date parameters
- Weights are normalized and validated
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional
import pandas as pd

from .config import EngineConfig


@dataclass
class Portfolio:
    """
    Represents a portfolio snapshot at a point in time.
    
    Attributes:
        as_of_date: The date this portfolio represents
        holdings: Dict mapping symbol to weight (0-1, summing to 1)
        benchmark: Name of benchmark for comparison
    """
    as_of_date: date
    holdings: Dict[str, float]
    benchmark: str = "nifty50"
    
    def __post_init__(self):
        # Validate weights sum to ~1 (allowing small float errors)
        total_weight = sum(self.holdings.values())
        if abs(total_weight - 1.0) > 0.001 and len(self.holdings) > 0:
            raise ValueError(f"Portfolio weights must sum to 1, got {total_weight}")
    
    @property
    def symbols(self) -> List[str]:
        return list(self.holdings.keys())
    
    @property
    def num_positions(self) -> int:
        return len(self.holdings)


def construct_portfolio(
    factor_scores: pd.Series,
    as_of_date: date,
    config: EngineConfig,
) -> Portfolio:
    """
    Construct a portfolio from factor scores.
    
    Ranks stocks by factor score (descending), selects top N,
    and assigns equal weights.
    
    Args:
        factor_scores: Series with symbol index and factor values (higher = better)
        as_of_date: Portfolio construction date
        config: Engine configuration
    
    Returns:
        Portfolio object with holdings
    """
    if len(factor_scores) == 0:
        raise ValueError("No factor scores provided")
    
    # Sort by factor score descending (higher momentum = better)
    ranked = factor_scores.sort_values(ascending=False)
    
    # Select top N positions
    n_positions = min(config.max_positions, len(ranked))
    top_symbols = ranked.head(n_positions).index.tolist()
    
    # Equal weight allocation
    weight = 1.0 / n_positions
    holdings = {symbol: weight for symbol in top_symbols}
    
    return Portfolio(
        as_of_date=as_of_date,
        holdings=holdings,
        benchmark="nifty50"
    )


def load_portfolio_from_csv(filepath: str) -> Portfolio:
    """
    Load a portfolio from a CSV file.
    
    Expected CSV format:
        symbol,weight,as_of_date
        RELIANCE,0.05,2024-01-15
        ...
    """
    df = pd.read_csv(filepath)
    
    if df.empty:
        raise ValueError(f"Empty portfolio file: {filepath}")
    
    # Parse as_of_date from first row
    as_of_date_str = df["as_of_date"].iloc[0]
    as_of_date = datetime.strptime(as_of_date_str, "%Y-%m-%d").date()
    
    # Build holdings dict
    holdings = dict(zip(df["symbol"], df["weight"]))
    
    return Portfolio(
        as_of_date=as_of_date,
        holdings=holdings
    )


def save_portfolio_to_csv(portfolio: Portfolio, filepath: str) -> None:
    """
    Save a portfolio to a CSV file.
    
    Output format:
        symbol,weight,as_of_date
    """
    rows = []
    for symbol, weight in sorted(portfolio.holdings.items()):
        rows.append({
            "symbol": symbol,
            "weight": round(weight, 6),
            "as_of_date": portfolio.as_of_date.isoformat()
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)


def construct_portfolio_from_scores(
    momentum_scores: pd.Series,
    as_of_date: date,
    max_positions: int = 30,
    min_momentum: Optional[float] = None,
) -> Portfolio:
    """
    Simplified portfolio construction directly from momentum scores.
    
    Args:
        momentum_scores: Series with symbol index and momentum values
        as_of_date: Portfolio construction date
        max_positions: Maximum number of positions (default 30)
        min_momentum: Optional minimum momentum threshold
    
    Returns:
        Portfolio object with equal-weighted holdings
    """
    scores = momentum_scores.copy()
    
    # Filter by minimum momentum if specified
    if min_momentum is not None:
        scores = scores[scores >= min_momentum]
    
    if len(scores) == 0:
        raise ValueError("No stocks pass the momentum filter")
    
    # Rank and select top N
    ranked = scores.sort_values(ascending=False)
    n_positions = min(max_positions, len(ranked))
    top_symbols = ranked.head(n_positions).index.tolist()
    
    # Equal weight
    weight = 1.0 / n_positions
    holdings = {symbol: weight for symbol in top_symbols}
    
    return Portfolio(
        as_of_date=as_of_date,
        holdings=holdings,
        benchmark="nifty50"
    )
