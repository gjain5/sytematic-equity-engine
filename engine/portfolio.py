"""
Portfolio module - handles portfolio construction and management.

Design decisions:
- Portfolio is a point-in-time snapshot, not a stateful object
- All operations are pure functions with explicit date parameters
- Weights are normalized and validated
"""

from dataclasses import dataclass
from datetime import date
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
    factor_scores: pd.DataFrame,
    as_of_date: date,
    config: EngineConfig,
) -> Portfolio:
    """
    Construct a portfolio from factor scores.
    
    Args:
        factor_scores: DataFrame with symbol index and factor columns
        as_of_date: Portfolio construction date
        config: Engine configuration
    
    Returns:
        Portfolio object with holdings
    
    Raises:
        NotImplementedError: Placeholder - actual construction logic TBD
    """
    raise NotImplementedError(
        f"Portfolio construction not yet implemented. "
        f"Called with as_of_date={as_of_date}, max_positions={config.max_positions}"
    )


def load_portfolio_from_csv(filepath: str) -> Portfolio:
    """
    Load a portfolio from a CSV file.
    
    Expected CSV format:
        symbol,weight,as_of_date
        RELIANCE,0.05,2024-01-15
        ...
    
    Raises:
        NotImplementedError: Placeholder - actual loading logic TBD
    """
    raise NotImplementedError(f"Portfolio loading from {filepath} not yet implemented")


def save_portfolio_to_csv(portfolio: Portfolio, filepath: str) -> None:
    """
    Save a portfolio to a CSV file.
    
    Raises:
        NotImplementedError: Placeholder - actual saving logic TBD
    """
    raise NotImplementedError(f"Portfolio saving to {filepath} not yet implemented")
