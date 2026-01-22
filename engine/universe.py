"""
Universe module - defines the investable asset universe.

Design decisions:
- Universe is loaded explicitly with an as_of_date for reproducibility
- No implicit "today()" calls - all dates must be passed in
- Universe can be extended to support different indices/filters
"""

from dataclasses import dataclass
from datetime import date
from typing import List, Optional
import pandas as pd


@dataclass
class Universe:
    """
    Represents an investable universe of securities.
    
    The universe is point-in-time to avoid lookahead bias.
    """
    name: str
    as_of_date: date
    symbols: List[str]
    
    def __len__(self) -> int:
        return len(self.symbols)
    
    def __contains__(self, symbol: str) -> bool:
        return symbol in self.symbols


def load_universe(
    universe_name: str,
    as_of_date: date,
    data_path: Optional[str] = None
) -> Universe:
    """
    Load universe constituents as of a specific date.
    
    Args:
        universe_name: Name of the universe (e.g., "nifty500")
        as_of_date: Point-in-time date for universe membership
        data_path: Optional path to universe data file
    
    Returns:
        Universe object with constituents
    
    Raises:
        NotImplementedError: Placeholder - actual loading logic TBD
    """
    raise NotImplementedError(
        f"Universe loading not yet implemented. "
        f"Called with: universe_name={universe_name}, as_of_date={as_of_date}"
    )


def get_available_universes() -> List[str]:
    """
    Return list of supported universe names.
    
    Raises:
        NotImplementedError: Placeholder - actual logic TBD
    """
    raise NotImplementedError("Available universes lookup not yet implemented")
