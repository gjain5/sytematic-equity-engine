"""
Universe module - defines the investable asset universe.

Design decisions:
- Universe is loaded explicitly with an as_of_date for reproducibility
- No implicit "today()" calls - all dates must be passed in
- Universe can be extended to support different indices/filters
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
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
    """
    if data_path is None:
        project_root = Path(__file__).parent.parent
        data_path = project_root / "data" / f"{universe_name}.csv"
    else:
        data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Universe file not found: {data_path}")
    
    df = pd.read_csv(data_path, comment="#")
    
    # Filter to symbols that were in the universe as of the given date
    # This enables point-in-time universe membership
    if "added_date" in df.columns:
        df["added_date"] = pd.to_datetime(df["added_date"]).dt.date
        df = df[df["added_date"] <= as_of_date]
    
    symbols = df["symbol"].tolist()
    
    return Universe(
        name=universe_name,
        as_of_date=as_of_date,
        symbols=symbols
    )


def get_available_universes() -> List[str]:
    """
    Return list of supported universe names.
    """
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    if not data_dir.exists():
        return []
    
    universes = []
    for f in data_dir.glob("*.csv"):
        universes.append(f.stem)
    
    return sorted(universes)
