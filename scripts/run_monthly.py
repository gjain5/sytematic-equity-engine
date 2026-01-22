"""
Monthly rebalance script.

Design decisions:
- Date is passed explicitly, never uses implicit "today()"
- All output goes to artifacts/ directory
- Script is idempotent - can be re-run safely

Usage:
    python -m scripts.run_monthly --as-of-date 2024-01-15
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.config import get_default_config


def parse_date(date_string: str) -> date:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_string, "%Y-%m-%d").date()


def run_monthly_rebalance(as_of_date: date) -> None:
    """
    Execute monthly portfolio rebalance.
    
    Args:
        as_of_date: The date to run the rebalance as of
    
    Raises:
        NotImplementedError: Placeholder - actual logic TBD
    """
    config = get_default_config()
    
    print(f"Running monthly rebalance as of {as_of_date}")
    print(f"Config: max_positions={config.max_positions}, universe={config.universe_name}")
    print(f"Artifacts will be written to: {config.artifacts_path}")
    
    # TODO: Implement actual rebalance logic
    # 1. Load universe as of as_of_date
    # 2. Compute factors (full factor suite for monthly)
    # 3. Construct portfolio with position limits
    # 4. Generate performance report
    # 5. Save to artifacts/
    
    raise NotImplementedError(
        "Monthly rebalance not yet implemented. "
        "This script will compute factors, construct portfolio, and update performance."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run monthly portfolio rebalance"
    )
    parser.add_argument(
        "--as-of-date",
        type=parse_date,
        required=True,
        help="Rebalance date in YYYY-MM-DD format"
    )
    
    args = parser.parse_args()
    run_monthly_rebalance(args.as_of_date)


if __name__ == "__main__":
    main()
