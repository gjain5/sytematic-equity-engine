"""
Portfolio analysis module for robustness and sanity checks.

Design decisions:
- All analysis is deterministic and reproducible
- Functions are pure with explicit inputs
- Answers: "Is this strategy robust, or just lucky?"
"""

from datetime import date
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def compute_turnover(
    old_holdings: Dict[str, float],
    new_holdings: Dict[str, float],
) -> float:
    """
    Compute portfolio turnover between two snapshots.
    
    Turnover = sum of absolute weight changes / 2
    (Divided by 2 because buys = sells in a fully invested portfolio)
    
    Args:
        old_holdings: Dict mapping symbol to weight (previous period)
        new_holdings: Dict mapping symbol to weight (current period)
    
    Returns:
        Turnover as decimal (0.5 = 50% turnover)
    """
    all_symbols = set(old_holdings.keys()) | set(new_holdings.keys())
    
    total_change = 0.0
    for symbol in all_symbols:
        old_weight = old_holdings.get(symbol, 0.0)
        new_weight = new_holdings.get(symbol, 0.0)
        total_change += abs(new_weight - old_weight)
    
    # Divide by 2: a complete portfolio replacement = 100% turnover
    return total_change / 2


def compute_churn_stats(
    old_holdings: Dict[str, float],
    new_holdings: Dict[str, float],
) -> Dict[str, any]:
    """
    Compute detailed churn statistics.
    
    Returns:
        Dict with:
        - positions_added: count of new positions
        - positions_removed: count of exited positions
        - positions_kept: count of unchanged positions
        - pct_replaced: percentage of portfolio replaced
    """
    old_symbols = set(old_holdings.keys())
    new_symbols = set(new_holdings.keys())
    
    added = new_symbols - old_symbols
    removed = old_symbols - new_symbols
    kept = old_symbols & new_symbols
    
    total_positions = len(new_symbols)
    pct_replaced = len(added) / total_positions if total_positions > 0 else 0
    
    return {
        "positions_added": len(added),
        "positions_removed": len(removed),
        "positions_kept": len(kept),
        "pct_replaced": round(pct_replaced, 4),
        "symbols_added": list(added),
        "symbols_removed": list(removed),
    }


def compute_return_contribution(
    holdings: Dict[str, float],
    returns: pd.Series,
) -> pd.DataFrame:
    """
    Compute return contribution by position.
    
    Args:
        holdings: Dict mapping symbol to weight
        returns: Series with symbol index and period returns
    
    Returns:
        DataFrame with columns: symbol, weight, return, contribution
        sorted by contribution descending
    """
    rows = []
    for symbol, weight in holdings.items():
        ret = returns.get(symbol, 0.0)
        contribution = weight * ret
        rows.append({
            "symbol": symbol,
            "weight": weight,
            "return": ret,
            "contribution": contribution,
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("contribution", ascending=False)
    
    return df


def compute_concentration_metrics(
    contributions: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute concentration risk metrics.
    
    Args:
        contributions: DataFrame from compute_return_contribution
    
    Returns:
        Dict with concentration metrics
    """
    if contributions.empty:
        return {}
    
    total_contribution = contributions["contribution"].sum()
    
    # Top 5 contribution
    top5 = contributions.head(5)
    top5_contribution = top5["contribution"].sum()
    
    # Bottom 5 (worst performers)
    bottom5 = contributions.tail(5)
    bottom5_contribution = bottom5["contribution"].sum()
    
    # Rest of portfolio
    rest = contributions.iloc[5:-5] if len(contributions) > 10 else pd.DataFrame()
    rest_contribution = rest["contribution"].sum() if not rest.empty else 0
    
    # Contribution Herfindahl (concentration measure)
    if total_contribution != 0:
        contrib_shares = (contributions["contribution"] / total_contribution).abs()
        herfindahl = (contrib_shares ** 2).sum()
    else:
        herfindahl = 0
    
    return {
        "total_return": round(total_contribution, 4),
        "top5_contribution": round(top5_contribution, 4),
        "top5_pct_of_total": round(top5_contribution / total_contribution, 4) if total_contribution != 0 else 0,
        "bottom5_contribution": round(bottom5_contribution, 4),
        "contribution_herfindahl": round(herfindahl, 4),
        "top_contributor": contributions.iloc[0]["symbol"] if len(contributions) > 0 else None,
        "worst_contributor": contributions.iloc[-1]["symbol"] if len(contributions) > 0 else None,
    }


def compute_holding_periods(
    portfolio_history: List[Dict[str, float]],
    dates: List[date],
) -> pd.DataFrame:
    """
    Track how long each symbol stays in portfolio.
    
    Args:
        portfolio_history: List of holdings dicts (oldest to newest)
        dates: Corresponding rebalance dates
    
    Returns:
        DataFrame with columns: symbol, first_held, last_held, periods_held, still_held
    """
    if not portfolio_history:
        return pd.DataFrame()
    
    # Track first and last appearance of each symbol
    symbol_periods = {}
    
    for i, holdings in enumerate(portfolio_history):
        rebal_date = dates[i]
        for symbol in holdings.keys():
            if symbol not in symbol_periods:
                symbol_periods[symbol] = {
                    "first_held": rebal_date,
                    "last_held": rebal_date,
                    "periods_held": 1,
                }
            else:
                symbol_periods[symbol]["last_held"] = rebal_date
                symbol_periods[symbol]["periods_held"] += 1
    
    # Check if still held in latest portfolio
    latest_holdings = portfolio_history[-1]
    latest_date = dates[-1]
    
    rows = []
    for symbol, info in symbol_periods.items():
        rows.append({
            "symbol": symbol,
            "first_held": info["first_held"],
            "last_held": info["last_held"],
            "periods_held": info["periods_held"],
            "still_held": symbol in latest_holdings,
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("periods_held", ascending=False)
    
    return df


def compute_holding_period_stats(holding_periods: pd.DataFrame) -> Dict[str, float]:
    """
    Compute summary statistics for holding periods.
    """
    if holding_periods.empty:
        return {}
    
    periods = holding_periods["periods_held"]
    
    return {
        "mean_holding_periods": round(periods.mean(), 2),
        "median_holding_periods": round(periods.median(), 2),
        "max_holding_periods": int(periods.max()),
        "min_holding_periods": int(periods.min()),
        "unique_symbols_ever_held": len(holding_periods),
        "symbols_held_once": int((periods == 1).sum()),
        "symbols_held_continuously": int((holding_periods["still_held"] & (periods == periods.max())).sum()),
    }


def apply_transaction_costs(
    nav_series: pd.Series,
    turnover_series: pd.Series,
    cost_bps: float = 50,
) -> pd.Series:
    """
    Apply transaction costs to NAV series.
    
    Args:
        nav_series: Original NAV series
        turnover_series: Turnover at each rebalance (0 for non-rebalance days)
        cost_bps: Transaction cost in basis points (default 50 = 0.5%)
    
    Returns:
        Adjusted NAV series after costs
    """
    cost_rate = cost_bps / 10000  # Convert bps to decimal
    
    # Apply costs: NAV reduced by (turnover * cost_rate) at each rebalance
    adjusted_nav = nav_series.copy()
    
    for i, turnover in enumerate(turnover_series):
        if turnover > 0 and i < len(adjusted_nav):
            cost_drag = turnover * cost_rate
            # Apply cost as multiplicative drag
            adjusted_nav.iloc[i:] = adjusted_nav.iloc[i:] * (1 - cost_drag)
    
    return adjusted_nav


def simulate_with_costs(
    portfolio_nav: pd.Series,
    rebalance_dates: List[date],
    turnover_per_rebalance: List[float],
    cost_bps: float = 50,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Simulate portfolio NAV with transaction costs.
    
    Args:
        portfolio_nav: Original NAV series
        rebalance_dates: List of rebalance dates
        turnover_per_rebalance: Turnover at each rebalance
        cost_bps: Cost in basis points
    
    Returns:
        Tuple of (adjusted NAV series, cost summary dict)
    """
    adjusted_nav = portfolio_nav.copy()
    total_costs = 0.0
    cost_rate = cost_bps / 10000
    
    for rebal_date, turnover in zip(rebalance_dates, turnover_per_rebalance):
        if turnover > 0:
            rebal_ts = pd.Timestamp(rebal_date)
            if rebal_ts in adjusted_nav.index:
                idx = adjusted_nav.index.get_loc(rebal_ts)
                cost_drag = turnover * cost_rate
                nav_at_rebal = adjusted_nav.iloc[idx]
                cost_amount = nav_at_rebal * cost_drag
                total_costs += cost_amount
                # Apply cost to all subsequent NAV values
                adjusted_nav.iloc[idx:] = adjusted_nav.iloc[idx:] * (1 - cost_drag)
    
    original_return = (portfolio_nav.iloc[-1] / portfolio_nav.iloc[0]) - 1
    adjusted_return = (adjusted_nav.iloc[-1] / adjusted_nav.iloc[0]) - 1
    
    cost_summary = {
        "cost_bps": cost_bps,
        "total_turnover": sum(turnover_per_rebalance),
        "avg_turnover_per_rebalance": np.mean(turnover_per_rebalance) if turnover_per_rebalance else 0,
        "original_return": round(original_return, 4),
        "adjusted_return": round(adjusted_return, 4),
        "return_drag_from_costs": round(original_return - adjusted_return, 4),
        "return_drag_pct": round((original_return - adjusted_return) / original_return * 100, 2) if original_return != 0 else 0,
    }
    
    return adjusted_nav, cost_summary


def compute_robustness_summary(
    turnover_history: List[float],
    holding_period_stats: Dict[str, float],
    concentration_metrics: Dict[str, float],
    cost_summary: Dict[str, float],
) -> Dict[str, any]:
    """
    Compute overall robustness summary.
    
    Returns assessment of strategy robustness.
    """
    avg_turnover = np.mean(turnover_history) if turnover_history else 0
    max_turnover = max(turnover_history) if turnover_history else 0
    
    # Robustness flags
    flags = []
    
    # High turnover warning
    if avg_turnover > 0.5:
        flags.append("HIGH_TURNOVER: Average turnover > 50%")
    
    # Concentration risk
    if concentration_metrics.get("top5_pct_of_total", 0) > 0.8:
        flags.append("CONCENTRATED: Top 5 positions drive > 80% of returns")
    
    # Cost sensitivity
    if cost_summary.get("return_drag_pct", 0) > 20:
        flags.append("COST_SENSITIVE: Transaction costs reduce returns by > 20%")
    
    # Short holding periods
    if holding_period_stats.get("mean_holding_periods", 0) < 2:
        flags.append("SHORT_HOLDING: Average holding < 2 periods")
    
    return {
        "avg_turnover": round(avg_turnover, 4),
        "max_turnover": round(max_turnover, 4),
        "mean_holding_periods": holding_period_stats.get("mean_holding_periods", 0),
        "top5_contribution_pct": concentration_metrics.get("top5_pct_of_total", 0),
        "cost_drag_pct": cost_summary.get("return_drag_pct", 0),
        "robustness_flags": flags,
        "is_robust": len(flags) == 0,
    }
