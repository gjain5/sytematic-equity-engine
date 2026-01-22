"""
Multi-period backtest runner.

Design decisions:
- Simulates monthly rebalances over a date range
- Tracks holdings, turnover, and performance across all rebalances
- Computes time-aware robustness metrics
- Answers: "Does momentum survive time, churn, and costs?"

Usage:
    python -m scripts.run_backtest --start-date 2023-01-15 --end-date 2025-01-15
"""

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from engine.config import get_default_config
from engine.universe import load_universe
from engine.factors.momentum import compute_momentum
from engine.portfolio import construct_portfolio_from_scores
from engine.data import load_prices_from_csv, fetch_prices_yfinance, save_prices_to_csv
from engine.performance import fetch_benchmark_prices, compute_performance_summary
from engine.analysis import (
    compute_turnover,
    compute_churn_stats,
    compute_holding_periods,
    compute_holding_period_stats,
    compute_return_contribution,
    compute_concentration_metrics,
    compute_robustness_summary,
)


def parse_date(date_string: str) -> date:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_string, "%Y-%m-%d").date()


def generate_rebalance_dates(start_date: date, end_date: date, day_of_month: int = 15) -> List[date]:
    """
    Generate monthly rebalance dates between start and end.
    
    Args:
        start_date: First possible rebalance date
        end_date: Last possible rebalance date
        day_of_month: Day of month to rebalance (default 15th)
    
    Returns:
        List of rebalance dates
    """
    dates = []
    current = date(start_date.year, start_date.month, day_of_month)
    
    # Adjust if start_date is after the 15th of that month
    if current < start_date:
        if current.month == 12:
            current = date(current.year + 1, 1, day_of_month)
        else:
            current = date(current.year, current.month + 1, day_of_month)
    
    while current <= end_date:
        dates.append(current)
        # Move to next month
        if current.month == 12:
            current = date(current.year + 1, 1, day_of_month)
        else:
            current = date(current.year, current.month + 1, day_of_month)
    
    return dates


def ensure_price_data(
    symbols: list,
    start_date: date,
    end_date: date,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Ensure price data is available for the full backtest period."""
    prices_path = PROJECT_ROOT / "data" / "prices.csv"
    
    # Calculate required lookback (need extra for momentum calculation)
    lookback_start = start_date - timedelta(days=400)
    
    if prices_path.exists() and not force_refresh:
        print(f"Loading existing price data from {prices_path}")
        prices = load_prices_from_csv(str(prices_path))
        
        if len(prices) > 0:
            latest_date = prices.index.max().date()
            earliest_date = prices.index.min().date()
            
            if earliest_date <= lookback_start and latest_date >= end_date:
                available_symbols = [s for s in symbols if s in prices.columns]
                if len(available_symbols) >= len(symbols) * 0.8:
                    print(f"Using cached data: {len(available_symbols)} symbols, {earliest_date} to {latest_date}")
                    return prices[available_symbols]
    
    # Fetch fresh data
    print(f"Fetching price data for {len(symbols)} symbols...")
    try:
        prices = fetch_prices_yfinance(
            symbols=symbols,
            start_date=lookback_start,
            end_date=end_date + timedelta(days=1),
        )
        prices_path.parent.mkdir(parents=True, exist_ok=True)
        save_prices_to_csv(prices, str(prices_path))
        print(f"Saved price data to {prices_path}")
        return prices
    except Exception as e:
        raise RuntimeError(f"Failed to fetch price data: {e}")


def ensure_benchmark_data(
    start_date: date,
    end_date: date,
    force_refresh: bool = False,
) -> pd.Series:
    """Ensure benchmark data is available for the full backtest period."""
    benchmark_path = PROJECT_ROOT / "data" / "benchmark.csv"
    lookback_start = start_date - timedelta(days=400)
    
    if benchmark_path.exists() and not force_refresh:
        df = pd.read_csv(benchmark_path, index_col=0, parse_dates=True)
        if len(df) > 0:
            latest = df.index.max().date()
            earliest = df.index.min().date()
            if earliest <= lookback_start and latest >= end_date:
                return df["benchmark"]
    
    print("Fetching benchmark (Nifty 50) data...")
    benchmark = fetch_benchmark_prices(lookback_start, end_date, symbol="^NSEI")
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"benchmark": benchmark}).to_csv(benchmark_path)
    return benchmark


def run_single_rebalance(
    as_of_date: date,
    prices: pd.DataFrame,
    universe_symbols: List[str],
    prev_holdings: Dict[str, float],
    max_positions: int = 30,
) -> Tuple[Dict[str, float], pd.Series, Dict]:
    """
    Run a single rebalance and return holdings + metrics.
    
    Returns:
        Tuple of (new_holdings, momentum_scores, rebalance_info)
    """
    # Filter to available symbols
    available_symbols = [s for s in universe_symbols if s in prices.columns]
    
    # Compute momentum
    momentum = compute_momentum(
        prices=prices[available_symbols],
        as_of_date=as_of_date,
        lookback_days=252,
        skip_recent_days=21,
    )
    
    # Construct portfolio
    portfolio = construct_portfolio_from_scores(
        momentum_scores=momentum,
        as_of_date=as_of_date,
        max_positions=max_positions,
    )
    
    new_holdings = portfolio.holdings
    
    # Compute turnover
    turnover = compute_turnover(prev_holdings, new_holdings)
    churn = compute_churn_stats(prev_holdings, new_holdings)
    
    # Compute contribution analysis
    contributions = compute_return_contribution(new_holdings, momentum)
    concentration = compute_concentration_metrics(contributions)
    
    rebalance_info = {
        "date": as_of_date,
        "num_positions": len(new_holdings),
        "turnover": turnover,
        "positions_added": churn["positions_added"],
        "positions_removed": churn["positions_removed"],
        "symbols_added": churn["symbols_added"],
        "symbols_removed": churn["symbols_removed"],
        "top5_contribution_pct": concentration.get("top5_pct_of_total", 0),
        "avg_momentum": momentum[list(new_holdings.keys())].mean(),
    }
    
    return new_holdings, momentum, rebalance_info


def simulate_backtest_nav(
    holdings_history: List[Dict[str, float]],
    rebalance_dates: List[date],
    prices: pd.DataFrame,
    cost_bps: float = 50,
) -> Tuple[pd.Series, pd.Series, List[float]]:
    """
    Simulate portfolio NAV across multiple rebalances with transaction costs.
    
    Returns:
        Tuple of (gross_nav, net_nav, turnover_per_rebalance)
    """
    if not holdings_history or not rebalance_dates:
        return pd.Series(), pd.Series(), []
    
    # Ensure holdings_history and rebalance_dates are aligned
    # Use the shorter of the two
    num_periods = min(len(holdings_history), len(rebalance_dates))
    holdings_history = holdings_history[:num_periods]
    rebalance_dates = rebalance_dates[:num_periods]
    
    prices.index = pd.to_datetime(prices.index)
    cost_rate = cost_bps / 10000
    
    # Get date range - start from first valid rebalance with holdings
    start_date = rebalance_dates[0]
    end_date = rebalance_dates[-1]
    
    # Filter prices to date range
    price_mask = (prices.index.date >= start_date) & (prices.index.date <= end_date)
    prices_filtered = prices[price_mask].copy()
    
    if prices_filtered.empty:
        return pd.Series(), pd.Series(), []
    
    # Initialize NAV series
    nav_gross = pd.Series(index=prices_filtered.index, dtype=float)
    nav_net = pd.Series(index=prices_filtered.index, dtype=float)
    
    gross_value = 100.0
    net_value = 100.0
    turnover_list = []
    
    current_holdings = {}
    rebalance_idx = 0
    prev_holdings = {}
    
    for i, (ts, row) in enumerate(prices_filtered.iterrows()):
        current_date = ts.date()
        
        # Check if this is a rebalance date
        if rebalance_idx < len(rebalance_dates) and current_date >= rebalance_dates[rebalance_idx]:
            # Get new holdings (skip empty holdings from failed rebalances)
            new_holdings = holdings_history[rebalance_idx] if rebalance_idx < len(holdings_history) else current_holdings
            
            # Calculate turnover
            turnover = compute_turnover(prev_holdings, new_holdings)
            turnover_list.append(turnover)
            
            # Apply transaction costs to net NAV
            if turnover > 0:
                cost = net_value * turnover * cost_rate
                net_value -= cost
            
            current_holdings = new_holdings
            prev_holdings = new_holdings.copy()
            rebalance_idx += 1
        
        # Calculate daily returns
        if i > 0 and current_holdings:
            daily_return = 0.0
            for symbol, weight in current_holdings.items():
                if symbol in row.index and symbol in prices_filtered.columns:
                    prev_price = prices_filtered[symbol].iloc[i - 1]
                    curr_price = row[symbol]
                    if pd.notna(prev_price) and pd.notna(curr_price) and prev_price > 0:
                        daily_return += weight * (curr_price / prev_price - 1)
            
            gross_value *= (1 + daily_return)
            net_value *= (1 + daily_return)
        
        nav_gross.iloc[i] = gross_value
        nav_net.iloc[i] = net_value
    
    return nav_gross, nav_net, turnover_list


def compute_backtest_metrics(
    nav_gross: pd.Series,
    nav_net: pd.Series,
    benchmark: pd.Series,
    holdings_history: List[Dict[str, float]],
    rebalance_dates: List[date],
    rebalance_info_list: List[Dict],
    turnover_list: List[float],
) -> Dict:
    """Compute comprehensive backtest metrics."""
    
    # Align benchmark
    benchmark.index = pd.to_datetime(benchmark.index)
    common_idx = nav_gross.index.intersection(benchmark.index)
    
    if len(common_idx) == 0:
        return {}
    
    nav_gross_aligned = nav_gross[common_idx]
    nav_net_aligned = nav_net[common_idx]
    benchmark_aligned = benchmark[common_idx]
    
    # Normalize benchmark
    benchmark_nav = (benchmark_aligned / benchmark_aligned.iloc[0]) * 100
    
    # Performance metrics (gross)
    gross_summary = compute_performance_summary(nav_gross_aligned, benchmark_nav)
    
    # Performance metrics (net of costs)
    net_summary = compute_performance_summary(nav_net_aligned, benchmark_nav)
    
    # Turnover statistics
    avg_turnover = np.mean(turnover_list) if turnover_list else 0
    max_turnover = max(turnover_list) if turnover_list else 0
    total_turnover = sum(turnover_list)
    
    # Holding period analysis
    holding_periods = compute_holding_periods(holdings_history, rebalance_dates)
    holding_stats = compute_holding_period_stats(holding_periods)
    
    # Concentration over time
    concentration_history = [info.get("top5_contribution_pct", 0) for info in rebalance_info_list]
    avg_concentration = np.mean(concentration_history) if concentration_history else 0
    
    # Cost impact
    gross_return = gross_summary.get("portfolio_total_return", 0)
    net_return = net_summary.get("portfolio_total_return", 0)
    cost_drag = gross_return - net_return
    cost_drag_pct = (cost_drag / gross_return * 100) if gross_return != 0 else 0
    
    # Time-aware robustness flags
    flags = []
    
    if avg_turnover > 0.4:
        flags.append(f"HIGH_AVG_TURNOVER: {avg_turnover*100:.1f}% average")
    
    if max_turnover > 0.7:
        flags.append(f"HIGH_MAX_TURNOVER: {max_turnover*100:.1f}% peak")
    
    if avg_concentration > 0.5:
        flags.append(f"CONCENTRATED: Top 5 avg {avg_concentration*100:.1f}% of returns")
    
    if cost_drag_pct > 15:
        flags.append(f"COST_SENSITIVE: {cost_drag_pct:.1f}% return lost to costs")
    
    if holding_stats.get("mean_holding_periods", 0) < 2:
        flags.append(f"SHORT_HOLDING: Avg {holding_stats.get('mean_holding_periods', 0):.1f} periods")
    
    # Check if strategy still beats benchmark after costs
    if net_return < gross_summary.get("benchmark_total_return", 0):
        flags.append("UNDERPERFORMS_AFTER_COSTS: Net return < benchmark")
    
    return {
        # Period info
        "start_date": rebalance_dates[0].isoformat() if rebalance_dates else None,
        "end_date": rebalance_dates[-1].isoformat() if rebalance_dates else None,
        "num_rebalances": len(rebalance_dates),
        "trading_days": len(nav_gross),
        
        # Gross performance
        "gross_total_return": round(gross_return, 4),
        "gross_ann_return": round(gross_summary.get("portfolio_ann_return", 0), 4),
        "gross_sharpe": round(gross_summary.get("sharpe_ratio", 0), 2),
        "gross_max_drawdown": round(gross_summary.get("portfolio_max_drawdown", 0), 4),
        
        # Net performance (after costs)
        "net_total_return": round(net_return, 4),
        "net_ann_return": round(net_summary.get("portfolio_ann_return", 0), 4),
        "net_sharpe": round(net_summary.get("sharpe_ratio", 0), 2),
        "net_max_drawdown": round(net_summary.get("portfolio_max_drawdown", 0), 4),
        
        # Benchmark
        "benchmark_total_return": round(gross_summary.get("benchmark_total_return", 0), 4),
        "benchmark_ann_return": round(gross_summary.get("benchmark_ann_return", 0), 4),
        
        # Alpha/Beta (gross)
        "alpha": round(gross_summary.get("alpha", 0), 4),
        "beta": round(gross_summary.get("beta", 1), 4),
        
        # Excess returns
        "gross_excess_return": round(gross_return - gross_summary.get("benchmark_total_return", 0), 4),
        "net_excess_return": round(net_return - gross_summary.get("benchmark_total_return", 0), 4),
        
        # Turnover
        "avg_turnover": round(avg_turnover, 4),
        "max_turnover": round(max_turnover, 4),
        "total_turnover": round(total_turnover, 4),
        
        # Transaction costs
        "cost_bps": 50,
        "total_cost_drag": round(cost_drag, 4),
        "cost_drag_pct": round(cost_drag_pct, 2),
        
        # Holding periods
        "mean_holding_periods": holding_stats.get("mean_holding_periods", 0),
        "median_holding_periods": holding_stats.get("median_holding_periods", 0),
        "unique_symbols_held": holding_stats.get("unique_symbols_ever_held", 0),
        
        # Concentration
        "avg_top5_concentration": round(avg_concentration, 4),
        
        # Robustness
        "robustness_flags": "|".join(flags) if flags else "",
        "is_robust": len(flags) == 0,
        "num_flags": len(flags),
    }


def run_backtest(
    start_date: date,
    end_date: date,
    max_positions: int = 30,
    force_refresh: bool = False,
) -> None:
    """
    Run multi-period backtest.
    
    Args:
        start_date: First rebalance date
        end_date: Last rebalance date
        max_positions: Max positions per rebalance
        force_refresh: Force re-download of data
    """
    config = get_default_config()
    
    print("=" * 70)
    print(f"MULTI-PERIOD BACKTEST: {start_date} to {end_date}")
    print("=" * 70)
    
    # Generate rebalance dates
    rebalance_dates = generate_rebalance_dates(start_date, end_date)
    print(f"Rebalance dates: {len(rebalance_dates)} monthly rebalances")
    print(f"First: {rebalance_dates[0]}, Last: {rebalance_dates[-1]}")
    print()
    
    # Load universe
    print("[1/5] Loading universe...")
    universe = load_universe(config.universe_name, end_date)
    print(f"      Universe: {universe.name} with {len(universe)} symbols")
    
    # Ensure price data for full period
    print("[2/5] Loading price data...")
    prices = ensure_price_data(
        symbols=universe.symbols,
        start_date=rebalance_dates[0],
        end_date=end_date,
        force_refresh=force_refresh,
    )
    available_symbols = [s for s in universe.symbols if s in prices.columns]
    print(f"      Price data for {len(available_symbols)} symbols")
    
    # Ensure benchmark data
    print("[3/5] Loading benchmark data...")
    benchmark = ensure_benchmark_data(rebalance_dates[0], end_date, force_refresh)
    print(f"      Benchmark data: {len(benchmark)} days")
    
    # Run rebalances
    print("[4/5] Running rebalances...")
    holdings_history = []
    rebalance_info_list = []
    prev_holdings = {}
    
    for i, rebal_date in enumerate(rebalance_dates):
        print(f"      [{i+1}/{len(rebalance_dates)}] {rebal_date}...", end=" ")
        
        try:
            holdings, momentum, info = run_single_rebalance(
                as_of_date=rebal_date,
                prices=prices,
                universe_symbols=available_symbols,
                prev_holdings=prev_holdings,
                max_positions=max_positions,
            )
            
            holdings_history.append(holdings)
            rebalance_info_list.append(info)
            
            print(f"turnover={info['turnover']*100:.1f}%, +{info['positions_added']}/-{info['positions_removed']}")
            
            prev_holdings = holdings
            
        except Exception as e:
            print(f"FAILED: {e}")
            # Use previous holdings if rebalance fails
            if prev_holdings:
                holdings_history.append(prev_holdings)
                rebalance_info_list.append({"date": rebal_date, "error": str(e)})
    
    # Simulate NAV
    print("[5/5] Simulating portfolio NAV...")
    nav_gross, nav_net, turnover_list = simulate_backtest_nav(
        holdings_history=holdings_history,
        rebalance_dates=rebalance_dates,
        prices=prices,
        cost_bps=50,
    )
    print(f"      NAV series: {len(nav_gross)} days")
    
    # Compute metrics
    print("\nComputing backtest metrics...")
    metrics = compute_backtest_metrics(
        nav_gross=nav_gross,
        nav_net=nav_net,
        benchmark=benchmark,
        holdings_history=holdings_history,
        rebalance_dates=rebalance_dates,
        rebalance_info_list=rebalance_info_list,
        turnover_list=turnover_list,
    )
    
    # Save artifacts
    config.artifacts_path.mkdir(parents=True, exist_ok=True)
    
    # Save per-rebalance info
    rebalance_df = pd.DataFrame(rebalance_info_list)
    rebalance_path = config.artifacts_path / "rebalances.csv"
    rebalance_df.to_csv(rebalance_path, index=False)
    print(f"Saved rebalance history to {rebalance_path}")
    
    # Save NAV time series
    nav_df = pd.DataFrame({
        "portfolio_nav_gross": nav_gross,
        "portfolio_nav_net": nav_net,
    })
    
    # Add benchmark
    benchmark.index = pd.to_datetime(benchmark.index)
    common_idx = nav_df.index.intersection(benchmark.index)
    benchmark_aligned = benchmark[common_idx]
    benchmark_nav = (benchmark_aligned / benchmark_aligned.iloc[0]) * 100
    nav_df["benchmark_nav"] = benchmark_nav
    
    nav_df.index.name = "date"
    performance_path = config.artifacts_path / "performance.csv"
    nav_df.to_csv(performance_path)
    print(f"Saved performance time series to {performance_path}")
    
    # Save metrics
    metrics_path = config.artifacts_path / "metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Saved backtest metrics to {metrics_path}")
    
    # Save holding periods
    holding_periods = compute_holding_periods(holdings_history, rebalance_dates)
    holding_path = config.artifacts_path / "holding_periods.csv"
    holding_periods.to_csv(holding_path, index=False)
    print(f"Saved holding periods to {holding_path}")
    
    # Print summary
    print()
    print("=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"Period: {metrics['start_date']} to {metrics['end_date']} ({metrics['num_rebalances']} rebalances)")
    print()
    print("  PERFORMANCE (Gross):")
    print(f"    Total Return: {metrics['gross_total_return']*100:.1f}%")
    print(f"    Annualized: {metrics['gross_ann_return']*100:.1f}%")
    print(f"    Sharpe Ratio: {metrics['gross_sharpe']:.2f}")
    print(f"    Max Drawdown: {metrics['gross_max_drawdown']*100:.1f}%")
    print()
    print("  PERFORMANCE (Net of 50bps costs):")
    print(f"    Total Return: {metrics['net_total_return']*100:.1f}%")
    print(f"    Annualized: {metrics['net_ann_return']*100:.1f}%")
    print(f"    Sharpe Ratio: {metrics['net_sharpe']:.2f}")
    print()
    print("  BENCHMARK:")
    print(f"    Total Return: {metrics['benchmark_total_return']*100:.1f}%")
    print(f"    Excess (Gross): {metrics['gross_excess_return']*100:.1f}%")
    print(f"    Excess (Net): {metrics['net_excess_return']*100:.1f}%")
    print()
    print("  TURNOVER & COSTS:")
    print(f"    Avg Turnover: {metrics['avg_turnover']*100:.1f}%")
    print(f"    Max Turnover: {metrics['max_turnover']*100:.1f}%")
    print(f"    Total Turnover: {metrics['total_turnover']*100:.1f}%")
    print(f"    Cost Drag: {metrics['cost_drag_pct']:.1f}% of returns")
    print()
    print("  HOLDING ANALYSIS:")
    print(f"    Mean Holding: {metrics['mean_holding_periods']:.1f} periods")
    print(f"    Median Holding: {metrics['median_holding_periods']:.1f} periods")
    print(f"    Unique Symbols: {metrics['unique_symbols_held']}")
    print()
    print("  ROBUSTNESS:")
    if metrics['is_robust']:
        print("    ✓ No robustness concerns")
    else:
        print(f"    ⚠️  {metrics['num_flags']} flag(s):")
        for flag in metrics['robustness_flags'].split("|"):
            print(f"       - {flag}")
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-period backtest"
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        required=True,
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        required=True,
        help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=30,
        help="Maximum number of positions (default: 30)"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download of price data"
    )
    
    args = parser.parse_args()
    run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        max_positions=args.max_positions,
        force_refresh=args.force_refresh,
    )


if __name__ == "__main__":
    main()
