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
from datetime import date, datetime, timedelta
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from engine.config import get_default_config
from engine.universe import load_universe
from engine.factors.momentum import compute_momentum
from engine.portfolio import construct_portfolio_from_scores, save_portfolio_to_csv
from engine.data import load_prices_from_csv, fetch_prices_yfinance, save_prices_to_csv
from engine.performance import (
    fetch_benchmark_prices,
    simulate_portfolio_nav,
    compute_drawdown,
    compute_rolling_return,
    compute_performance_summary,
)
from engine.analysis import (
    compute_turnover,
    compute_churn_stats,
    compute_return_contribution,
    compute_concentration_metrics,
    simulate_with_costs,
    compute_robustness_summary,
)


def parse_date(date_string: str) -> date:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_string, "%Y-%m-%d").date()


def ensure_price_data(
    symbols: list,
    as_of_date: date,
    lookback_days: int = 400,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Ensure price data is available, fetching from yfinance if needed.
    
    Args:
        symbols: List of symbols to get prices for
        as_of_date: End date for price data
        lookback_days: Number of calendar days of history needed
        force_refresh: Force re-download even if data exists
    
    Returns:
        DataFrame with price data
    """
    prices_path = PROJECT_ROOT / "data" / "prices.csv"
    
    # Try to load existing data
    if prices_path.exists() and not force_refresh:
        print(f"Loading existing price data from {prices_path}")
        prices = load_prices_from_csv(str(prices_path))
        
        # Check if we have sufficient data
        if len(prices) > 0:
            latest_date = prices.index.max().date()
            earliest_date = prices.index.min().date()
            
            # Check coverage
            required_start = as_of_date - timedelta(days=lookback_days)
            if earliest_date <= required_start and latest_date >= as_of_date:
                # Filter to symbols we need
                available_symbols = [s for s in symbols if s in prices.columns]
                if len(available_symbols) >= len(symbols) * 0.8:  # At least 80% coverage
                    print(f"Using cached data: {len(available_symbols)} symbols, {earliest_date} to {latest_date}")
                    return prices[available_symbols]
    
    # Fetch fresh data
    print(f"Fetching price data for {len(symbols)} symbols...")
    start_date = as_of_date - timedelta(days=lookback_days + 60)  # Extra buffer
    
    try:
        prices = fetch_prices_yfinance(
            symbols=symbols,
            start_date=start_date,
            end_date=as_of_date + timedelta(days=1),
        )
        
        # Save for future use
        prices_path.parent.mkdir(parents=True, exist_ok=True)
        save_prices_to_csv(prices, str(prices_path))
        print(f"Saved price data to {prices_path}")
        
        return prices
    except Exception as e:
        raise RuntimeError(f"Failed to fetch price data: {e}")


def run_monthly_rebalance(
    as_of_date: date,
    max_positions: int = 30,
    force_refresh: bool = False,
) -> None:
    """
    Execute monthly portfolio rebalance.
    
    Pipeline:
    1. Load universe as of as_of_date
    2. Ensure price data is available
    3. Compute momentum factor
    4. Construct portfolio (top N by momentum, equal weight)
    5. Save portfolio to artifacts/
    6. Update performance history
    
    Args:
        as_of_date: The date to run the rebalance as of
        max_positions: Maximum number of positions in portfolio
        force_refresh: Force re-download of price data
    """
    config = get_default_config()
    
    print("=" * 60)
    print(f"MONTHLY REBALANCE: {as_of_date}")
    print("=" * 60)
    print(f"Config: max_positions={max_positions}, universe={config.universe_name}")
    print(f"Artifacts path: {config.artifacts_path}")
    print()
    
    # Step 1: Load universe
    print("[1/5] Loading universe...")
    universe = load_universe(config.universe_name, as_of_date)
    print(f"      Universe: {universe.name} with {len(universe)} symbols")
    
    # Step 2: Ensure price data
    print("[2/5] Loading price data...")
    prices = ensure_price_data(
        symbols=universe.symbols,
        as_of_date=as_of_date,
        force_refresh=force_refresh,
    )
    available_symbols = [s for s in universe.symbols if s in prices.columns]
    print(f"      Price data available for {len(available_symbols)}/{len(universe)} symbols")
    
    # Step 3: Compute momentum
    print("[3/5] Computing momentum factor...")
    momentum = compute_momentum(
        prices=prices[available_symbols],
        as_of_date=as_of_date,
        lookback_days=252,  # 12 months
        skip_recent_days=21,  # Skip last month
    )
    print(f"      Momentum computed for {len(momentum)} symbols")
    print(f"      Top 5: {momentum.nlargest(5).to_dict()}")
    print(f"      Bottom 5: {momentum.nsmallest(5).to_dict()}")
    
    # Step 4: Construct portfolio
    print("[4/5] Constructing portfolio...")
    portfolio = construct_portfolio_from_scores(
        momentum_scores=momentum,
        as_of_date=as_of_date,
        max_positions=max_positions,
    )
    print(f"      Portfolio: {portfolio.num_positions} positions, equal weight {1/portfolio.num_positions:.2%}")
    
    # Step 5: Save portfolio
    print("[5/6] Saving portfolio...")
    config.artifacts_path.mkdir(parents=True, exist_ok=True)
    
    portfolio_path = config.portfolio_path
    save_portfolio_to_csv(portfolio, str(portfolio_path))
    print(f"      Saved portfolio to {portfolio_path}")
    
    # Step 6: Fetch benchmark and compute performance
    print("[6/6] Computing performance metrics...")
    benchmark = ensure_benchmark_data(as_of_date, force_refresh=force_refresh)
    compute_and_save_performance(portfolio, prices, benchmark, momentum, config)
    
    print()
    print("=" * 60)
    print("REBALANCE COMPLETE")
    print("=" * 60)
    print(f"Portfolio saved: {portfolio_path}")
    print(f"Holdings: {list(portfolio.holdings.keys())}")


def ensure_benchmark_data(
    as_of_date: date,
    lookback_days: int = 400,
    force_refresh: bool = False,
) -> pd.Series:
    """
    Ensure benchmark (Nifty 50) data is available.
    """
    benchmark_path = PROJECT_ROOT / "data" / "benchmark.csv"
    
    if benchmark_path.exists() and not force_refresh:
        df = pd.read_csv(benchmark_path, index_col=0, parse_dates=True)
        if len(df) > 0:
            latest = df.index.max().date()
            earliest = df.index.min().date()
            required_start = as_of_date - timedelta(days=lookback_days)
            if earliest <= required_start and latest >= as_of_date:
                return df["benchmark"]
    
    print("      Fetching benchmark (Nifty 50) data...")
    start_date = as_of_date - timedelta(days=lookback_days + 60)
    benchmark = fetch_benchmark_prices(start_date, as_of_date, symbol="^NSEI")
    
    # Save
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"benchmark": benchmark}).to_csv(benchmark_path)
    
    return benchmark


def compute_and_save_performance(
    portfolio,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    momentum: pd.Series,
    config,
) -> None:
    """
    Compute full performance metrics and save to artifacts.
    
    Generates:
    - performance.csv: Time series of portfolio vs benchmark NAV
    - metrics.csv: Summary statistics
    """
    as_of_date = portfolio.as_of_date
    
    # Load portfolio as DataFrame
    portfolio_df = pd.DataFrame([
        {"symbol": s, "weight": w, "as_of_date": as_of_date.isoformat()}
        for s, w in portfolio.holdings.items()
    ])
    
    # Determine simulation period (use available price history)
    prices.index = pd.to_datetime(prices.index)
    benchmark.index = pd.to_datetime(benchmark.index)
    
    # Find common date range
    price_start = prices.index.min()
    price_end = prices.index.max()
    bench_start = benchmark.index.min()
    bench_end = benchmark.index.max()
    
    start_date = max(price_start, bench_start)
    end_date = min(price_end, bench_end, pd.Timestamp(as_of_date))
    
    # Simulate portfolio NAV
    portfolio_nav = simulate_portfolio_nav(
        portfolio_holdings=portfolio_df,
        prices=prices,
        start_date=start_date.date(),
        end_date=end_date.date(),
        initial_value=100.0,
    )
    
    # Normalize benchmark to start at 100
    benchmark_filtered = benchmark[(benchmark.index >= start_date) & (benchmark.index <= end_date)]
    benchmark_nav = (benchmark_filtered / benchmark_filtered.iloc[0]) * 100
    benchmark_nav.name = "benchmark_nav"
    
    # Align dates
    combined = pd.DataFrame({
        "portfolio_nav": portfolio_nav,
        "benchmark_nav": benchmark_nav,
    }).dropna()
    
    # Compute drawdowns
    portfolio_dd = compute_drawdown(combined["portfolio_nav"])
    benchmark_dd = compute_drawdown(combined["benchmark_nav"])
    
    # Compute rolling returns (approximate with available data)
    combined["portfolio_return_6m"] = compute_rolling_return(combined["portfolio_nav"], 126)
    combined["portfolio_return_12m"] = compute_rolling_return(combined["portfolio_nav"], 252)
    combined["benchmark_return_6m"] = compute_rolling_return(combined["benchmark_nav"], 126)
    combined["benchmark_return_12m"] = compute_rolling_return(combined["benchmark_nav"], 252)
    combined["portfolio_drawdown"] = portfolio_dd
    combined["benchmark_drawdown"] = benchmark_dd
    
    # Excess return
    combined["excess_return"] = (combined["portfolio_nav"] / combined["portfolio_nav"].iloc[0]) - \
                                 (combined["benchmark_nav"] / combined["benchmark_nav"].iloc[0])
    
    # Save time series
    combined.index.name = "date"
    combined.to_csv(config.performance_path)
    print(f"      Saved performance time series to {config.performance_path}")
    
    # Compute summary metrics
    summary = compute_performance_summary(
        portfolio_prices=combined["portfolio_nav"],
        benchmark_prices=combined["benchmark_nav"],
    )
    
    # Add momentum info
    portfolio_symbols = list(portfolio.holdings.keys())
    summary["portfolio_avg_momentum"] = round(momentum[portfolio_symbols].mean(), 4)
    summary["num_positions"] = portfolio.num_positions
    summary["rebalance_date"] = as_of_date.isoformat()
    
    # ========== ROBUSTNESS ANALYSIS ==========
    
    # Load previous portfolio for turnover calculation
    prev_portfolio_path = config.artifacts_path / "portfolio_prev.csv"
    if config.portfolio_path.exists():
        # Move current to prev before saving new
        try:
            prev_df = pd.read_csv(config.portfolio_path)
            prev_holdings = dict(zip(prev_df["symbol"], prev_df["weight"]))
        except Exception:
            prev_holdings = {}
    else:
        prev_holdings = {}
    
    # Compute turnover
    turnover = compute_turnover(prev_holdings, portfolio.holdings)
    churn = compute_churn_stats(prev_holdings, portfolio.holdings)
    
    summary["turnover"] = round(turnover, 4)
    summary["positions_added"] = churn["positions_added"]
    summary["positions_removed"] = churn["positions_removed"]
    
    # Compute return contribution (using momentum as proxy for expected return)
    contributions = compute_return_contribution(portfolio.holdings, momentum)
    concentration = compute_concentration_metrics(contributions)
    
    summary["top5_contribution_pct"] = concentration.get("top5_pct_of_total", 0)
    summary["top_contributor"] = concentration.get("top_contributor", "")
    summary["contribution_herfindahl"] = concentration.get("contribution_herfindahl", 0)
    
    # Transaction cost analysis (50 bps round-trip)
    cost_bps = 50
    turnover_list = [turnover] if turnover > 0 else [0.5]  # Assume initial 50% turnover
    adjusted_nav, cost_summary = simulate_with_costs(
        portfolio_nav=combined["portfolio_nav"],
        rebalance_dates=[as_of_date],
        turnover_per_rebalance=turnover_list,
        cost_bps=cost_bps,
    )
    
    summary["cost_bps"] = cost_bps
    summary["return_after_costs"] = cost_summary["adjusted_return"]
    summary["cost_drag_pct"] = cost_summary["return_drag_pct"]
    
    # Robustness summary
    robustness = compute_robustness_summary(
        turnover_history=[turnover],
        holding_period_stats={"mean_holding_periods": 1},  # Single period
        concentration_metrics=concentration,
        cost_summary=cost_summary,
    )
    
    summary["is_robust"] = robustness["is_robust"]
    summary["robustness_flags"] = "|".join(robustness["robustness_flags"]) if robustness["robustness_flags"] else ""
    
    # Save contribution analysis
    contrib_path = config.artifacts_path / "contributions.csv"
    contributions.to_csv(contrib_path, index=False)
    
    # Add adjusted NAV to performance data
    combined["portfolio_nav_after_costs"] = adjusted_nav
    combined.to_csv(config.performance_path)
    
    # Save metrics
    metrics_path = config.artifacts_path / "metrics.csv"
    pd.DataFrame([summary]).to_csv(metrics_path, index=False)
    print(f"      Saved metrics summary to {metrics_path}")
    
    # Print summary
    print()
    print("  Performance Summary:")
    print(f"    Period: {summary['start_date']} to {summary['end_date']}")
    print(f"    Portfolio Return: {summary['portfolio_total_return']*100:.1f}%")
    print(f"    Benchmark Return: {summary['benchmark_total_return']*100:.1f}%")
    print(f"    Excess Return: {summary['excess_return']*100:.1f}%")
    print(f"    Max Drawdown: {summary['portfolio_max_drawdown']*100:.1f}%")
    print(f"    Alpha: {summary['alpha']*100:.1f}%")
    print(f"    Beta: {summary['beta']:.2f}")
    print(f"    Sharpe: {summary['sharpe_ratio']:.2f}")
    
    print()
    print("  Robustness Analysis:")
    print(f"    Turnover: {turnover*100:.1f}%")
    print(f"    Positions Added: {churn['positions_added']}, Removed: {churn['positions_removed']}")
    print(f"    Top 5 Contribution: {concentration.get('top5_pct_of_total', 0)*100:.1f}% of total")
    print(f"    Return After Costs ({cost_bps}bps): {cost_summary['adjusted_return']*100:.1f}%")
    print(f"    Cost Drag: {cost_summary['return_drag_pct']:.1f}%")
    if robustness["robustness_flags"]:
        print(f"    ⚠️  Flags: {', '.join(robustness['robustness_flags'])}")
    else:
        print(f"    ✓ No robustness concerns")


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
    run_monthly_rebalance(
        as_of_date=args.as_of_date,
        max_positions=args.max_positions,
        force_refresh=args.force_refresh,
    )


if __name__ == "__main__":
    main()
