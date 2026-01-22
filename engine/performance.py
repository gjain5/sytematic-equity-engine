"""
Performance measurement and benchmarking module.

Design decisions:
- All computations are deterministic and date-driven
- Benchmark data is fetched/cached separately from portfolio logic
- Metrics are computed as pure functions with explicit inputs
"""

from datetime import date, timedelta
from typing import Optional, Tuple
import pandas as pd
import numpy as np


def fetch_benchmark_prices(
    start_date: date,
    end_date: date,
    symbol: str = "^NSEI",
) -> pd.Series:
    """
    Fetch benchmark index prices from Yahoo Finance.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        symbol: Yahoo Finance symbol (default: ^NSEI for Nifty 50)
                Use ^CRSLDX for Nifty 500 if available
    
    Returns:
        Series with date index and adjusted close prices
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")
    
    data = yf.download(
        symbol,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=True,
    )
    
    if data.empty:
        raise ValueError(f"No data found for benchmark {symbol}")
    
    prices = data["Close"]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    prices.name = "benchmark"
    
    return prices


def load_benchmark_from_csv(filepath: str) -> pd.Series:
    """Load benchmark prices from CSV file."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    if "benchmark" in df.columns:
        return df["benchmark"]
    return df.iloc[:, 0]


def save_benchmark_to_csv(prices: pd.Series, filepath: str) -> None:
    """Save benchmark prices to CSV file."""
    df = pd.DataFrame({"benchmark": prices})
    df.to_csv(filepath)


def compute_returns(prices: pd.Series) -> pd.Series:
    """Compute daily returns from prices."""
    return prices.pct_change().dropna()


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Compute cumulative returns from daily returns."""
    return (1 + returns).cumprod() - 1


def compute_total_return(prices: pd.Series) -> float:
    """Compute total return from first to last price."""
    if len(prices) < 2:
        return 0.0
    return (prices.iloc[-1] / prices.iloc[0]) - 1


def compute_annualized_return(total_return: float, days: int) -> float:
    """Annualize a total return given number of calendar days."""
    if days <= 0:
        return 0.0
    years = days / 365.25
    if years < 0.1:  # Less than ~36 days
        return total_return
    return (1 + total_return) ** (1 / years) - 1


def compute_rolling_return(
    prices: pd.Series,
    window_days: int,
) -> pd.Series:
    """
    Compute rolling returns over a window.
    
    Args:
        prices: Price series with datetime index
        window_days: Number of trading days for the window
    
    Returns:
        Series of rolling returns
    """
    return prices.pct_change(periods=window_days)


def compute_drawdown(prices: pd.Series) -> pd.Series:
    """
    Compute drawdown series from prices.
    
    Drawdown = (current price - running max) / running max
    
    Returns:
        Series of drawdowns (negative values)
    """
    running_max = prices.expanding().max()
    drawdown = (prices - running_max) / running_max
    return drawdown


def compute_max_drawdown(prices: pd.Series) -> float:
    """Compute maximum drawdown from prices."""
    drawdown = compute_drawdown(prices)
    return drawdown.min()


def compute_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """
    Compute volatility (standard deviation of returns).
    
    Args:
        returns: Daily returns series
        annualize: Whether to annualize (multiply by sqrt(252))
    
    Returns:
        Volatility as decimal
    """
    vol = returns.std()
    if annualize:
        vol *= np.sqrt(252)
    return vol


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
) -> float:
    """
    Compute Sharpe ratio.
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate (default 5%)
    
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Annualized return
    total_return = (1 + returns).prod() - 1
    days = len(returns)
    ann_return = compute_annualized_return(total_return, days)
    
    # Annualized volatility
    ann_vol = compute_volatility(returns, annualize=True)
    
    if ann_vol == 0:
        return 0.0
    
    return (ann_return - risk_free_rate) / ann_vol


def compute_alpha_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Tuple[float, float]:
    """
    Compute alpha and beta vs benchmark using linear regression.
    
    Args:
        portfolio_returns: Daily portfolio returns
        benchmark_returns: Daily benchmark returns
    
    Returns:
        Tuple of (annualized alpha, beta)
    """
    # Align the series
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 10:
        return 0.0, 1.0
    
    port_ret = aligned.iloc[:, 0].values
    bench_ret = aligned.iloc[:, 1].values
    
    # Simple linear regression: portfolio = alpha + beta * benchmark
    # Using numpy for efficiency
    cov_matrix = np.cov(port_ret, bench_ret)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 1.0
    
    # Alpha (daily)
    alpha_daily = port_ret.mean() - beta * bench_ret.mean()
    
    # Annualize alpha
    alpha_annual = alpha_daily * 252
    
    return alpha_annual, beta


def compute_rolling_alpha(
    portfolio_prices: pd.Series,
    benchmark_prices: pd.Series,
    window_days: int = 126,  # 6 months
) -> pd.Series:
    """
    Compute rolling alpha vs benchmark.
    
    Args:
        portfolio_prices: Portfolio price series
        benchmark_prices: Benchmark price series
        window_days: Rolling window in trading days
    
    Returns:
        Series of rolling annualized alpha
    """
    port_returns = compute_returns(portfolio_prices)
    bench_returns = compute_returns(benchmark_prices)
    
    # Align
    aligned = pd.concat([port_returns, bench_returns], axis=1).dropna()
    aligned.columns = ["portfolio", "benchmark"]
    
    def calc_alpha(window):
        if len(window) < 20:
            return np.nan
        port = window["portfolio"].values
        bench = window["benchmark"].values
        
        cov = np.cov(port, bench)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
        alpha_daily = port.mean() - beta * bench.mean()
        return alpha_daily * 252
    
    rolling_alpha = aligned.rolling(window=window_days).apply(
        lambda x: calc_alpha(aligned.loc[x.index]), raw=False
    )["portfolio"]
    
    # Simpler approach: use rolling correlation and volatility
    rolling_alpha = pd.Series(index=aligned.index, dtype=float)
    
    for i in range(window_days, len(aligned)):
        window = aligned.iloc[i-window_days:i]
        alpha, _ = compute_alpha_beta(window["portfolio"], window["benchmark"])
        rolling_alpha.iloc[i] = alpha
    
    return rolling_alpha


def compute_performance_summary(
    portfolio_prices: pd.Series,
    benchmark_prices: pd.Series,
    as_of_date: Optional[date] = None,
) -> dict:
    """
    Compute comprehensive performance summary.
    
    Args:
        portfolio_prices: Portfolio NAV/price series
        benchmark_prices: Benchmark price series
        as_of_date: Optional date to compute metrics as of
    
    Returns:
        Dictionary of performance metrics
    """
    # Filter to as_of_date if provided
    if as_of_date:
        as_of_ts = pd.Timestamp(as_of_date)
        portfolio_prices = portfolio_prices[portfolio_prices.index <= as_of_ts]
        benchmark_prices = benchmark_prices[benchmark_prices.index <= as_of_ts]
    
    # Align series
    start_date = max(portfolio_prices.index.min(), benchmark_prices.index.min())
    end_date = min(portfolio_prices.index.max(), benchmark_prices.index.max())
    
    portfolio_prices = portfolio_prices[(portfolio_prices.index >= start_date) & 
                                         (portfolio_prices.index <= end_date)]
    benchmark_prices = benchmark_prices[(benchmark_prices.index >= start_date) & 
                                         (benchmark_prices.index <= end_date)]
    
    # Compute returns
    port_returns = compute_returns(portfolio_prices)
    bench_returns = compute_returns(benchmark_prices)
    
    # Total returns
    port_total = compute_total_return(portfolio_prices)
    bench_total = compute_total_return(benchmark_prices)
    
    # Trading days
    trading_days = len(port_returns)
    calendar_days = (end_date - start_date).days
    
    # Annualized returns
    port_ann = compute_annualized_return(port_total, calendar_days)
    bench_ann = compute_annualized_return(bench_total, calendar_days)
    
    # Volatility
    port_vol = compute_volatility(port_returns)
    bench_vol = compute_volatility(bench_returns)
    
    # Drawdowns
    port_max_dd = compute_max_drawdown(portfolio_prices)
    bench_max_dd = compute_max_drawdown(benchmark_prices)
    
    # Alpha/Beta
    alpha, beta = compute_alpha_beta(port_returns, bench_returns)
    
    # Sharpe
    port_sharpe = compute_sharpe_ratio(port_returns)
    
    return {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "trading_days": trading_days,
        "portfolio_total_return": round(port_total, 4),
        "benchmark_total_return": round(bench_total, 4),
        "portfolio_ann_return": round(port_ann, 4),
        "benchmark_ann_return": round(bench_ann, 4),
        "portfolio_volatility": round(port_vol, 4),
        "benchmark_volatility": round(bench_vol, 4),
        "portfolio_max_drawdown": round(port_max_dd, 4),
        "benchmark_max_drawdown": round(bench_max_dd, 4),
        "alpha": round(alpha, 4),
        "beta": round(beta, 4),
        "sharpe_ratio": round(port_sharpe, 4),
        "excess_return": round(port_total - bench_total, 4),
    }


def simulate_portfolio_nav(
    portfolio_holdings: pd.DataFrame,
    prices: pd.DataFrame,
    start_date: date,
    end_date: date,
    initial_value: float = 100.0,
) -> pd.Series:
    """
    Simulate portfolio NAV over time given holdings and prices.
    
    Args:
        portfolio_holdings: DataFrame with columns [symbol, weight, as_of_date]
        prices: DataFrame with date index and symbol columns
        start_date: Start date for simulation
        end_date: End date for simulation
        initial_value: Starting portfolio value
    
    Returns:
        Series of portfolio NAV with date index
    """
    # Get holdings as dict
    holdings = dict(zip(portfolio_holdings["symbol"], portfolio_holdings["weight"]))
    
    # Filter prices to date range
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    mask = (prices.index >= pd.Timestamp(start_date)) & (prices.index <= pd.Timestamp(end_date))
    prices = prices[mask]
    
    if prices.empty:
        return pd.Series(dtype=float)
    
    # Filter to symbols we hold
    available = [s for s in holdings.keys() if s in prices.columns]
    if not available:
        return pd.Series(dtype=float)
    
    # Compute weighted returns
    weights = pd.Series({s: holdings[s] for s in available})
    weights = weights / weights.sum()  # Renormalize
    
    symbol_returns = prices[available].pct_change()
    portfolio_returns = (symbol_returns * weights).sum(axis=1)
    
    # Compute NAV
    nav = (1 + portfolio_returns).cumprod() * initial_value
    nav.iloc[0] = initial_value
    nav.name = "portfolio_nav"
    
    return nav
