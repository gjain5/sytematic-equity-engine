"""
Streamlit dashboard for portfolio visualization.

Design decisions:
- Read-only dashboard, no write actions
- Fetches data from FastAPI backend (internal URL)
- Clean, minimal UI focused on essentials
- Answers: "Is this strategy beating the index, and when does it fail?"

Run with: streamlit run app/dashboard.py
"""

import os

import pandas as pd
import requests
import streamlit as st

# API base URL - configurable via environment variable
# Default to internal localhost for OCI deployment (Nginx proxies externally)
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")

# Page configuration
st.set_page_config(
    page_title="Systematic Equity Portfolio",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Systematic Equity Portfolio")
st.markdown("---")


def load_portfolio() -> pd.DataFrame:
    """Load portfolio from API."""
    try:
        response = requests.get(f"{API_BASE}/portfolio", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data.get("holdings", []))
        return pd.DataFrame()
    except requests.exceptions.RequestException:
        return pd.DataFrame()


def load_performance() -> pd.DataFrame:
    """Load performance from API."""
    try:
        response = requests.get(f"{API_BASE}/performance", timeout=5)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data.get("performance", []))
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            return df
        return pd.DataFrame()
    except requests.exceptions.RequestException:
        return pd.DataFrame()


def load_metrics() -> dict:
    """Load metrics summary from API."""
    try:
        response = requests.get(f"{API_BASE}/metrics", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("metrics", {})
        return {}
    except requests.exceptions.RequestException:
        return {}


# Load all data
portfolio_df = load_portfolio()
performance_df = load_performance()
metrics = load_metrics()

# Strategy Overview Section
st.header("Strategy Overview")

if metrics:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Portfolio Return",
            f"{metrics.get('portfolio_total_return', 0)*100:.1f}%",
            delta=f"{metrics.get('excess_return', 0)*100:.1f}% vs benchmark"
        )
    with col2:
        st.metric(
            "Benchmark Return",
            f"{metrics.get('benchmark_total_return', 0)*100:.1f}%"
        )
    with col3:
        st.metric(
            "Max Drawdown",
            f"{metrics.get('portfolio_max_drawdown', 0)*100:.1f}%"
        )
    with col4:
        st.metric(
            "Sharpe Ratio",
            f"{metrics.get('sharpe_ratio', 0):.2f}"
        )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Alpha (Ann.)", f"{metrics.get('alpha', 0)*100:.1f}%")
    with col2:
        st.metric("Beta", f"{metrics.get('beta', 1):.2f}")
    with col3:
        st.metric("Positions", metrics.get('num_positions', 0))
    with col4:
        st.metric("Last Rebalance", metrics.get('rebalance_date', 'N/A'))
else:
    st.info("Run the strategy to generate performance metrics.")

st.markdown("---")

# Performance vs Benchmark Chart
st.header("Portfolio vs Benchmark")

if not performance_df.empty and "portfolio_nav" in performance_df.columns:
    # NAV comparison chart
    nav_data = performance_df[["portfolio_nav", "benchmark_nav"]].copy()
    nav_data.columns = ["Portfolio", "Benchmark (Nifty 50)"]
    st.line_chart(nav_data, use_container_width=True)
    
    # Excess return chart
    if "excess_return" in performance_df.columns:
        st.subheader("Cumulative Excess Return")
        excess_data = performance_df[["excess_return"]].copy()
        excess_data.columns = ["Excess Return vs Benchmark"]
        st.area_chart(excess_data, use_container_width=True)
else:
    st.warning("No performance data available. Run the strategy to generate performance history.")

st.markdown("---")

# Drawdown Analysis
st.header("Drawdown Analysis")

if not performance_df.empty and "portfolio_drawdown" in performance_df.columns:
    dd_data = performance_df[["portfolio_drawdown", "benchmark_drawdown"]].copy()
    dd_data.columns = ["Portfolio Drawdown", "Benchmark Drawdown"]
    # Convert to percentage for display
    dd_data = dd_data * 100
    st.area_chart(dd_data, use_container_width=True)
    
    # Drawdown stats
    col1, col2 = st.columns(2)
    with col1:
        max_dd = performance_df["portfolio_drawdown"].min()
        max_dd_date = performance_df["portfolio_drawdown"].idxmin()
        st.metric("Worst Portfolio Drawdown", f"{max_dd*100:.1f}%", delta=f"on {max_dd_date.strftime('%Y-%m-%d')}")
    with col2:
        bench_max_dd = performance_df["benchmark_drawdown"].min()
        bench_max_dd_date = performance_df["benchmark_drawdown"].idxmin()
        st.metric("Worst Benchmark Drawdown", f"{bench_max_dd*100:.1f}%", delta=f"on {bench_max_dd_date.strftime('%Y-%m-%d')}")
else:
    st.info("Drawdown data not available.")

st.markdown("---")

# Rolling Returns (when available)
if not performance_df.empty and "portfolio_return_6m" in performance_df.columns:
    st.header("Rolling Returns")
    
    # Filter out NaN values for rolling returns
    roll_data = performance_df[["portfolio_return_6m", "benchmark_return_6m"]].dropna()
    if not roll_data.empty:
        roll_data.columns = ["Portfolio 6M Return", "Benchmark 6M Return"]
        roll_data = roll_data * 100  # Convert to percentage
        st.line_chart(roll_data, use_container_width=True)
    
    st.markdown("---")

# Portfolio Holdings Section
st.header("Current Portfolio Holdings")

if portfolio_df.empty:
    st.warning("No portfolio data available. Run the strategy to generate holdings.")
else:
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Holdings", len(portfolio_df))
    with col2:
        if "weight" in portfolio_df.columns:
            top_weight = portfolio_df["weight"].max() * 100
            st.metric("Max Position Weight", f"{top_weight:.1f}%")
    with col3:
        if "as_of_date" in portfolio_df.columns:
            st.metric("As of Date", portfolio_df["as_of_date"].iloc[0])
    
    # Holdings table
    st.dataframe(
        portfolio_df,
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")

# Raw Data Expander
with st.expander("View Raw Performance Data"):
    if not performance_df.empty:
        st.dataframe(
            performance_df.reset_index(),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No data available.")

# Footer
st.markdown("---")
st.caption("Systematic Equity Engine | Momentum Strategy | Read-only Dashboard")
