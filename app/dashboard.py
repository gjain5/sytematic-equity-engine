"""
Streamlit dashboard for portfolio visualization.

Design decisions:
- Read-only dashboard, no write actions
- Reads directly from CSV files (no API dependency for simplicity)
- Clean, minimal UI focused on essentials

Run with: streamlit run app/dashboard.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Page configuration
st.set_page_config(
    page_title="Systematic Equity Portfolio",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Systematic Equity Portfolio")
st.markdown("---")


def load_portfolio() -> pd.DataFrame:
    """Load portfolio from CSV file."""
    portfolio_path = ARTIFACTS_DIR / "portfolio.csv"
    if not portfolio_path.exists():
        return pd.DataFrame()
    return pd.read_csv(portfolio_path)


def load_performance() -> pd.DataFrame:
    """Load performance from CSV file."""
    performance_path = ARTIFACTS_DIR / "performance.csv"
    if not performance_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(performance_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


# Portfolio Holdings Section
st.header("Portfolio Holdings")

portfolio_df = load_portfolio()

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

# Performance Section
st.header("Performance vs Benchmark")

performance_df = load_performance()

if performance_df.empty:
    st.warning("No performance data available. Run the strategy to generate performance history.")
else:
    # Performance chart
    if "date" in performance_df.columns:
        chart_data = performance_df.set_index("date")
        
        # Filter to only numeric columns for charting
        numeric_cols = chart_data.select_dtypes(include=["float64", "int64"]).columns.tolist()
        
        if numeric_cols:
            st.line_chart(chart_data[numeric_cols])
        else:
            st.info("No numeric columns found for charting.")
    
    # Performance table
    with st.expander("View Performance Data"):
        st.dataframe(
            performance_df,
            use_container_width=True,
            hide_index=True,
        )

# Footer
st.markdown("---")
st.caption("Systematic Equity Engine | Read-only Dashboard")
