"""
Systematic Equity Engine Dashboard

Read-only analytics dashboard for momentum strategy visualization.
Consumes data from API endpoints - no business logic in UI.

Answers:
1. Is the strategy healthy?
2. How does it perform vs benchmark?
3. Why does it work (or fail)?
4. What risks exist?

Run with: streamlit run app/dashboard.py
"""

import os
import pandas as pd
import requests
import streamlit as st

# ==============================================================================
# CONFIGURATION
# ==============================================================================

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Momentum Strategy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# DATA LOADERS (pure consumers - no computation)
# ==============================================================================

@st.cache_data(ttl=60)
def load_metrics() -> dict:
    """Load strategy metrics from API."""
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=10)
        return r.json().get("metrics", {}) if r.status_code == 200 else {}
    except Exception:
        return {}


@st.cache_data(ttl=60)
def load_performance() -> pd.DataFrame:
    """Load NAV time series from API."""
    try:
        r = requests.get(f"{API_BASE}/performance", timeout=10)
        if r.status_code == 200:
            df = pd.DataFrame(r.json().get("performance", []))
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_portfolio() -> pd.DataFrame:
    """Load current portfolio holdings from API."""
    try:
        r = requests.get(f"{API_BASE}/portfolio", timeout=10)
        return pd.DataFrame(r.json().get("holdings", [])) if r.status_code == 200 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_contributions() -> pd.DataFrame:
    """Load return contribution analysis from API."""
    try:
        r = requests.get(f"{API_BASE}/contributions", timeout=10)
        return pd.DataFrame(r.json().get("contributions", [])) if r.status_code == 200 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_rebalances() -> pd.DataFrame:
    """Load rebalance history from API."""
    try:
        r = requests.get(f"{API_BASE}/rebalances", timeout=10)
        return pd.DataFrame(r.json().get("rebalances", [])) if r.status_code == 200 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_holding_periods() -> pd.DataFrame:
    """Load holding period analysis from API."""
    try:
        r = requests.get(f"{API_BASE}/holding_periods", timeout=10)
        return pd.DataFrame(r.json().get("holding_periods", [])) if r.status_code == 200 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================

st.sidebar.title("📊 Strategy Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Performance", "Robustness & Risk", "Portfolio", "Rebalance History"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Momentum Strategy | Read-only")

# Load all data once
metrics = load_metrics()
performance_df = load_performance()
portfolio_df = load_portfolio()
contributions_df = load_contributions()
rebalances_df = load_rebalances()
holding_periods_df = load_holding_periods()

# ==============================================================================
# PAGE 1: OVERVIEW
# Purpose: Quick health check - is strategy working?
# ==============================================================================

if page == "Overview":
    st.title("Strategy Overview")
    st.markdown("*Quick health check: Is the strategy working?*")
    
    if not metrics:
        st.error("No metrics available. Run backtest first.")
        st.stop()
    
    # --- Strategy Metadata ---
    st.markdown("### Strategy Info")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Strategy", "Momentum (12-1)")
    col2.metric("Universe", "Nifty 500")
    col3.metric("Rebalance", "Monthly")
    col4.metric("Positions", metrics.get("unique_symbols_held", 30))
    
    st.markdown("---")
    
    # --- Key Performance Metrics ---
    st.markdown("### Key Metrics")
    
    # Use gross or net metrics based on what's available
    gross_return = metrics.get("gross_total_return", metrics.get("portfolio_total_return", 0))
    net_return = metrics.get("net_total_return", gross_return)
    bench_return = metrics.get("benchmark_total_return", 0)
    sharpe = metrics.get("gross_sharpe", metrics.get("sharpe_ratio", 0))
    max_dd = metrics.get("gross_max_drawdown", metrics.get("portfolio_max_drawdown", 0))
    avg_turnover = metrics.get("avg_turnover", metrics.get("turnover", 0))
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Total Return (Gross)", 
        f"{gross_return*100:.1f}%",
        delta=f"+{(gross_return - bench_return)*100:.1f}% vs bench"
    )
    col2.metric(
        "Total Return (Net)",
        f"{net_return*100:.1f}%",
        delta=f"+{(net_return - bench_return)*100:.1f}% vs bench"
    )
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col4.metric("Max Drawdown", f"{max_dd*100:.1f}%")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Benchmark Return", f"{bench_return*100:.1f}%")
    col2.metric("Alpha (Ann.)", f"{metrics.get('alpha', 0)*100:.1f}%")
    col3.metric("Beta", f"{metrics.get('beta', 1):.2f}")
    col4.metric("Avg Turnover", f"{avg_turnover*100:.1f}%")
    
    st.markdown("---")
    
    # --- Robustness Status ---
    st.markdown("### Robustness Status")
    
    is_robust = metrics.get("is_robust", True)
    flags = metrics.get("robustness_flags", "")
    
    if is_robust or not flags:
        st.success("✅ **All Clear** - No robustness concerns detected")
    else:
        st.warning(f"⚠️ **Flags Detected:**")
        for flag in flags.split("|"):
            if flag:
                st.markdown(f"- {flag}")
    
    st.markdown("---")
    
    # --- Mini NAV Chart ---
    st.markdown("### NAV vs Benchmark")
    
    if not performance_df.empty:
        # Detect column names (backtest vs single run)
        nav_col = "portfolio_nav_gross" if "portfolio_nav_gross" in performance_df.columns else "portfolio_nav"
        bench_col = "benchmark_nav"
        
        if nav_col in performance_df.columns and bench_col in performance_df.columns:
            chart_data = performance_df[[nav_col, bench_col]].copy()
            chart_data.columns = ["Portfolio", "Benchmark"]
            st.line_chart(chart_data, use_container_width=True, height=300)
        else:
            st.info("NAV data not available")
    else:
        st.info("Performance data not available")
    
    # --- Period Info ---
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Start Date", metrics.get("start_date", "N/A"))
    col2.metric("End Date", metrics.get("end_date", "N/A"))
    col3.metric("Rebalances", metrics.get("num_rebalances", 1))


# ==============================================================================
# PAGE 2: PERFORMANCE
# Purpose: Detailed performance analysis vs benchmark
# ==============================================================================

elif page == "Performance":
    st.title("Performance Analysis")
    st.markdown("*How does the strategy perform vs benchmark?*")
    
    if performance_df.empty:
        st.error("No performance data available. Run backtest first.")
        st.stop()
    
    # Detect column names
    nav_gross = "portfolio_nav_gross" if "portfolio_nav_gross" in performance_df.columns else "portfolio_nav"
    nav_net = "portfolio_nav_net" if "portfolio_nav_net" in performance_df.columns else nav_gross
    bench_col = "benchmark_nav"
    
    # --- Cumulative NAV ---
    st.markdown("### Cumulative NAV")
    
    if nav_gross in performance_df.columns and bench_col in performance_df.columns:
        nav_data = pd.DataFrame()
        nav_data["Portfolio (Gross)"] = performance_df[nav_gross]
        if nav_net != nav_gross and nav_net in performance_df.columns:
            nav_data["Portfolio (Net)"] = performance_df[nav_net]
        nav_data["Benchmark"] = performance_df[bench_col]
        st.line_chart(nav_data, use_container_width=True)
    
    st.markdown("---")
    
    # --- Excess Return ---
    st.markdown("### Cumulative Excess Return")
    
    if nav_gross in performance_df.columns and bench_col in performance_df.columns:
        # Calculate excess return
        port_ret = performance_df[nav_gross] / performance_df[nav_gross].iloc[0] - 1
        bench_ret = performance_df[bench_col] / performance_df[bench_col].iloc[0] - 1
        excess = (port_ret - bench_ret) * 100
        
        excess_df = pd.DataFrame({"Excess Return (%)": excess})
        st.area_chart(excess_df, use_container_width=True)
    
    st.markdown("---")
    
    # --- Drawdown Chart ---
    st.markdown("### Drawdown")
    
    # Calculate drawdowns from NAV
    if nav_gross in performance_df.columns:
        nav = performance_df[nav_gross]
        rolling_max = nav.expanding().max()
        drawdown = (nav / rolling_max - 1) * 100
        
        if bench_col in performance_df.columns:
            bench = performance_df[bench_col]
            bench_max = bench.expanding().max()
            bench_dd = (bench / bench_max - 1) * 100
            dd_df = pd.DataFrame({
                "Portfolio Drawdown (%)": drawdown,
                "Benchmark Drawdown (%)": bench_dd
            })
        else:
            dd_df = pd.DataFrame({"Portfolio Drawdown (%)": drawdown})
        
        st.area_chart(dd_df, use_container_width=True)
        
        # Drawdown stats
        col1, col2 = st.columns(2)
        col1.metric("Max Portfolio Drawdown", f"{drawdown.min():.1f}%")
        if bench_col in performance_df.columns:
            col2.metric("Max Benchmark Drawdown", f"{bench_dd.min():.1f}%")
    
    st.markdown("---")
    
    # --- Monthly Returns Table ---
    st.markdown("### Performance Summary")
    
    if metrics:
        perf_data = {
            "Metric": ["Total Return", "Annualized Return", "Sharpe Ratio", "Max Drawdown", "Alpha", "Beta"],
            "Portfolio (Gross)": [
                f"{metrics.get('gross_total_return', 0)*100:.1f}%",
                f"{metrics.get('gross_ann_return', 0)*100:.1f}%",
                f"{metrics.get('gross_sharpe', 0):.2f}",
                f"{metrics.get('gross_max_drawdown', 0)*100:.1f}%",
                f"{metrics.get('alpha', 0)*100:.1f}%",
                f"{metrics.get('beta', 1):.2f}",
            ],
            "Portfolio (Net)": [
                f"{metrics.get('net_total_return', metrics.get('gross_total_return', 0))*100:.1f}%",
                f"{metrics.get('net_ann_return', metrics.get('gross_ann_return', 0))*100:.1f}%",
                f"{metrics.get('net_sharpe', metrics.get('gross_sharpe', 0)):.2f}",
                f"{metrics.get('net_max_drawdown', metrics.get('gross_max_drawdown', 0))*100:.1f}%",
                "-",
                "-",
            ],
            "Benchmark": [
                f"{metrics.get('benchmark_total_return', 0)*100:.1f}%",
                f"{metrics.get('benchmark_ann_return', 0)*100:.1f}%",
                "-",
                "-",
                "-",
                "-",
            ],
        }
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 3: ROBUSTNESS & RISK
# Purpose: Understand risks and strategy stability
# ==============================================================================

elif page == "Robustness & Risk":
    st.title("Robustness & Risk Analysis")
    st.markdown("*What risks exist? Is this luck or skill?*")
    
    # --- Turnover Analysis ---
    st.markdown("### Turnover Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Turnover", f"{metrics.get('avg_turnover', 0)*100:.1f}%")
    col2.metric("Max Turnover", f"{metrics.get('max_turnover', 0)*100:.1f}%")
    col3.metric("Total Turnover", f"{metrics.get('total_turnover', 0)*100:.1f}%")
    col4.metric("Cost Drag", f"{metrics.get('cost_drag_pct', 0):.1f}%")
    
    # Turnover over time chart
    if not rebalances_df.empty and "turnover" in rebalances_df.columns:
        st.markdown("#### Turnover by Rebalance")
        if "date" in rebalances_df.columns:
            turn_df = rebalances_df[["date", "turnover"]].copy()
            turn_df["date"] = pd.to_datetime(turn_df["date"])
            turn_df = turn_df.set_index("date")
            turn_df["turnover"] = turn_df["turnover"] * 100
            turn_df.columns = ["Turnover (%)"]
            st.bar_chart(turn_df, use_container_width=True)
    
    st.markdown("---")
    
    # --- Transaction Cost Impact ---
    st.markdown("### Transaction Cost Impact")
    
    gross_ret = metrics.get("gross_total_return", 0)
    net_ret = metrics.get("net_total_return", gross_ret)
    cost_drag = gross_ret - net_ret
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Gross Return", f"{gross_ret*100:.1f}%")
    col2.metric("Net Return (50bps)", f"{net_ret*100:.1f}%")
    col3.metric("Cost Impact", f"-{cost_drag*100:.1f}%")
    
    # NAV comparison chart (gross vs net)
    if not performance_df.empty:
        nav_gross = "portfolio_nav_gross" if "portfolio_nav_gross" in performance_df.columns else "portfolio_nav"
        nav_net = "portfolio_nav_net" if "portfolio_nav_net" in performance_df.columns else None
        
        if nav_net and nav_net in performance_df.columns:
            cost_df = performance_df[[nav_gross, nav_net]].copy()
            cost_df.columns = ["Gross NAV", "Net NAV"]
            st.line_chart(cost_df, use_container_width=True, height=250)
    
    st.markdown("---")
    
    # --- Holding Period Analysis ---
    st.markdown("### Holding Period Analysis")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Holding", f"{metrics.get('mean_holding_periods', 0):.1f} periods")
    col2.metric("Median Holding", f"{metrics.get('median_holding_periods', 0):.1f} periods")
    col3.metric("Unique Symbols", metrics.get("unique_symbols_held", 0))
    
    if not holding_periods_df.empty and "periods_held" in holding_periods_df.columns:
        st.markdown("#### Holding Period Distribution")
        
        # Create histogram data
        periods = holding_periods_df["periods_held"]
        hist_data = periods.value_counts().sort_index()
        hist_df = pd.DataFrame({"Count": hist_data})
        hist_df.index.name = "Periods Held"
        st.bar_chart(hist_df, use_container_width=True, height=200)
        
        # Top persistent holdings
        st.markdown("#### Most Persistent Holdings")
        top_holdings = holding_periods_df.nlargest(10, "periods_held")[["symbol", "periods_held", "still_held"]]
        top_holdings.columns = ["Symbol", "Periods Held", "Still Held"]
        st.dataframe(top_holdings, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # --- Concentration Analysis ---
    st.markdown("### Return Concentration")
    
    col1, col2 = st.columns(2)
    col1.metric("Top 5 Contribution", f"{metrics.get('avg_top5_concentration', 0)*100:.1f}%")
    col2.metric("Herfindahl Index", f"{metrics.get('contribution_herfindahl', 0):.4f}")
    
    if not contributions_df.empty:
        st.markdown("#### Top Contributors")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Best Performers**")
            top5 = contributions_df.head(5)[["symbol", "contribution"]].copy()
            top5["contribution"] = (top5["contribution"] * 100).round(2)
            top5.columns = ["Symbol", "Contribution (%)"]
            st.dataframe(top5, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Worst Performers**")
            bottom5 = contributions_df.tail(5)[["symbol", "contribution"]].copy()
            bottom5["contribution"] = (bottom5["contribution"] * 100).round(2)
            bottom5.columns = ["Symbol", "Contribution (%)"]
            st.dataframe(bottom5, use_container_width=True, hide_index=True)
        
        # Contribution bar chart
        st.markdown("#### All Contributions")
        contrib_chart = contributions_df[["symbol", "contribution"]].copy()
        contrib_chart["contribution"] = contrib_chart["contribution"] * 100
        contrib_chart = contrib_chart.set_index("symbol")
        contrib_chart.columns = ["Contribution (%)"]
        st.bar_chart(contrib_chart, use_container_width=True, height=300)
    
    st.markdown("---")
    
    # --- Robustness Flags ---
    st.markdown("### Robustness Assessment")
    
    is_robust = metrics.get("is_robust", True)
    flags = metrics.get("robustness_flags", "")
    
    if is_robust or not flags:
        st.success("✅ **Strategy passes all robustness checks**")
        st.markdown("""
        - Average turnover is reasonable
        - Returns not overly concentrated in few positions
        - Strategy survives transaction costs
        - Holding periods indicate signal persistence
        """)
    else:
        st.error("⚠️ **Robustness concerns detected:**")
        for flag in flags.split("|"):
            if flag:
                st.markdown(f"- {flag}")


# ==============================================================================
# PAGE 4: PORTFOLIO
# Purpose: Current holdings and latest changes
# ==============================================================================

elif page == "Portfolio":
    st.title("Current Portfolio")
    st.markdown("*What do we hold and why?*")
    
    if portfolio_df.empty:
        st.warning("No portfolio data available.")
        st.stop()
    
    # --- Summary Stats ---
    st.markdown("### Portfolio Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Holdings", len(portfolio_df))
    
    if "weight" in portfolio_df.columns:
        col2.metric("Max Weight", f"{portfolio_df['weight'].max()*100:.1f}%")
        col3.metric("Min Weight", f"{portfolio_df['weight'].min()*100:.1f}%")
    
    if "as_of_date" in portfolio_df.columns:
        col4.metric("As of", portfolio_df["as_of_date"].iloc[0])
    
    st.markdown("---")
    
    # --- Holdings Table ---
    st.markdown("### Current Holdings")
    
    display_df = portfolio_df.copy()
    if "weight" in display_df.columns:
        display_df["weight"] = (display_df["weight"] * 100).round(2)
        display_df = display_df.rename(columns={"weight": "Weight (%)"})
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # --- Latest Rebalance Changes ---
    st.markdown("### Latest Rebalance Changes")
    
    if not rebalances_df.empty:
        latest = rebalances_df.iloc[-1] if len(rebalances_df) > 0 else None
        
        if latest is not None:
            col1, col2, col3 = st.columns(3)
            col1.metric("Date", latest.get("date", "N/A"))
            col2.metric("Positions Added", int(latest.get("positions_added", 0)))
            col3.metric("Positions Removed", int(latest.get("positions_removed", 0)))
            
            # Show added/removed symbols if available
            added = latest.get("symbols_added", "[]")
            removed = latest.get("symbols_removed", "[]")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Added:**")
                if added and added != "[]":
                    try:
                        symbols = eval(added) if isinstance(added, str) else added
                        for s in symbols[:10]:
                            st.markdown(f"+ {s}")
                        if len(symbols) > 10:
                            st.markdown(f"*...and {len(symbols)-10} more*")
                    except Exception:
                        st.text(str(added)[:200])
                else:
                    st.text("None")
            
            with col2:
                st.markdown("**Removed:**")
                if removed and removed != "[]":
                    try:
                        symbols = eval(removed) if isinstance(removed, str) else removed
                        for s in symbols[:10]:
                            st.markdown(f"- {s}")
                        if len(symbols) > 10:
                            st.markdown(f"*...and {len(symbols)-10} more*")
                    except Exception:
                        st.text(str(removed)[:200])
                else:
                    st.text("None")
    else:
        st.info("No rebalance history available.")


# ==============================================================================
# PAGE 5: REBALANCE HISTORY
# Purpose: Track all rebalances over time
# ==============================================================================

elif page == "Rebalance History":
    st.title("Rebalance History")
    st.markdown("*Timeline of all portfolio changes*")
    
    if rebalances_df.empty:
        st.warning("No rebalance history available. Run backtest first.")
        st.stop()
    
    # --- Summary ---
    st.markdown("### Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rebalances", len(rebalances_df))
    
    if "turnover" in rebalances_df.columns:
        col2.metric("Avg Turnover", f"{rebalances_df['turnover'].mean()*100:.1f}%")
        col3.metric("Max Turnover", f"{rebalances_df['turnover'].max()*100:.1f}%")
    
    if "positions_added" in rebalances_df.columns:
        col4.metric("Avg Positions Changed", f"{(rebalances_df['positions_added'] + rebalances_df['positions_removed']).mean():.1f}")
    
    st.markdown("---")
    
    # --- Timeline Table ---
    st.markdown("### Rebalance Timeline")
    
    display_df = rebalances_df.copy()
    
    # Format columns for display
    if "turnover" in display_df.columns:
        display_df["turnover"] = (display_df["turnover"] * 100).round(1)
    
    if "avg_momentum" in display_df.columns:
        display_df["avg_momentum"] = display_df["avg_momentum"].round(2)
    
    if "top5_contribution_pct" in display_df.columns:
        display_df["top5_contribution_pct"] = (display_df["top5_contribution_pct"] * 100).round(1)
    
    # Select columns to display
    display_cols = ["date", "num_positions", "turnover", "positions_added", "positions_removed"]
    if "avg_momentum" in display_df.columns:
        display_cols.append("avg_momentum")
    
    display_df = display_df[[c for c in display_cols if c in display_df.columns]]
    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # --- Turnover Chart ---
    if "turnover" in rebalances_df.columns and "date" in rebalances_df.columns:
        st.markdown("### Turnover Over Time")
        
        turn_df = rebalances_df[["date", "turnover"]].copy()
        turn_df["date"] = pd.to_datetime(turn_df["date"])
        turn_df = turn_df.set_index("date")
        turn_df["turnover"] = turn_df["turnover"] * 100
        turn_df.columns = ["Turnover (%)"]
        st.line_chart(turn_df, use_container_width=True)
    
    # --- Churn Chart ---
    if "positions_added" in rebalances_df.columns and "date" in rebalances_df.columns:
        st.markdown("### Position Changes Over Time")
        
        churn_df = rebalances_df[["date", "positions_added", "positions_removed"]].copy()
        churn_df["date"] = pd.to_datetime(churn_df["date"])
        churn_df = churn_df.set_index("date")
        churn_df.columns = ["Added", "Removed"]
        st.bar_chart(churn_df, use_container_width=True)


# ==============================================================================
# FOOTER
# ==============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Status")
st.sidebar.markdown(f"- Metrics: {'✅' if metrics else '❌'}")
st.sidebar.markdown(f"- Performance: {'✅' if not performance_df.empty else '❌'}")
st.sidebar.markdown(f"- Portfolio: {'✅' if not portfolio_df.empty else '❌'}")
st.sidebar.markdown(f"- Rebalances: {'✅' if not rebalances_df.empty else '❌'}")
st.sidebar.markdown(f"- Holdings: {'✅' if not holding_periods_df.empty else '❌'}")
