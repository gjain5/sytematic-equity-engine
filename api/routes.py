"""
API route definitions.

Design decisions:
- All routes are read-only (GET only)
- Data is read from flat files in artifacts/
- Paths are resolved relative to project root
"""

from pathlib import Path
from typing import Dict, List, Any

from fastapi import APIRouter, HTTPException
import pandas as pd

router = APIRouter()

# Resolve paths relative to this file's location
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns simple status for monitoring/load balancer health checks.
    """
    return {"status": "healthy"}


@router.get("/portfolio")
async def get_portfolio() -> Dict[str, Any]:
    """
    Get current portfolio holdings.
    
    Reads from artifacts/portfolio.csv and returns as JSON.
    """
    portfolio_path = ARTIFACTS_DIR / "portfolio.csv"
    
    if not portfolio_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Portfolio file not found. Run the strategy first."
        )
    
    try:
        df = pd.read_csv(portfolio_path)
        return {
            "holdings": df.to_dict(orient="records"),
            "count": len(df),
            "source": "artifacts/portfolio.csv"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading portfolio: {str(e)}"
        )


@router.get("/performance")
async def get_performance() -> Dict[str, Any]:
    """
    Get portfolio performance history.
    
    Reads from artifacts/performance.csv and returns as JSON.
    """
    performance_path = ARTIFACTS_DIR / "performance.csv"
    
    if not performance_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Performance file not found. Run the strategy first."
        )
    
    try:
        df = pd.read_csv(performance_path)
        return {
            "performance": df.to_dict(orient="records"),
            "count": len(df),
            "source": "artifacts/performance.csv"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading performance: {str(e)}"
        )


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get performance metrics summary.
    
    Reads from artifacts/metrics.csv and returns as JSON.
    """
    metrics_path = ARTIFACTS_DIR / "metrics.csv"
    
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics file not found. Run the strategy first."
        )
    
    try:
        df = pd.read_csv(metrics_path)
        if len(df) == 0:
            return {"metrics": {}}
        # Return first row as dict (most recent metrics)
        return {
            "metrics": df.iloc[0].to_dict(),
            "source": "artifacts/metrics.csv"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading metrics: {str(e)}"
        )


@router.get("/contributions")
async def get_contributions() -> Dict[str, Any]:
    """
    Get return contribution analysis by position.
    
    Reads from artifacts/contributions.csv and returns as JSON.
    """
    contrib_path = ARTIFACTS_DIR / "contributions.csv"
    
    if not contrib_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Contributions file not found. Run the strategy first."
        )
    
    try:
        df = pd.read_csv(contrib_path)
        return {
            "contributions": df.to_dict(orient="records"),
            "count": len(df),
            "source": "artifacts/contributions.csv"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading contributions: {str(e)}"
        )


@router.get("/rebalances")
async def get_rebalances() -> Dict[str, Any]:
    """
    Get rebalance history with turnover and churn stats.
    
    Reads from artifacts/rebalances.csv and returns as JSON.
    """
    rebalances_path = ARTIFACTS_DIR / "rebalances.csv"
    
    if not rebalances_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Rebalances file not found. Run backtest first."
        )
    
    try:
        df = pd.read_csv(rebalances_path)
        return {
            "rebalances": df.to_dict(orient="records"),
            "count": len(df),
            "source": "artifacts/rebalances.csv"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading rebalances: {str(e)}"
        )


@router.get("/holding_periods")
async def get_holding_periods() -> Dict[str, Any]:
    """
    Get holding period analysis per symbol.
    
    Reads from artifacts/holding_periods.csv and returns as JSON.
    """
    holdings_path = ARTIFACTS_DIR / "holding_periods.csv"
    
    if not holdings_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Holding periods file not found. Run backtest first."
        )
    
    try:
        df = pd.read_csv(holdings_path)
        return {
            "holding_periods": df.to_dict(orient="records"),
            "count": len(df),
            "source": "artifacts/holding_periods.csv"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading holding periods: {str(e)}"
        )


# ==============================================================================
# VALUE STRATEGY ENDPOINTS
# ==============================================================================

@router.get("/metrics_value")
async def get_metrics_value() -> Dict[str, Any]:
    """Get Value strategy metrics."""
    metrics_path = ARTIFACTS_DIR / "metrics_value.csv"
    if not metrics_path.exists():
        return {"metrics": {}, "error": "Value metrics not found"}
    try:
        df = pd.read_csv(metrics_path)
        return {"metrics": df.iloc[0].to_dict() if len(df) > 0 else {}, "source": "artifacts/metrics_value.csv"}
    except Exception as e:
        return {"metrics": {}, "error": str(e)}


@router.get("/performance_value")
async def get_performance_value() -> Dict[str, Any]:
    """Get Value strategy performance time series."""
    perf_path = ARTIFACTS_DIR / "performance_value.csv"
    if not perf_path.exists():
        return {"performance": [], "error": "Value performance not found"}
    try:
        df = pd.read_csv(perf_path)
        return {"performance": df.to_dict(orient="records"), "count": len(df), "source": "artifacts/performance_value.csv"}
    except Exception as e:
        return {"performance": [], "error": str(e)}


@router.get("/rebalances_value")
async def get_rebalances_value() -> Dict[str, Any]:
    """Get Value strategy rebalance history."""
    rebal_path = ARTIFACTS_DIR / "rebalances_value.csv"
    if not rebal_path.exists():
        return {"rebalances": [], "error": "Value rebalances not found"}
    try:
        df = pd.read_csv(rebal_path)
        return {"rebalances": df.to_dict(orient="records"), "count": len(df), "source": "artifacts/rebalances_value.csv"}
    except Exception as e:
        return {"rebalances": [], "error": str(e)}


@router.get("/holding_periods_value")
async def get_holding_periods_value() -> Dict[str, Any]:
    """Get Value strategy holding periods."""
    hold_path = ARTIFACTS_DIR / "holding_periods_value.csv"
    if not hold_path.exists():
        return {"holding_periods": [], "error": "Value holding periods not found"}
    try:
        df = pd.read_csv(hold_path)
        return {"holding_periods": df.to_dict(orient="records"), "count": len(df), "source": "artifacts/holding_periods_value.csv"}
    except Exception as e:
        return {"holding_periods": [], "error": str(e)}


# ==============================================================================
# BLEND STRATEGY ENDPOINTS
# ==============================================================================

@router.get("/metrics_blend")
async def get_metrics_blend() -> Dict[str, Any]:
    """Get Blend (75% Momentum / 25% Value) strategy metrics."""
    metrics_path = ARTIFACTS_DIR / "metrics_blend.csv"
    if not metrics_path.exists():
        return {"metrics": {}, "error": "Blend metrics not found"}
    try:
        df = pd.read_csv(metrics_path)
        return {"metrics": df.iloc[0].to_dict() if len(df) > 0 else {}, "source": "artifacts/metrics_blend.csv"}
    except Exception as e:
        return {"metrics": {}, "error": str(e)}


@router.get("/performance_blend")
async def get_performance_blend() -> Dict[str, Any]:
    """Get Blend strategy performance time series."""
    perf_path = ARTIFACTS_DIR / "performance_blend.csv"
    if not perf_path.exists():
        return {"performance": [], "error": "Blend performance not found"}
    try:
        df = pd.read_csv(perf_path)
        return {"performance": df.to_dict(orient="records"), "count": len(df), "source": "artifacts/performance_blend.csv"}
    except Exception as e:
        return {"performance": [], "error": str(e)}


@router.get("/rebalances_blend")
async def get_rebalances_blend() -> Dict[str, Any]:
    """Get Blend strategy rebalance history."""
    rebal_path = ARTIFACTS_DIR / "rebalances_blend.csv"
    if not rebal_path.exists():
        return {"rebalances": [], "error": "Blend rebalances not found"}
    try:
        df = pd.read_csv(rebal_path)
        return {"rebalances": df.to_dict(orient="records"), "count": len(df), "source": "artifacts/rebalances_blend.csv"}
    except Exception as e:
        return {"rebalances": [], "error": str(e)}


@router.get("/holding_periods_blend")
async def get_holding_periods_blend() -> Dict[str, Any]:
    """Get Blend strategy holding periods."""
    hold_path = ARTIFACTS_DIR / "holding_periods_blend.csv"
    if not hold_path.exists():
        return {"holding_periods": [], "error": "Blend holding periods not found"}
    try:
        df = pd.read_csv(hold_path)
        return {"holding_periods": df.to_dict(orient="records"), "count": len(df), "source": "artifacts/holding_periods_blend.csv"}
    except Exception as e:
        return {"holding_periods": [], "error": str(e)}
