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
