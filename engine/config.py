"""
Engine configuration module.

Design decisions:
- All config is passed explicitly, no global state
- Paths are relative to project root for portability
- Configuration is immutable after creation
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class EngineConfig:
    """
    Immutable configuration for the strategy engine.
    
    All paths are relative to project root and resolved at runtime.
    This ensures reproducibility across different environments.
    """
    
    # Project root is determined relative to this file's location
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    # Asset universe settings
    universe_name: str = "nifty500"
    
    # Factor settings
    enabled_factors: List[str] = field(default_factory=lambda: ["momentum", "value"])
    
    # Portfolio settings
    max_positions: int = 30
    rebalance_frequency: str = "monthly"  # "weekly" or "monthly"
    
    # Artifact paths (relative to project root)
    artifacts_dir: str = "artifacts"
    
    @property
    def artifacts_path(self) -> Path:
        return self.project_root / self.artifacts_dir
    
    @property
    def portfolio_path(self) -> Path:
        return self.artifacts_path / "portfolio.csv"
    
    @property
    def performance_path(self) -> Path:
        return self.artifacts_path / "performance.csv"


def get_default_config() -> EngineConfig:
    """Factory function for default configuration."""
    return EngineConfig()
