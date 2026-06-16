"""altsignal — non-proprietary alternative-data signals for public companies."""

from __future__ import annotations

__version__ = "0.1.0"

from .models import (
    DriverContribution,
    Entity,
    ForecastResult,
    LagStat,
    Observation,
    Signal,
    TriangulationResult,
)

__all__ = [
    "Entity",
    "Signal",
    "Observation",
    "ForecastResult",
    "LagStat",
    "DriverContribution",
    "TriangulationResult",
    "__version__",
]
