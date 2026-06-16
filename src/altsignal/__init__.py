"""altsignal — non-proprietary alternative-data signals for public companies."""

from __future__ import annotations

__version__ = "0.1.0"

from .models import (
    DriverContribution,
    Entity,
    FactorCoef,
    ForecastResult,
    LagStat,
    MultiFactorResult,
    Observation,
    ScreenRow,
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
    "ScreenRow",
    "FactorCoef",
    "MultiFactorResult",
    "__version__",
]
