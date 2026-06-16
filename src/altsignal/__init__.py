"""altsignal — non-proprietary alternative-data signals for public companies."""

from __future__ import annotations

__version__ = "0.1.0"

from .models import Entity, ForecastResult, LagStat, Observation, Signal

__all__ = [
    "Entity",
    "Signal",
    "Observation",
    "ForecastResult",
    "LagStat",
    "__version__",
]
