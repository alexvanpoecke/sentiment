"""Analytical workflows built on top of signals."""

from .forecast import forecast_kpi, run_forecast
from .triangulate import triangulate

__all__ = ["run_forecast", "forecast_kpi", "triangulate"]
