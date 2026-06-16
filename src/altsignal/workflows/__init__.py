"""Analytical workflows built on top of signals."""

from .forecast import forecast_kpi, run_forecast
from .multifactor import run_multifactor
from .screen import screen
from .triangulate import triangulate

__all__ = ["run_forecast", "forecast_kpi", "triangulate", "screen", "run_multifactor"]
