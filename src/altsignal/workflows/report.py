"""Report builder: assemble one company dossier from several workflows.

Composes the triangulated ensemble nowcast and the multifactor regression for a
company, plus its point-in-time panel coverage, into a single research dossier.
It reuses the existing workflow functions (the shared SQLite raw cache means the
repeated resolve/fetch is cheap), so the dossier never diverges from what the
individual commands report.
"""

from __future__ import annotations

from ..config import Settings, get_settings
from ..models import Entity, MultiFactorResult, TriangulationResult
from ..store import get_store
from .forecast import resolve_and_revenue
from .multifactor import run_multifactor
from .triangulate import triangulate


def build_report(
    query: str,
    *,
    drivers: list[str] | None = None,
    seasonal: bool = False,
    geo: str = "US",
    max_lag: int = 4,
    quarters: int = 16,
    alpha: float = 0.20,
    min_n: int = 6,
    lag_by: str = "skill",
    sign: str = "any",
    store=None,
    settings: Settings | None = None,
) -> tuple[Entity, TriangulationResult, MultiFactorResult, list[dict]]:
    """Run triangulation + multifactor for ``query`` and gather panel coverage.

    Returns (entity, triangulation_result, multifactor_result, panel_rows).
    """
    settings = settings or get_settings()
    store = store if store is not None else get_store()

    # Resolve + fetch revenue once, then thread the result into both workflows so
    # entity resolution and the EDGAR companyfacts parse don't run twice per dossier.
    entity, revenue = resolve_and_revenue(query, quarters=quarters, store=store, settings=settings)

    common = dict(
        drivers=drivers, geo=geo, max_lag=max_lag, quarters=quarters,
        alpha=alpha, min_n=min_n, lag_by=lag_by, sign=sign,
        entity=entity, revenue=revenue, store=store, settings=settings,
    )
    _, tri = triangulate(query, **common)
    _, mf = run_multifactor(query, seasonal=seasonal, **common)
    panel_rows = store.panel_summary(entity.key)
    return entity, tri, mf, panel_rows
