"""Triangulation: blend several independent demand signals into one nowcast.

Runs the single-signal forecast for each driver (Google Trends, Wikipedia, GDELT,
FRED macro), then combines their predicted YoY into an ensemble weighted by each
driver's out-of-sample skill. A driver that fails (rate-limit, no data) is
skipped with a note rather than sinking the whole analysis; the spread across
drivers is reported as an agreement/divergence signal.
"""

from __future__ import annotations

from collections import Counter
from statistics import pstdev

from ..config import Settings, get_settings
from ..entities.resolver import resolve as resolve_entity
from ..features.align import add_quarters, quarter_of
from ..models import DriverContribution, Entity, TriangulationResult
from ..registry import get_connector
from .forecast import _load_driver, _revenue_levels_and_yoy, _trim_signal, forecast_kpi

DEFAULT_DRIVERS = ["google_trends", "wikipedia", "gdelt"]


def skill_weights(skills: list[float | None]) -> list[float]:
    """Normalized ensemble weights from per-driver out-of-sample skill: positive
    skill -> proportional weight; if no driver has positive skill, weight equally."""
    raw = [max(s, 0.0) if s is not None else 0.0 for s in skills]
    if sum(raw) <= 0:
        raw = [1.0] * len(skills)
    total = sum(raw)
    return [r / total for r in raw] if total > 0 else []


def triangulate(
    query: str,
    *,
    drivers: list[str] | None = None,
    geo: str = "US",
    max_lag: int = 4,
    quarters: int = 16,
    alpha: float = 0.20,
    min_n: int = 6,
    lag_by: str = "skill",
    sign: str = "any",
    store=None,
    settings: Settings | None = None,
) -> tuple[Entity, TriangulationResult]:
    settings = settings or get_settings()
    edgar = get_connector("edgar", store, settings)
    entity = resolve_entity(query, settings=settings, store=store, edgar=edgar)
    if not entity.cik:
        raise RuntimeError(f"Could not resolve {query!r} to a SEC filer.")
    revenue = _trim_signal(edgar.quarterly_revenue(entity.cik), keep=quarters + 5)
    levels = _revenue_levels_and_yoy(revenue)[0]

    if drivers is None:
        drivers = list(DEFAULT_DRIVERS)
        if entity.macro_series and get_connector("fred", store, settings).available():
            drivers.append("fred")

    res = TriangulationResult(entity_key=entity.key)
    for drv in drivers:
        drv_term = None
        if drv == "fred":
            if not entity.macro_series:
                res.notes.append("Skipped fred: entity has no mapped macro series.")
                continue
            drv_term = entity.macro_series[0]
        try:
            sig, label = _load_driver(entity, drv, drv_term, None, geo, quarters, store, settings)
        except Exception as exc:  # noqa: BLE001 - one bad driver shouldn't sink the rest
            res.warnings.append(f"Driver '{drv}' unavailable: {type(exc).__name__}: {exc}")
            continue
        fr = forecast_kpi(
            entity, revenue, sig, driver_label=label, max_lag=max_lag,
            alpha=alpha, min_n=min_n, lag_by=lag_by, sign=sign,
        )
        skill = next((ls.skill for ls in fr.lag_table if ls.lag == fr.best_lag), None)
        res.drivers.append(
            DriverContribution(
                label=label, source=sig.source, n=fr.n_obs, lag=fr.best_lag,
                corr=fr.corr, skill=skill, predicted_yoy=fr.predicted_yoy,
                target_period=fr.target_period,
            )
        )
        if fr.warnings:  # surface per-driver caveats instead of silently dropping them
            res.warnings.append(f"[{sig.source}] " + "; ".join(fr.warnings))

    forecasts = [d for d in res.drivers if d.predicted_yoy is not None and d.target_period]
    if not forecasts:
        res.warnings.append("No driver produced a forecast.")
        return entity, res

    # Blend only forecasts for ONE quarter so we combine comparable numbers. Pick the
    # quarter the most drivers target; exclude (don't silently mix) any that differ.
    consensus = Counter(d.target_period for d in forecasts).most_common(1)[0][0]
    res.target_period = consensus
    on_target = [d for d in forecasts if d.target_period == consensus]
    for d in forecasts:
        if d.target_period != consensus:
            res.notes.append(
                f"Excluded {d.source} from the ensemble: it targets "
                f"{d.target_period.year} Q{quarter_of(d.target_period)}, not "
                f"{consensus.year} Q{quarter_of(consensus)}."
            )

    # Weight by positive out-of-sample skill; if nothing beats naive, weight equally.
    skills = [d.skill for d in on_target]
    if not any(s is not None and s > 0 for s in skills):
        res.warnings.append(
            "No driver beats naive persistence out-of-sample; using equal weights (low confidence)."
        )
    for d, w in zip(on_target, skill_weights(skills)):
        d.weight = w

    res.ensemble_yoy = sum(d.weight * d.predicted_yoy for d in on_target)
    preds = [d.predicted_yoy for d in on_target]
    res.agreement_stdev = pstdev(preds) if len(preds) >= 2 else 0.0

    base = levels.get(add_quarters(consensus, -4))
    if base is not None:
        res.base_level = base
        res.ensemble_level = base * (1 + res.ensemble_yoy)
    return entity, res
