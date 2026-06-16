"""Signal -> KPI forecast workflow (the headline example, generalized).

Given a company, a KPI (quarterly revenue from EDGAR) and a demand *driver*
(Google Trends / Wikipedia / FRED / CSV):
  1. align both to calendar quarters and take YoY growth
  2. search lags 0..max_lag for where the driver best leads revenue
  3. fit OLS at the best lag
  4. forecast the upcoming quarter with a prediction interval
  5. walk-forward backtest vs a naive persistence benchmark
"""

from __future__ import annotations

import csv
import re
from datetime import date
from pathlib import Path

from ..config import Settings, get_settings
from ..connectors.base import ConnectorError
from ..entities.resolver import resolve as resolve_entity
from ..features.align import add_quarters, calendar_quarter_end, to_quarterly, yoy_quarterly
from ..features.lag import lagged_pairs, scan_lags
from ..features.stats import fit_ols
from ..models import Entity, ForecastResult, Observation, Signal
from ..registry import get_connector


def _trim_signal(sig: Signal, keep: int) -> Signal:
    obs = sig.sorted()[-keep:]
    return Signal(
        entity_key=sig.entity_key,
        source=sig.source,
        metric=sig.metric,
        freq=sig.freq,
        geo=sig.geo,
        unit=sig.unit,
        observations=obs,
        meta=dict(sig.meta),
    )


# Shared default driver set (triangulate/multifactor); screen overrides it.
DEFAULT_DRIVERS = ["google_trends", "wikipedia", "gdelt"]


def resolve_and_revenue(
    query: str, *, quarters: int = 16, store=None, settings: Settings | None = None, edgar=None
) -> tuple[Entity, Signal]:
    """Shared workflow prologue: resolve a query to an Entity and fetch its trimmed
    quarterly revenue. Raises RuntimeError if the query isn't a SEC filer."""
    settings = settings or get_settings()
    edgar = edgar or get_connector("edgar", store, settings)
    entity = resolve_entity(query, settings=settings, store=store, edgar=edgar)
    if not entity.cik:
        raise RuntimeError(f"Could not resolve {query!r} to a SEC filer.")
    revenue = _trim_signal(edgar.quarterly_revenue(entity.cik), keep=quarters + 5)
    return entity, revenue


def _revenue_levels_and_yoy(sig: Signal) -> tuple[dict[date, float], dict[date, float], int]:
    """Calendar-quarter-keyed revenue levels and (gap-safe, date-based) YoY growth.

    Returns (levels, yoy, collisions) where ``collisions`` counts fiscal quarters
    that mapped onto an already-used calendar quarter (the later one is kept).
    """
    levels: dict[date, float] = {}
    collisions = 0
    for o in sig.sorted():  # ascending: a later fiscal quarter wins a collision
        k = calendar_quarter_end(o.ts)
        if k in levels:
            collisions += 1
        levels[k] = o.value
    # `levels` is already ascending (sig.sorted() is ascending and calendar_quarter_end
    # is monotonic), and is only read by key, so no re-sort is needed.
    return levels, yoy_quarterly(levels), collisions


def _driver_quarterly_yoy(sig: Signal) -> dict[date, float]:
    # Always aggregate by mean: YoY is scale-invariant, so mean and sum agree for
    # full quarters, and mean is robust to partial boundary quarters (e.g. an
    # in-progress current quarter) where a sum would be biased low.
    q = to_quarterly(sig.series(), agg="mean")
    return yoy_quarterly(q)


def forecast_kpi(
    entity: Entity,
    kpi_signal: Signal,
    driver_signal: Signal,
    *,
    driver_label: str,
    max_lag: int = 4,
    alpha: float = 0.20,
    min_n: int = 6,
    lag_by: str = "skill",
    sign: str = "any",
) -> ForecastResult:
    levels, target_yoy, collisions = _revenue_levels_and_yoy(kpi_signal)
    driver_yoy = _driver_quarterly_yoy(driver_signal)

    res = ForecastResult(
        entity_key=entity.key,
        kpi_metric=kpi_signal.metric,
        kpi_source=kpi_signal.source,
        driver_metric=driver_signal.metric,
        driver_source=driver_signal.source,
        driver_label=driver_label,
        alpha=alpha,
    )
    res.notes.append(
        f"Revenue concept: {kpi_signal.meta.get('concept', 'n/a')}; "
        f"driver aggregated to quarters by mean; YoY is date-based (gap-safe)."
    )
    res.notes.append(
        "Lag chosen by "
        + ("out-of-sample backtest skill" if lag_by == "skill" else "in-sample correlation")
        + ("" if sign == "any" else f", constrained to {sign} correlation")
        + "."
    )
    if collisions:
        res.warnings.append(
            f"{collisions} fiscal quarter(s) collided onto the same calendar quarter "
            f"(off-calendar fiscal year); kept the later one."
        )
    unit = driver_signal.unit or ""
    if "index" in unit or "intensity" in unit:
        res.notes.append("Driver is a relative measure (not absolute counts); analysis uses YoY.")

    if len(target_yoy) < 3 or len(driver_yoy) < 3:
        res.warnings.append("Not enough overlapping history to estimate a relationship.")
        return res

    best_lag, table = scan_lags(
        driver_yoy, target_yoy, max_lag=max_lag, min_n=min_n, lag_by=lag_by, sign=sign
    )
    res.lag_table = table
    res.best_lag = best_lag
    chosen = next((ls for ls in table if ls.lag == best_lag), None)

    xs, ys, qs = lagged_pairs(driver_yoy, target_yoy, best_lag)
    res.n_obs = len(xs)
    if len(xs) < 3:
        res.warnings.append(f"Only {len(xs)} aligned points at lag {best_lag}; cannot fit.")
        return res

    if chosen is not None:
        res.corr, res.corr_p = chosen.r, chosen.p_value
        if chosen.folds:  # backtest was already computed during the lag scan
            res.backtest_n = chosen.folds
            res.backtest_mae_yoy = chosen.model_mae
            res.backtest_naive_mae_yoy = chosen.naive_mae

    if res.n_obs < min_n:
        res.warnings.append(
            f"Small sample (n={res.n_obs}); estimates are noisy. Treat as directional."
        )
    # Multiple-testing caution: the winning lag was the best of (max_lag+1) candidates.
    adj_p = min(1.0, res.corr_p * len(table))
    if adj_p > 0.10:
        res.warnings.append(
            f"Correlation not significant after multiple-testing correction "
            f"(Bonferroni x{len(table)}: adj p={adj_p:.2f})."
        )
    if lag_by == "skill" and chosen is not None and chosen.skill is not None and chosen.skill <= 0:
        res.warnings.append(
            "Best lag still loses to naive persistence out-of-sample (negative skill); "
            "treat the forecast as low-confidence."
        )

    try:
        reg = fit_ols(xs, ys)
    except ValueError as exc:
        res.warnings.append(f"Could not fit regression ({exc}); reporting correlation only.")
        return res
    res.slope, res.intercept = reg.slope, reg.intercept
    res.r2, res.resid_std = reg.r2, reg.resid_std

    # --- forecast the nearest upcoming quarter we have a driver reading for ---
    last_t = qs[-1]
    for step in range(1, max_lag + 2):
        cand = add_quarters(last_t, step)
        dk = add_quarters(cand, -best_lag)
        if dk in driver_yoy:
            res.target_period = cand
            res.current_driver_yoy = driver_yoy[dk]
            break

    if res.target_period is not None and res.current_driver_yoy is not None:
        yhat, lo, hi = reg.predict(res.current_driver_yoy, alpha=alpha)
        res.predicted_yoy, res.pi_low_yoy, res.pi_high_yoy = yhat, lo, hi
        base = levels.get(add_quarters(res.target_period, -4))
        if base is not None:
            res.base_level = base
            res.predicted_level = base * (1 + yhat)
            res.pi_low_level = base * (1 + lo)
            res.pi_high_level = base * (1 + hi)
        else:
            res.warnings.append("No year-ago revenue level available to convert YoY to a $ level.")
    else:
        res.warnings.append("No current driver reading at the required lag; cannot project forward.")

    return res


# --------------------------------------------------------------------------- #
# Orchestration: ticker in, (Entity, ForecastResult) out                       #
# --------------------------------------------------------------------------- #
def _load_csv_driver(path: str) -> tuple[Signal, str]:
    rows: list[Observation] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 2:
                continue
            try:
                d = date.fromisoformat(row[0].strip()[:10])
                v = float(row[1])
            except ValueError:
                if i == 0:
                    continue  # header
                raise
            rows.append(Observation(ts=d, value=v))
    if not rows:
        raise ValueError(f"no (date,value) rows parsed from {path}")
    sig = Signal(
        entity_key=f"csv:{Path(path).stem}",
        source="csv",
        metric="driver",
        freq="M",
        observations=rows,
    )
    return sig, f"CSV: {Path(path).name}"


def _wikipedia_page_candidates(entity: Entity, page: str | None) -> list[str]:
    """Article title candidates from most to least specific. If `page` is explicit, use only that."""
    if page:
        return [page.strip()]
    seen: set[str] = set()
    out: list[str] = []

    def add(t: str) -> None:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)

    add(entity.short_name)
    # Strip "/ STATE" suffix that SEC appends (e.g. "Rivian Automotive / DE")
    stripped = re.split(r"\s+/", entity.short_name)[0].strip()
    add(stripped)
    # First two words (e.g. "Rivian Automotive")
    words = stripped.split()
    if len(words) > 2:
        add(" ".join(words[:2]))
    # First word only (e.g. "Rivian")
    if words:
        add(words[0])
    return out


def _load_driver(
    entity: Entity, driver: str, term: str | None, page: str | None, geo: str, quarters: int,
    store, settings,
) -> tuple[Signal, str]:
    if driver == "google_trends":
        t = term or (entity.seed_terms[0] if entity.seed_terms else entity.short_name)
        conn = get_connector("google_trends", store, settings)
        sig = conn.fetch(term=t, geo=geo, quarters=quarters)[0]
        return sig, f'Google Trends: "{t}" ({geo})'
    if driver == "wikipedia":
        conn = get_connector("wikipedia", store, settings)
        candidates = _wikipedia_page_candidates(entity, page)
        last_exc: ConnectorError | None = None
        for pg in candidates:
            try:
                sig = conn.fetch(page=pg, quarters=quarters)[0]
                return sig, f"Wikipedia pageviews: {pg}"
            except ConnectorError as exc:
                if exc.status == 404:
                    last_exc = exc
                    continue
                raise
        raise last_exc or ConnectorError(
            f"wikipedia: no article found for {entity.short_name!r}", source="wikipedia"
        )
    if driver == "fred":
        if not term:
            raise ValueError("driver 'fred' needs --term SERIES_ID")
        conn = get_connector("fred", store, settings)
        sig = conn.fetch(series_id=term, quarters=quarters)[0]
        return sig, f"FRED: {term}"
    if driver == "gdelt":
        q = term or entity.short_name or entity.name
        conn = get_connector("gdelt", store, settings)
        sig = conn.fetch(query=q, quarters=quarters)[0]
        return sig, f"GDELT news volume: {q}"
    raise ValueError(f"unknown driver {driver!r} (use google_trends|wikipedia|fred|gdelt|csv)")


def run_forecast(
    query: str,
    *,
    driver: str = "google_trends",
    term: str | None = None,
    page: str | None = None,
    geo: str = "US",
    max_lag: int = 4,
    quarters: int = 16,
    alpha: float = 0.20,
    min_n: int = 6,
    lag_by: str = "skill",
    sign: str = "any",
    driver_csv: str | None = None,
    store=None,
    settings: Settings | None = None,
) -> tuple[Entity, ForecastResult]:
    settings = settings or get_settings()
    entity, revenue = resolve_and_revenue(query, quarters=quarters, store=store, settings=settings)

    if driver_csv:
        driver_signal, driver_label = _load_csv_driver(driver_csv)
    else:
        driver_signal, driver_label = _load_driver(
            entity, driver, term, page, geo, quarters, store, settings
        )

    result = forecast_kpi(
        entity,
        revenue,
        driver_signal,
        driver_label=driver_label,
        max_lag=max_lag,
        alpha=alpha,
        min_n=min_n,
        lag_by=lag_by,
        sign=sign,
    )
    return entity, result
