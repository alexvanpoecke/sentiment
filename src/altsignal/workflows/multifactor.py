"""Multifactor forecast: one regression on several signals (+ optional seasonality).

Unlike triangulation (which forecasts from each signal separately and averages),
this combines all drivers as *features in a single multiple regression*:
    revenueYoY ~ b0 + b1*driver1YoY[lag1] + b2*driver2YoY[lag2] + ... (+ quarter dummies)
Each driver enters at its own best lag. Walk-forward skill vs naive persistence
tells you whether the combined model actually generalizes.
"""

from __future__ import annotations

from ..config import Settings, get_settings
from ..features.align import add_quarters, quarter_of
from ..features.lag import scan_lags, skill_score
from ..features.stats import fit_ols_multi, mean
from ..models import Entity, FactorCoef, MultiFactorResult, Signal
from .forecast import (
    DEFAULT_DRIVERS,
    _driver_quarterly_yoy,
    _load_driver,
    _revenue_levels_and_yoy,
    resolve_and_revenue,
)

_SEASON_QUARTERS = (1, 2, 3)  # Q1/Q2/Q3 dummies; Q4 is the baseline


def assemble_design(
    target_yoy: dict,
    driver_feats: list[tuple[str, int, dict]],
    seasonal: bool,
) -> tuple[list, list[list[float]], list[float], list[str]]:
    """Build (quarters, X, y, feature_names) from per-driver lagged feature maps.

    ``driver_feats`` is a list of (name, lag, {quarter: value}). Rows are the
    quarters present in the target AND every driver's map (inner join)."""
    common = set(target_yoy)
    for _name, _lag, fmap in driver_feats:
        common &= set(fmap)
    quarters = sorted(common)

    names = [f"{name} (lag {lag})" for name, lag, _f in driver_feats]
    season_quarters: list[int] = []
    if seasonal:
        # Only dummy the calendar quarters actually present; an absent quarter's
        # column would be all-zeros -> singular X'X. Absent quarters (and Q4) are baseline.
        present = {quarter_of(q) for q in quarters}
        season_quarters = [s for s in _SEASON_QUARTERS if s in present]
        names += [f"Q{s}" for s in season_quarters]

    x: list[list[float]] = []
    y: list[float] = []
    for q in quarters:
        row = [fmap[q] for _n, _lag, fmap in driver_feats]
        if season_quarters:
            qn = quarter_of(q)
            row += [1.0 if qn == s else 0.0 for s in season_quarters]
        x.append(row)
        y.append(target_yoy[q])
    return quarters, x, y, names


def forecast_multi(
    entity: Entity,
    kpi_signal: Signal,
    drivers: list[tuple[Signal, str]],
    *,
    max_lag: int = 4,
    alpha: float = 0.20,
    min_n: int = 6,
    lag_by: str = "skill",
    sign: str = "any",
    seasonal: bool = False,
) -> MultiFactorResult:
    levels, target_yoy, _collisions = _revenue_levels_and_yoy(kpi_signal)
    res = MultiFactorResult(
        entity_key=entity.key, kpi_metric=kpi_signal.metric, kpi_source=kpi_signal.source,
        seasonal=seasonal, alpha=alpha,
    )

    # Each driver enters at its own best lag; keep its full YoY map for forecasting.
    driver_feats: list[tuple[str, int, dict]] = []
    driver_dyoys: list[dict] = []  # parallel to driver_feats (positional, not keyed by label)
    for sig, label in drivers:
        dyoy = _driver_quarterly_yoy(sig)
        if len(dyoy) < 3 or len(target_yoy) < 3:
            res.warnings.append(f"{label}: insufficient overlapping history; skipped.")
            continue
        best_lag, _table = scan_lags(
            dyoy, target_yoy, max_lag=max_lag, min_n=min_n, lag_by=lag_by, sign=sign
        )
        fmap = {q: dyoy[add_quarters(q, -best_lag)] for q in target_yoy if add_quarters(q, -best_lag) in dyoy}
        driver_feats.append((label, best_lag, fmap))
        driver_dyoys.append(dyoy)
    res.driver_labels = [f"{n} (lag {lag})" for n, lag, _f in driver_feats]

    if not driver_feats:
        res.warnings.append("No usable drivers.")
        return res

    quarters, x, y, names = assemble_design(target_yoy, driver_feats, seasonal)
    res.n_obs = len(quarters)
    k = len(names)
    if res.n_obs < k + 2:
        res.warnings.append(
            f"Only {res.n_obs} aligned quarters for {k} features; insufficient to fit. "
            f"Use fewer drivers or drop --seasonal."
        )
        return res

    try:
        reg = fit_ols_multi(x, y)
    except ValueError as exc:
        res.warnings.append(f"Could not fit multifactor model ({exc}).")
        return res
    res.r2 = reg.r2
    res.features = [FactorCoef("intercept", reg.coef[0], reg.t[0], reg.p[0])]
    res.features += [
        FactorCoef(name, reg.coef[i], reg.t[i], reg.p[i]) for i, name in enumerate(names, start=1)
    ]
    if res.n_obs < k + 5:
        res.warnings.append(
            f"Small sample (n={res.n_obs}, {k} features): coefficients are noisy / prone to overfit."
        )

    # --- forecast the next quarter ---
    qf = add_quarters(quarters[-1], 1)
    x0: list[float] = []
    have_all = True
    for idx, (_label, lag, _fmap) in enumerate(driver_feats):
        dk = add_quarters(qf, -lag)
        dyoy = driver_dyoys[idx]
        if dk not in dyoy:
            have_all = False
            break
        x0.append(dyoy[dk])
    if have_all:
        # Append seasonal dummies matching exactly the columns that survived in the
        # design (driver columns come first; absent-quarter dummies were dropped).
        qn = quarter_of(qf)
        x0 += [1.0 if season_name == f"Q{qn}" else 0.0 for season_name in names[len(driver_feats):]]
        yhat, lo, hi = reg.predict(x0, alpha=alpha)
        res.target_period = qf
        res.predicted_yoy, res.pi_low_yoy, res.pi_high_yoy = yhat, lo, hi
        base = levels.get(add_quarters(qf, -4))
        if base is not None:
            res.base_level = base
            res.predicted_level = base * (1 + yhat)
            res.pi_low_level = base * (1 + lo)
            res.pi_high_level = base * (1 + hi)
        else:
            res.warnings.append("No year-ago revenue level to convert YoY to a $ level.")
    else:
        res.warnings.append("Missing a current driver reading at its lag; cannot project forward.")

    # --- walk-forward backtest vs naive persistence ---
    errs, naive = [], []
    for i in range(k + 2, res.n_obs):
        prev_q = add_quarters(quarters[i], -1)
        if prev_q not in target_yoy:
            continue
        try:
            reg_i = fit_ols_multi(x[:i], y[:i])
        except ValueError:
            continue
        errs.append(abs(reg_i.predict(x[i])[0] - y[i]))
        naive.append(abs(target_yoy[prev_q] - y[i]))
    if errs:
        res.backtest_n = len(errs)
        res.backtest_mae = mean(errs)
        res.backtest_naive_mae = mean(naive)
        res.skill = skill_score(res.backtest_mae, res.backtest_naive_mae)
        if res.skill is not None and res.skill <= 0:
            res.warnings.append(
                "Multifactor model does not beat naive persistence out-of-sample "
                "(negative skill); treat as low-confidence."
            )
    return res


def run_multifactor(
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
    seasonal: bool = False,
    store=None,
    settings: Settings | None = None,
) -> tuple[Entity, MultiFactorResult]:
    settings = settings or get_settings()
    entity, revenue = resolve_and_revenue(query, quarters=quarters, store=store, settings=settings)

    drivers = drivers or list(DEFAULT_DRIVERS)
    loaded: list[tuple[Signal, str]] = []
    load_warnings: list[str] = []
    for drv in drivers:
        drv_term = entity.macro_series[0] if drv == "fred" and entity.macro_series else None
        try:
            loaded.append(_load_driver(entity, drv, drv_term, None, geo, quarters, store, settings))
        except Exception as exc:  # noqa: BLE001
            load_warnings.append(f"Driver '{drv}' unavailable: {type(exc).__name__}: {exc}")

    res = forecast_multi(
        entity, revenue, loaded, max_lag=max_lag, alpha=alpha, min_n=min_n,
        lag_by=lag_by, sign=sign, seasonal=seasonal,
    )
    res.warnings = load_warnings + res.warnings
    return entity, res
