"""Cross-sectional skill screen.

Backtests each (company x driver) pair across a ticker universe and ranks by
out-of-sample skill, so you can see which signals actually beat naive persistence
and for which companies — before investing in more connectors or premium data.
Reuses the single-signal forecast (forecast_kpi) and its walk-forward skill.
"""

from __future__ import annotations

from ..config import Settings, get_settings
from ..models import ScreenRow
from .forecast import _load_driver, forecast_kpi, resolve_and_revenue

# Free, scalable defaults (Google Trends rate-limits hard in bulk -> opt-in only).
DEFAULT_DRIVERS = ["wikipedia", "gdelt"]


def _rank_key(row: ScreenRow) -> tuple[bool, float]:
    # rows with a skill score first (highest skill first); errors / no-skill last
    return (row.skill is None, -(row.skill if row.skill is not None else 0.0))


def screen(
    tickers: list[str],
    *,
    drivers: list[str] | None = None,
    max_lag: int = 4,
    quarters: int = 16,
    min_n: int = 6,
    lag_by: str = "skill",
    sign: str = "any",
    store=None,
    settings: Settings | None = None,
) -> list[ScreenRow]:
    settings = settings or get_settings()
    drivers = drivers or list(DEFAULT_DRIVERS)
    rows: list[ScreenRow] = []

    for ticker in tickers:
        try:
            entity, revenue = resolve_and_revenue(
                ticker, quarters=quarters, store=store, settings=settings
            )
        except Exception as exc:  # noqa: BLE001 - one bad ticker shouldn't sink the screen
            rows.append(ScreenRow(ticker=ticker, error=f"{type(exc).__name__}: {exc}"))
            continue

        for drv in drivers:
            drv_term = entity.macro_series[0] if drv == "fred" and entity.macro_series else None
            try:
                sig, _label = _load_driver(entity, drv, drv_term, None, "US", quarters, store, settings)
                fr = forecast_kpi(
                    entity, revenue, sig, driver_label=drv, max_lag=max_lag,
                    min_n=min_n, lag_by=lag_by, sign=sign,
                )
                skill = next((ls.skill for ls in fr.lag_table if ls.lag == fr.best_lag), None)
                rows.append(
                    ScreenRow(
                        ticker=entity.ticker or ticker, name=entity.short_name, driver=drv,
                        n=fr.n_obs, lag=fr.best_lag, corr=fr.corr, skill=skill,
                        predicted_yoy=fr.predicted_yoy, target_period=fr.target_period,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    ScreenRow(
                        ticker=entity.ticker or ticker, name=entity.short_name, driver=drv,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

    rows.sort(key=_rank_key)
    return rows
