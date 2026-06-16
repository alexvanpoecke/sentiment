"""Scheduled signal refresh -> point-in-time panel.

Pulls each watchlist company's revenue (EDGAR) and demand drivers and appends a
dated *vintage* of every series to the store's panel (see ``Store.record_panel``).
Run it on a schedule (cron / Task Scheduler): over time you accumulate a real
as-of history so backtests can use only what was knowable at each past date,
instead of today's revised numbers.

Reuses the forecast workflow's resolution + driver loading, so a series is
fetched exactly the way the forecast/triangulate workflows fetch it.
"""

from __future__ import annotations

import tomllib
from datetime import date

from ..config import Settings, get_settings
from ..models import RefreshCapture, RefreshResult, Signal
from ..store import get_store
from .forecast import SCALABLE_DRIVERS, _load_driver, resolve_and_revenue


def load_watchlist(settings: Settings | None = None) -> dict:
    """Read configs/watchlist.toml. Returns {} if it's absent."""
    settings = settings or get_settings()
    path = settings.industries_path.parent / "watchlist.toml"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def refresh(
    tickers: list[str] | None = None,
    *,
    drivers: list[str] | None = None,
    geo: str = "US",
    quarters: int | None = None,
    captured_at: date | None = None,
    store=None,
    settings: Settings | None = None,
) -> RefreshResult:
    """Capture revenue + drivers for each ticker into the point-in-time panel.

    ``tickers``/``drivers`` fall back to configs/watchlist.toml, then to the
    scalable defaults. One bad ticker or driver is recorded as a failed capture
    and never sinks the rest of the run.
    """
    settings = settings or get_settings()
    store = store if store is not None else get_store()
    captured_at = captured_at or date.today()

    wl = load_watchlist(settings)
    tickers = tickers or wl.get("tickers") or []
    drivers = drivers or wl.get("drivers") or list(SCALABLE_DRIVERS)
    # `is None` (not falsy): an explicit quarters=0 must not silently become 16.
    quarters = quarters if quarters is not None else wl.get("quarters", 16)
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]

    res = RefreshResult(captured_at=captured_at, tickers=tickers)
    if not tickers:
        res.warnings.append("No tickers given and configs/watchlist.toml has none.")
        return res

    for ticker in tickers:
        try:
            entity, revenue = resolve_and_revenue(
                ticker, quarters=quarters, store=store, settings=settings
            )
        except Exception as exc:  # noqa: BLE001 - one bad ticker shouldn't sink the run
            res.captures.append(
                RefreshCapture(
                    entity_key=ticker, source="edgar", metric="revenue",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        _capture(res, store, entity.key, revenue, captured_at)
        for drv in drivers:
            drv_term = None
            if drv == "fred":
                if not entity.macro_series:  # nothing to fetch — skip, don't fail every run
                    res.notes.append(f"{entity.key}: skipped fred (no mapped macro series).")
                    continue
                drv_term = entity.macro_series[0]
            try:
                sig, _label = _load_driver(
                    entity, drv, drv_term, None, geo, quarters, store, settings
                )
            except Exception as exc:  # noqa: BLE001
                res.captures.append(
                    RefreshCapture(
                        entity_key=entity.key, source=drv, metric="?",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            _capture(res, store, entity.key, sig, captured_at)

    return res


def _capture(res: RefreshResult, store, entity_key: str, sig: Signal, captured_at: date) -> None:
    n = store.record_panel(entity_key, sig, captured_at)
    res.captures.append(
        RefreshCapture(
            entity_key=entity_key, source=sig.source, metric=sig.metric, geo=sig.geo, n_obs=n
        )
    )
