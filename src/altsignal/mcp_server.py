"""MCP server: expose altsignal's connectors and workflows as tools.

Lets an MCP client (e.g. Claude) drive the same operations as the CLI —
resolve a company, pull a raw signal, and run the forecast / triangulate /
screen / multifactor workflows — and get structured JSON back.

This module is only imported when you run the server, so it stays behind the
optional ``mcp`` extra and never burdens the pure-Python core::

    pip install -e ".[mcp]"
    altsignal-mcp            # speaks MCP over stdio

The tools wrap the exact same workflow functions the CLI calls, so behaviour
(lag selection, prediction intervals, backtests, per-driver fallbacks) is
identical; only the presentation layer (JSON vs Rich tables) differs.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date
from typing import Any

try:  # the server is behind the optional `mcp` extra
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
    raise SystemExit(
        "altsignal MCP server needs the 'mcp' extra. Install it with:\n"
        '    pip install -e ".[mcp]"'
    ) from exc

from .config import get_settings
from .models import Entity, Signal
from .registry import all_connectors, get_connector

mcp = FastMCP(
    "altsignal",
    instructions=(
        "Build non-proprietary alternative-data signals for any US public company and "
        "forecast its next-quarter revenue from lagged demand drivers (Google Trends, "
        "Wikipedia pageviews, GDELT news volume, FRED macro series). Typical flow: "
        "resolve_company -> (forecast | triangulate | multifactor); use screen to rank a "
        "universe of tickers, and list_sources to see which connectors are usable right now. "
        "All forecasts are estimates for research, not investment advice."
    ),
)


# --------------------------------------------------------------------------- #
# Serialization helpers                                                        #
# --------------------------------------------------------------------------- #
def _jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses + dates into JSON-safe primitives."""
    if isinstance(obj, date):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _entity_dict(ent: Entity) -> dict[str, Any]:
    d = _jsonable(ent)
    d["short_name"] = ent.short_name  # a property, so asdict() omits it
    return d


def _signal_dict(sig: Signal, limit: int) -> dict[str, Any]:
    obs = sig.sorted()
    if limit and limit > 0:
        obs = obs[-limit:]
    return {
        "entity_key": sig.entity_key,
        "source": sig.source,
        "metric": sig.metric,
        "label": sig.label,
        "freq": sig.freq,
        "geo": sig.geo,
        "unit": sig.unit,
        "n": len(sig),
        "meta": _jsonable(sig.meta),
        "observations": [
            {
                "ts": o.ts.isoformat(),
                "value": o.value,
                "as_of": o.as_of.isoformat() if o.as_of else None,
            }
            for o in obs
        ],
    }


# --------------------------------------------------------------------------- #
# Tools                                                                        #
# --------------------------------------------------------------------------- #
@mcp.tool()
def list_sources() -> list[dict[str, Any]]:
    """List the registered data connectors and whether each is usable right now.

    Availability depends on API keys / network (e.g. FRED needs FRED_API_KEY,
    Reddit needs client credentials). Use this to see which drivers you can pass
    to the forecast/triangulate/screen tools.
    """
    settings = get_settings()
    out: list[dict[str, Any]] = []
    for name, cls in all_connectors().items():
        inst = cls(settings=settings)
        available = inst.available()
        out.append(
            {
                "source": name,
                "title": cls.title,
                "free": bool(cls.free),
                "available": available,
                "note": cls.note or "",
                "status_note": "" if available else inst.availability_note(),
            }
        )
    return out


@mcp.tool()
def resolve_company(query: str) -> dict[str, Any]:
    """Resolve a ticker or company name to a SEC filer, with the inferred sector,
    seed search terms, peers, macro series, and which connectors are routed to it.

    Run this first to confirm a company resolves and to discover good driver terms.
    """
    from .entities.resolver import resolve as _resolve

    return _entity_dict(_resolve(query))


@mcp.tool()
def get_signal(
    query: str,
    source: str = "edgar",
    term: str | None = None,
    page: str | None = None,
    board: str | None = None,
    geo: str = "US",
    quarters: int = 16,
    limit: int = 12,
) -> dict[str, Any]:
    """Fetch one raw time series from a single connector for a company.

    `source` is a connector id from list_sources (edgar, google_trends, wikipedia,
    fred, gdelt, reddit, greenhouse). `term` is the search term (google_trends),
    FRED series id, or news query; `page` is a Wikipedia article title; `board` is
    a Greenhouse board token. `limit` caps how many of the most recent observations
    are returned (0 = all).
    """
    from .entities.resolver import resolve as _resolve

    settings = get_settings()
    ent = _resolve(query)
    conn = get_connector(source, settings=settings)
    if source == "edgar":
        sigs = conn.fetch(cik=ent.cik, query=query, metric="revenue")
    elif source == "google_trends":
        t = term or (ent.seed_terms[0] if ent.seed_terms else ent.short_name)
        sigs = conn.fetch(term=t, geo=geo, quarters=quarters)
    elif source == "wikipedia":
        sigs = conn.fetch(page=page or ent.name or ent.short_name)
    elif source == "fred":
        sigs = conn.fetch(series_id=term)
    elif source == "gdelt":
        sigs = conn.fetch(query=term or ent.short_name, quarters=quarters)
    elif source == "reddit":
        sigs = conn.fetch(query=term or ent.short_name)
    elif source == "greenhouse":
        sigs = conn.fetch(board=board or term)
    else:
        sigs = conn.fetch(term=term, page=page, query=query)

    return {
        "entity": {"ticker": ent.ticker, "name": ent.name, "short_name": ent.short_name},
        "signals": [_signal_dict(s, limit) for s in sigs],
    }


@mcp.tool()
def forecast(
    query: str,
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
    fallback: bool = True,
) -> dict[str, Any]:
    """Forecast a company's next-quarter revenue (YoY and $ level) from one demand driver.

    Aligns revenue and the driver to calendar quarters, takes YoY growth, searches
    lags 0..max_lag for where the driver best leads revenue (by out-of-sample skill
    when lag_by='skill', else max correlation), fits OLS, projects the upcoming
    quarter with a prediction interval, and walk-forward backtests vs naive
    persistence. driver: google_trends | wikipedia | fred | gdelt. `term` defaults
    to the top seed term; `sign` can constrain the lag to 'positive'/'negative'
    correlation. If Google Trends is rate-limited and fallback=True, retries with
    Wikipedia pageviews. Check `result.warnings` before trusting the number.
    """
    from .connectors.trends import TrendsUnavailable
    from .workflows.forecast import run_forecast

    kw = dict(
        term=term, page=page, geo=geo, max_lag=max_lag, quarters=quarters,
        alpha=alpha, min_n=min_n, lag_by=lag_by, sign=sign,
    )
    used_fallback = False
    try:
        ent, res = run_forecast(query, driver=driver, **kw)
    except TrendsUnavailable:
        if fallback and driver == "google_trends":
            ent, res = run_forecast(query, driver="wikipedia", **{**kw, "term": None})
            used_fallback = True
        else:
            raise
    return {
        "entity": _entity_dict(ent),
        "result": _jsonable(res),
        "fell_back_to_wikipedia": used_fallback,
    }


@mcp.tool()
def triangulate(
    query: str,
    drivers: list[str] | None = None,
    geo: str = "US",
    max_lag: int = 4,
    quarters: int = 16,
    alpha: float = 0.20,
    min_n: int = 6,
    lag_by: str = "skill",
    sign: str = "any",
) -> dict[str, Any]:
    """Blend several independent demand drivers into one skill-weighted ensemble nowcast.

    Runs the single-driver forecast for each driver, weights each by its
    out-of-sample skill (equal weights if none beats naive persistence), and reports
    the ensemble YoY/level plus the dispersion across drivers (agreement_stdev) as a
    confidence signal. `drivers` defaults to google_trends,wikipedia,gdelt (+fred if
    the company has mapped macro series). A driver that fails is skipped with a note.
    """
    from .workflows.triangulate import triangulate as _triangulate

    ent, res = _triangulate(
        query, drivers=drivers, geo=geo, max_lag=max_lag, quarters=quarters,
        alpha=alpha, min_n=min_n, lag_by=lag_by, sign=sign,
    )
    return {"entity": _entity_dict(ent), "result": _jsonable(res)}


@mcp.tool()
def screen(
    tickers: list[str],
    drivers: list[str] | None = None,
    max_lag: int = 4,
    quarters: int = 16,
    min_n: int = 6,
    lag_by: str = "skill",
    sign: str = "any",
) -> dict[str, Any]:
    """Backtest a universe of tickers x drivers and rank by out-of-sample skill.

    Shows which (company, driver) pairs actually beat naive persistence, so you can
    see where alt-data signals work before investing in more connectors. `drivers`
    defaults to wikipedia,gdelt (Google Trends rate-limits in bulk, so it's opt-in).
    Rows with a skill score sort first (highest skill first); unresolved tickers or
    failed drivers carry an `error` field.
    """
    from .workflows.screen import screen as _screen

    tk = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tk:
        raise ValueError("no tickers given")
    rows = _screen(
        tk, drivers=drivers, max_lag=max_lag, quarters=quarters, min_n=min_n,
        lag_by=lag_by, sign=sign,
    )
    return {"rows": _jsonable(rows)}


@mcp.tool()
def multifactor(
    query: str,
    drivers: list[str] | None = None,
    seasonal: bool = False,
    geo: str = "US",
    max_lag: int = 4,
    quarters: int = 16,
    alpha: float = 0.20,
    min_n: int = 6,
    lag_by: str = "skill",
    sign: str = "any",
) -> dict[str, Any]:
    """Forecast revenue from one regression on several drivers at once (+ optional seasonality).

    Unlike triangulate (which averages separate single-driver forecasts), this fits
    a single multiple regression: revenueYoY ~ b0 + sum(b_i * driver_i_YoY[best_lag_i])
    (+ quarter-of-year dummies if seasonal=True). Each driver enters at its own best
    lag. Reports per-coefficient t/p stats and walk-forward skill. With few quarters
    this overfits easily — watch `result.warnings` and the n_obs vs feature count.
    """
    from .workflows.multifactor import run_multifactor

    ent, res = run_multifactor(
        query, drivers=drivers, seasonal=seasonal, geo=geo, max_lag=max_lag,
        quarters=quarters, alpha=alpha, min_n=min_n, lag_by=lag_by, sign=sign,
    )
    return {"entity": _entity_dict(ent), "result": _jsonable(res)}


def main() -> None:
    """Console-script entry point: serve over stdio."""
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
