"""altsignal command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import REPORTS_DIR, get_settings
from .registry import all_connectors, get_connector

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Build non-proprietary alternative-data signals and forecast a company KPI.",
)
console = Console()


def _fail(msg: str) -> "typer.Exit":
    console.print(f"[red]error:[/] {msg}")
    return typer.Exit(code=1)


@app.command()
def sources() -> None:
    """List registered data connectors and whether they're usable right now."""
    settings = get_settings()
    t = Table(title="altsignal connectors")
    t.add_column("source", style="cyan")
    t.add_column("title")
    t.add_column("free")
    t.add_column("status")
    t.add_column("note", style="dim")
    for name, cls in all_connectors().items():
        inst = cls(settings=settings)
        status = "[green]ready[/]" if inst.available() else f"[yellow]{inst.availability_note()}[/]"
        t.add_row(name, cls.title, "yes" if cls.free else "paid", status, cls.note or "")
    console.print(t)


@app.command()
def resolve(query: str = typer.Argument(..., help="Ticker or company name")) -> None:
    """Resolve a company and show inferred sector, seed terms, and routed connectors."""
    from .entities.resolver import resolve as _resolve

    try:
        ent = _resolve(query)
    except Exception as e:  # noqa: BLE001
        raise _fail(str(e))
    t = Table(show_header=False, box=None)
    for k, v in [
        ("query", ent.query),
        ("ticker", ent.ticker),
        ("name", ent.name),
        ("cik", ent.cik),
        ("sic", f"{ent.sic} ({ent.sic_description})" if ent.sic else None),
        ("sector", ent.sector),
        ("fiscal year end", ent.fiscal_year_end),
        ("country", ent.country),
        ("seed terms", ", ".join(ent.seed_terms) or None),
        ("connectors", ", ".join(ent.connectors) or None),
        ("macro series", ", ".join(ent.macro_series) or None),
        ("subreddits", ", ".join(ent.subreddits) or None),
    ]:
        t.add_row(f"[bold]{k}[/]", str(v) if v is not None else "[dim]n/a[/]")
    console.print(t)


@app.command()
def signal(
    query: str = typer.Argument(..., help="Ticker or company name"),
    source: str = typer.Option("edgar", help="Connector source id (see `sources`)"),
    term: Optional[str] = typer.Option(None, help="Search term (google_trends) or FRED series id"),
    page: Optional[str] = typer.Option(None, help="Wikipedia article title"),
    board: Optional[str] = typer.Option(None, help="Greenhouse board token (source=greenhouse)"),
    geo: str = typer.Option("US"),
    quarters: int = typer.Option(16),
    limit: int = typer.Option(12, help="How many recent observations to print"),
) -> None:
    """Fetch and print one raw signal from a single connector."""
    from .entities.resolver import resolve as _resolve

    settings = get_settings()
    try:
        ent = _resolve(query)
        conn = get_connector(source, settings=settings)
        if source == "edgar":
            sigs = conn.fetch(cik=ent.cik, query=query, metric="revenue")
        elif source == "google_trends":
            sigs = conn.fetch(term=term or (ent.seed_terms[0] if ent.seed_terms else ent.short_name), geo=geo, quarters=quarters)
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
    except Exception as e:  # noqa: BLE001
        raise _fail(str(e))

    for sig in sigs:
        console.print(
            f"[bold cyan]{sig.label}[/]  ({sig.source}, freq={sig.freq}, unit={sig.unit or 'n/a'}, "
            f"n={len(sig)})  {sig.meta}"
        )
        t = Table()
        t.add_column("date")
        t.add_column("value", justify="right")
        for o in sig.sorted()[-limit:]:
            t.add_row(o.ts.isoformat(), f"{o.value:,.2f}")
        console.print(t)


@app.command()
def forecast(
    query: str = typer.Argument(..., help="Ticker or company name"),
    driver: str = typer.Option("google_trends", help="google_trends | wikipedia | fred | gdelt | csv"),
    term: Optional[str] = typer.Option(None, help="Search term / FRED series; defaults to top seed term"),
    page: Optional[str] = typer.Option(None, help="Wikipedia article (driver=wikipedia)"),
    geo: str = typer.Option("US"),
    max_lag: int = typer.Option(4, help="Max lead (quarters) to search"),
    quarters: int = typer.Option(16, help="History window (quarters) for the correlation"),
    alpha: float = typer.Option(0.20, help="Prediction interval level = 1 - alpha"),
    min_n: int = typer.Option(6, help="Min overlapping points for a lag to qualify"),
    lag_by: str = typer.Option("skill", help="Lag selection: 'skill' (out-of-sample) or 'corr' (max |r|)"),
    sign: str = typer.Option("any", help="Constrain lag to 'any' | 'positive' | 'negative' correlation"),
    driver_csv: Optional[str] = typer.Option(None, help="Use a 2-col date,value CSV as the driver"),
    fallback: bool = typer.Option(True, help="Fall back to Wikipedia if Google Trends is blocked"),
    out_dir: Optional[str] = typer.Option(None, help="Where to write the Markdown memo"),
) -> None:
    """Correlation + lag + forecast: the headline workflow, for any ticker/signal."""
    from .connectors.trends import TrendsUnavailable
    from .reports.render import build_markdown, render_console
    from .workflows.forecast import run_forecast

    kw = dict(
        term=term, page=page, geo=geo, max_lag=max_lag, quarters=quarters,
        alpha=alpha, min_n=min_n, lag_by=lag_by, sign=sign, driver_csv=driver_csv,
    )
    try:
        ent, res = run_forecast(query, driver=driver, **kw)
    except TrendsUnavailable as e:
        if fallback and driver == "google_trends" and not driver_csv:
            console.print(
                f"[yellow]Google Trends unavailable ({e}); falling back to Wikipedia pageviews.[/]"
            )
            try:
                ent, res = run_forecast(query, driver="wikipedia", **{**kw, "term": None})
            except Exception as e2:  # noqa: BLE001
                raise _fail(str(e2))
        else:
            raise _fail(str(e))
    except Exception as e:  # noqa: BLE001
        raise _fail(str(e))

    render_console(ent, res, console)

    out_path = Path(out_dir) if out_dir else REPORTS_DIR
    out_path.mkdir(parents=True, exist_ok=True)
    fname = f"{(ent.ticker or ent.query).upper()}_{res.driver_source}.md"
    md_file = out_path / fname
    md_file.write_text(build_markdown(ent, res), encoding="utf-8")
    console.print(f"\n[dim]Memo written to {md_file}[/]")


@app.command()
def triangulate(
    query: str = typer.Argument(..., help="Ticker or company name"),
    drivers: Optional[str] = typer.Option(
        None, help="Comma-separated drivers (default: google_trends,wikipedia,gdelt[,fred])"
    ),
    geo: str = typer.Option("US"),
    max_lag: int = typer.Option(4),
    quarters: int = typer.Option(16),
    alpha: float = typer.Option(0.20),
    min_n: int = typer.Option(6),
    lag_by: str = typer.Option("skill", help="Lag selection: 'skill' or 'corr'"),
    sign: str = typer.Option("any", help="Constrain lags to 'any' | 'positive' | 'negative'"),
    out_dir: Optional[str] = typer.Option(None, help="Where to write the Markdown memo"),
) -> None:
    """Blend multiple demand signals into one skill-weighted ensemble nowcast."""
    from .reports.render import build_triangulation_markdown, render_triangulation
    from .workflows.triangulate import triangulate as _triangulate

    drv = [d.strip() for d in drivers.split(",") if d.strip()] if drivers else None
    try:
        ent, res = _triangulate(
            query, drivers=drv, geo=geo, max_lag=max_lag, quarters=quarters,
            alpha=alpha, min_n=min_n, lag_by=lag_by, sign=sign,
        )
    except Exception as e:  # noqa: BLE001
        raise _fail(str(e))

    render_triangulation(ent, res, console)
    out_path = Path(out_dir) if out_dir else REPORTS_DIR
    out_path.mkdir(parents=True, exist_ok=True)
    md_file = out_path / f"{(ent.ticker or ent.query).upper()}_triangulation.md"
    md_file.write_text(build_triangulation_markdown(ent, res), encoding="utf-8")
    console.print(f"\n[dim]Memo written to {md_file}[/]")


@app.command()
def screen(
    tickers: str = typer.Argument(..., help="Comma-separated tickers, e.g. WGO,THO,LCII,PATK"),
    drivers: Optional[str] = typer.Option(
        None, help="Comma-separated drivers (default: wikipedia,gdelt; trends rate-limits in bulk)"
    ),
    max_lag: int = typer.Option(4),
    quarters: int = typer.Option(16),
    min_n: int = typer.Option(6),
    lag_by: str = typer.Option("skill"),
    sign: str = typer.Option("any"),
    out_dir: Optional[str] = typer.Option(None, help="Where to write the Markdown memo"),
) -> None:
    """Backtest a universe of tickers x drivers and rank by out-of-sample skill."""
    from .reports.render import build_screen_markdown, render_screen
    from .workflows.screen import screen as _screen

    tk = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not tk:
        raise _fail("no tickers given")
    drv = [d.strip() for d in drivers.split(",") if d.strip()] if drivers else None
    try:
        rows = _screen(
            tk, drivers=drv, max_lag=max_lag, quarters=quarters, min_n=min_n,
            lag_by=lag_by, sign=sign,
        )
    except Exception as e:  # noqa: BLE001
        raise _fail(str(e))

    render_screen(rows, console)
    out_path = Path(out_dir) if out_dir else REPORTS_DIR
    out_path.mkdir(parents=True, exist_ok=True)
    md_file = out_path / "screen.md"
    md_file.write_text(build_screen_markdown(rows), encoding="utf-8")
    console.print(f"\n[dim]Memo written to {md_file}[/]")


@app.command()
def multifactor(
    query: str = typer.Argument(..., help="Ticker or company name"),
    drivers: Optional[str] = typer.Option(
        None, help="Comma-separated drivers (default: google_trends,wikipedia,gdelt)"
    ),
    seasonal: bool = typer.Option(False, help="Add quarter-of-year dummies (seasonality)"),
    geo: str = typer.Option("US"),
    max_lag: int = typer.Option(4),
    quarters: int = typer.Option(16),
    alpha: float = typer.Option(0.20),
    min_n: int = typer.Option(6),
    lag_by: str = typer.Option("skill"),
    sign: str = typer.Option("any"),
    out_dir: Optional[str] = typer.Option(None, help="Where to write the Markdown memo"),
) -> None:
    """Combine several signals in one regression (multi-driver, optional seasonality)."""
    from .reports.render import build_multifactor_markdown, render_multifactor
    from .workflows.multifactor import run_multifactor

    drv = [d.strip() for d in drivers.split(",") if d.strip()] if drivers else None
    try:
        ent, res = run_multifactor(
            query, drivers=drv, seasonal=seasonal, geo=geo, max_lag=max_lag,
            quarters=quarters, alpha=alpha, min_n=min_n, lag_by=lag_by, sign=sign,
        )
    except Exception as e:  # noqa: BLE001
        raise _fail(str(e))

    render_multifactor(ent, res, console)
    out_path = Path(out_dir) if out_dir else REPORTS_DIR
    out_path.mkdir(parents=True, exist_ok=True)
    md_file = out_path / f"{(ent.ticker or ent.query).upper()}_multifactor.md"
    md_file.write_text(build_multifactor_markdown(ent, res), encoding="utf-8")
    console.print(f"\n[dim]Memo written to {md_file}[/]")


@app.command()
def refresh(
    tickers: Optional[str] = typer.Argument(
        None, help="Comma-separated tickers; defaults to configs/watchlist.toml"
    ),
    drivers: Optional[str] = typer.Option(
        None, help="Comma-separated drivers (default: wikipedia,gdelt; trends rate-limits in bulk)"
    ),
    geo: str = typer.Option("US"),
    quarters: int = typer.Option(16, help="History window (quarters) to capture per series"),
) -> None:
    """Capture revenue + drivers for a watchlist into the point-in-time panel.

    Run this on a schedule (cron / Task Scheduler) to accumulate a real as-of
    history so backtests use only what was knowable at each past date.
    """
    from .workflows.refresh import refresh as _refresh

    tk = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
    drv = [d.strip() for d in drivers.split(",") if d.strip()] if drivers else None
    try:
        res = _refresh(tk, drivers=drv, geo=geo, quarters=quarters)
    except Exception as e:  # noqa: BLE001
        raise _fail(str(e))

    t = Table(title=f"Panel refresh — captured {res.captured_at.isoformat()}", title_justify="left")
    for col, justify in (
        ("entity", "left"), ("source", "left"), ("metric", "left"),
        ("geo", "left"), ("obs", "right"), ("status", "left"),
    ):
        t.add_column(col, justify=justify)
    for c in res.captures:
        ok = c.error is None
        t.add_row(
            c.entity_key, c.source, c.metric, c.geo or "", str(c.n_obs) if ok else "-",
            "[green]ok[/]" if ok else f"[red]{c.error}[/]",
        )
    console.print(t)
    console.print(
        f"[dim]{res.n_ok} series captured ({res.n_obs} observations), "
        f"{res.n_failed} failed across {len(res.tickers)} ticker(s).[/]"
    )
    for w in res.warnings:
        console.print(f"[yellow]! {w}[/]")


@app.command()
def panel(
    entity: Optional[str] = typer.Argument(None, help="Filter to one entity key (ticker)"),
) -> None:
    """Show point-in-time panel coverage: periods and vintages captured so far."""
    from .store import get_store

    rows = get_store().panel_summary(entity.strip().upper() if entity else None)
    if not rows:
        console.print(
            "[yellow]Panel is empty.[/] Run [bold]altsignal refresh[/] to start capturing vintages."
        )
        return
    t = Table(title="Point-in-time panel coverage", title_justify="left")
    for col, justify in (
        ("entity", "left"), ("source", "left"), ("metric", "left"), ("geo", "left"),
        ("periods", "right"), ("vintages", "right"), ("capture window", "left"),
    ):
        t.add_column(col, justify=justify)
    for r in rows:
        window = (
            f"{r['first_capture']} → {r['last_capture']}"
            if r["first_capture"] != r["last_capture"]
            else r["first_capture"]
        )
        t.add_row(
            r["entity_key"], r["source"], r["metric"], r["geo"] or "",
            str(r["n_obs"]), str(r["n_vintages"]), window,
        )
    console.print(t)


@app.command()
def report(
    query: str = typer.Argument(..., help="Ticker or company name"),
    drivers: Optional[str] = typer.Option(
        None, help="Comma-separated drivers (default: google_trends,wikipedia,gdelt)"
    ),
    seasonal: bool = typer.Option(False, help="Add quarter-of-year dummies to the regression"),
    geo: str = typer.Option("US"),
    max_lag: int = typer.Option(4),
    quarters: int = typer.Option(16),
    alpha: float = typer.Option(0.20),
    min_n: int = typer.Option(6),
    lag_by: str = typer.Option("skill"),
    sign: str = typer.Option("any"),
    out_dir: Optional[str] = typer.Option(None, help="Where to write the Markdown dossier"),
) -> None:
    """Build one company dossier: triangulation + multifactor + panel coverage."""
    from .reports.render import (
        build_dossier_markdown,
        render_multifactor,
        render_triangulation,
    )
    from .workflows.report import build_report

    drv = [d.strip() for d in drivers.split(",") if d.strip()] if drivers else None
    try:
        ent, tri, mf, panel_rows = build_report(
            query, drivers=drv, seasonal=seasonal, geo=geo, max_lag=max_lag,
            quarters=quarters, alpha=alpha, min_n=min_n, lag_by=lag_by, sign=sign,
        )
    except Exception as e:  # noqa: BLE001
        raise _fail(str(e))

    render_triangulation(ent, tri, console)
    render_multifactor(ent, mf, console)
    console.print(
        f"[dim]Panel: {len(panel_rows)} series tracked for {ent.ticker or ent.query}.[/]"
    )

    out_path = Path(out_dir) if out_dir else REPORTS_DIR
    out_path.mkdir(parents=True, exist_ok=True)
    md_file = out_path / f"{(ent.ticker or ent.query).upper()}_dossier.md"
    md_file.write_text(build_dossier_markdown(ent, tri, mf, panel_rows), encoding="utf-8")
    console.print(f"\n[dim]Dossier written to {md_file}[/]")


if __name__ == "__main__":  # pragma: no cover
    app()
