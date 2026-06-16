"""Render a ForecastResult to a rich console view and a Markdown memo."""

from __future__ import annotations

import math
from datetime import date

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..features.align import quarter_of
from ..models import Entity, ForecastResult, TriangulationResult


def quarter_label(d: date | None) -> str:
    if d is None:
        return "n/a"
    return f"{d.year} Q{quarter_of(d)}"


def pct(x: float | None, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def money(x: float | None) -> str:
    if x is None:
        return "n/a"
    a = abs(x)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"${x / div:,.2f}{suf}"
    return f"${x:,.0f}"


def skill_cell(skill: float | None) -> str:
    return "n/a" if skill is None else f"{skill * 100:+.0f}%"


def weight_cell(weight: float) -> str:
    return f"{weight * 100:.0f}%" if weight else "-"


def _model_str(res: ForecastResult) -> str:
    return f"revenueYoY = {res.intercept:+.3f} {res.slope:+.3f} * driverYoY"


def _skill(res: ForecastResult) -> str:
    if res.backtest_mae_yoy is None or res.backtest_naive_mae_yoy is None:
        return "n/a"
    naive = res.backtest_naive_mae_yoy
    if naive <= 0:
        return "n/a"
    impr = (naive - res.backtest_mae_yoy) / naive
    verdict = "beats" if impr > 0 else "worse than"
    return f"{pct(impr)} ({verdict} naive)"


def _entity_head(entity: Entity) -> str:
    return (
        f"[bold]{entity.name or entity.query}[/]  "
        f"[cyan]{entity.ticker or ''}[/]  (CIK {entity.cik or 'n/a'})\n"
        f"Sector: {entity.sector or entity.sic_description or 'n/a'}"
        f"  |  SIC {entity.sic or 'n/a'}  |  FY-end {entity.fiscal_year_end or 'n/a'}\n"
        f"Seed terms: {', '.join(entity.seed_terms) or 'n/a'}"
    )


def render_console(entity: Entity, res: ForecastResult, console: Console | None = None) -> None:
    console = console or Console()
    console.print(Panel(_entity_head(entity), title="Entity", border_style="cyan", expand=False))

    console.print(
        f"[bold]KPI:[/] {res.kpi_source}:{res.kpi_metric}    "
        f"[bold]Driver:[/] {res.driver_label}    "
        f"[bold]n:[/] {res.n_obs} quarters"
    )

    if res.lag_table:
        t = Table(title="Lag scan (driver leads revenue YoY)", title_justify="left")
        t.add_column("lag (q)", justify="right")
        t.add_column("n", justify="right")
        t.add_column("corr r", justify="right")
        t.add_column("p-value", justify="right")
        t.add_column("skill (oos)", justify="right")
        for ls in res.lag_table:
            style = "bold green" if ls.lag == res.best_lag else ""
            r = "n/a" if ls.r != ls.r else f"{ls.r:+.3f}"
            sk = skill_cell(ls.skill)
            t.add_row(str(ls.lag), str(ls.n), r, f"{ls.p_value:.3f}", sk, style=style)
        console.print(t)

    fit = Table(show_header=False, box=None)
    fit.add_row("Best lag", f"{res.best_lag} quarter(s)")
    fit.add_row("Correlation r", f"{res.corr:+.3f}  (p={res.corr_p:.3f})")
    fit.add_row("R^2", f"{res.r2:.3f}")
    fit.add_row("Regression", _model_str(res))
    fit.add_row("Backtest MAE", f"{pct(res.backtest_mae_yoy, signed=False)} over {res.backtest_n} folds")
    fit.add_row("  vs naive MAE", f"{pct(res.backtest_naive_mae_yoy, signed=False)}  |  skill: {_skill(res)}")
    console.print(Panel(fit, title="Fit & backtest", border_style="blue", expand=False))

    fc = Table(show_header=False, box=None)
    fc.add_row("Target quarter", quarter_label(res.target_period))
    fc.add_row("Driver YoY (lagged input)", pct(res.current_driver_yoy))
    fc.add_row("Predicted revenue YoY", f"[bold]{pct(res.predicted_yoy)}[/]")
    if res.predicted_level is not None:
        ci = f"  [{money(res.pi_low_level)} ... {money(res.pi_high_level)}]"
        fc.add_row(
            "Predicted revenue",
            f"[bold]{money(res.predicted_level)}[/]{ci}  ({int((1 - res.alpha) * 100)}% PI)",
        )
        fc.add_row("(year-ago base)", money(res.base_level))
    border = "green" if res.predicted_yoy is not None else "yellow"
    console.print(Panel(fc, title="Forecast", border_style=border, expand=False))

    for n in res.notes:
        console.print(f"[dim]- {n}[/]")
    for w in res.warnings:
        console.print(f"[yellow]! {w}[/]")


def build_markdown(entity: Entity, res: ForecastResult) -> str:
    lines: list[str] = []
    A = lines.append
    A(f"# Signal forecast — {entity.name or entity.query} ({entity.ticker or ''})\n")
    A(f"- **CIK:** {entity.cik or 'n/a'}")
    A(f"- **Sector:** {entity.sector or entity.sic_description or 'n/a'} (SIC {entity.sic or 'n/a'})")
    A(f"- **Fiscal year end:** {entity.fiscal_year_end or 'n/a'}")
    A(f"- **Seed terms:** {', '.join(entity.seed_terms) or 'n/a'}")
    A(f"- **KPI:** `{res.kpi_source}:{res.kpi_metric}`")
    A(f"- **Driver:** {res.driver_label}")
    A(f"- **Aligned quarters:** {res.n_obs}\n")

    A("## Lag scan\n")
    A("| lag (q) | n | corr r | p-value | skill (oos) | |")
    A("|--:|--:|--:|--:|--:|:--|")
    for ls in res.lag_table:
        r = "n/a" if ls.r != ls.r else f"{ls.r:+.3f}"
        sk = skill_cell(ls.skill)
        A(f"| {ls.lag} | {ls.n} | {r} | {ls.p_value:.3f} | {sk} | {'**best**' if ls.lag == res.best_lag else ''} |")
    A("")

    A("## Fit & backtest\n")
    A(f"- **Best lag:** {res.best_lag} quarter(s)")
    A(f"- **Correlation r:** {res.corr:+.3f} (p = {res.corr_p:.3f})")
    A(f"- **R²:** {res.r2:.3f}")
    A(f"- **Model:** `{_model_str(res)}`")
    A(f"- **Backtest MAE (YoY):** {pct(res.backtest_mae_yoy, signed=False)} over {res.backtest_n} folds")
    A(f"- **Naive MAE (YoY):** {pct(res.backtest_naive_mae_yoy, signed=False)} — skill: {_skill(res)}\n")

    A("## Forecast\n")
    A(f"- **Target quarter:** {quarter_label(res.target_period)}")
    A(f"- **Driver YoY (lagged input):** {pct(res.current_driver_yoy)}")
    A(f"- **Predicted revenue YoY:** {pct(res.predicted_yoy)}")
    if res.predicted_level is not None:
        A(
            f"- **Predicted revenue:** {money(res.predicted_level)} "
            f"({int((1 - res.alpha) * 100)}% PI {money(res.pi_low_level)} … {money(res.pi_high_level)})"
        )
        A(f"- **Year-ago base:** {money(res.base_level)}")
    A("")

    if res.notes:
        A("## Notes\n")
        for n in res.notes:
            A(f"- {n}")
        A("")
    if res.warnings:
        A("## Caveats\n")
        for w in res.warnings:
            A(f"- ⚠ {w}")
        A("")
    A("---")
    A(
        "_Signals are estimates from public data, not investment advice. "
        "Google Trends is a relative index; correlation is not causation; small samples are noisy._"
    )
    return "\n".join(lines)


def render_triangulation(
    entity: Entity, res: TriangulationResult, console: Console | None = None
) -> None:
    console = console or Console()
    console.print(Panel(_entity_head(entity), title="Entity", border_style="cyan", expand=False))

    t = Table(title="Driver triangulation", title_justify="left")
    for col, justify in (
        ("driver", "left"), ("n", "right"), ("lag", "right"), ("corr r", "right"),
        ("skill (oos)", "right"), ("pred YoY", "right"), ("weight", "right"),
    ):
        t.add_column(col, justify=justify)
    for d in res.drivers:
        sk = skill_cell(d.skill)
        w = weight_cell(d.weight)
        style = "dim" if d.predicted_yoy is None else ""
        t.add_row(
            d.label, str(d.n), str(d.lag), f"{d.corr:+.3f}", sk, pct(d.predicted_yoy), w, style=style
        )
    console.print(t)

    e = Table(show_header=False, box=None)
    e.add_row("Target quarter", quarter_label(res.target_period))
    e.add_row("Ensemble revenue YoY", f"[bold]{pct(res.ensemble_yoy)}[/]")
    if res.ensemble_level is not None:
        e.add_row("Ensemble revenue", f"[bold]{money(res.ensemble_level)}[/]")
        e.add_row("(year-ago base)", money(res.base_level))
    if res.agreement_stdev is not None:
        e.add_row("Driver spread (sd)", pct(res.agreement_stdev, signed=False))
    console.print(Panel(e, title="Ensemble nowcast", border_style="green", expand=False))

    for n in res.notes:
        console.print(f"[dim]- {n}[/]")
    for w in res.warnings:
        console.print(f"[yellow]! {w}[/]")


def build_triangulation_markdown(entity: Entity, res: TriangulationResult) -> str:
    lines: list[str] = []
    A = lines.append
    A(f"# Triangulated nowcast — {entity.name or entity.query} ({entity.ticker or ''})\n")
    A(f"- **CIK:** {entity.cik or 'n/a'} · **Sector:** {entity.sector or 'n/a'}")
    A(f"- **Target quarter:** {quarter_label(res.target_period)}\n")
    A("## Drivers\n")
    A("| driver | n | lag | corr r | skill (oos) | pred YoY | weight |")
    A("|:--|--:|--:|--:|--:|--:|--:|")
    for d in res.drivers:
        sk = skill_cell(d.skill)
        w = weight_cell(d.weight)
        A(f"| {d.label} | {d.n} | {d.lag} | {d.corr:+.3f} | {sk} | {pct(d.predicted_yoy)} | {w} |")
    A("")
    A("## Ensemble\n")
    A(f"- **Ensemble revenue YoY:** {pct(res.ensemble_yoy)}")
    if res.ensemble_level is not None:
        A(f"- **Ensemble revenue:** {money(res.ensemble_level)} (year-ago base {money(res.base_level)})")
    if res.agreement_stdev is not None:
        A(f"- **Driver spread (sd):** {pct(res.agreement_stdev, signed=False)}")
    A("")
    for header, items in (("Notes", res.notes), ("Caveats", res.warnings)):
        if items:
            A(f"## {header}\n")
            for it in items:
                A(f"- {it}")
            A("")
    A("---")
    A("_Ensemble of independent public-data signals, weighted by out-of-sample skill. Not investment advice._")
    return "\n".join(lines)
