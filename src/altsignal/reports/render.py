"""Render a ForecastResult to a rich console view and a Markdown memo."""

from __future__ import annotations

import math
from datetime import date

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..features.align import quarter_of
from ..models import (
    Entity,
    ForecastResult,
    MultiFactorResult,
    ScreenRow,
    TriangulationResult,
)


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
    if (
        res.backtest_mae_yoy is None
        or res.backtest_naive_mae_yoy is None
        or res.backtest_naive_mae_yoy <= 0
    ):
        return "n/a"
    skill = (res.backtest_naive_mae_yoy - res.backtest_mae_yoy) / res.backtest_naive_mae_yoy
    return _skill_verdict(skill)


def _print_footnotes(console: Console, res) -> None:
    """Print a result's dim notes then yellow warnings (shared by the renderers)."""
    for n in res.notes:
        console.print(f"[dim]- {n}[/]")
    for w in res.warnings:
        console.print(f"[yellow]! {w}[/]")


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

    _print_footnotes(console, res)


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

    _print_footnotes(console, res)


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


def render_screen(rows: list[ScreenRow], console: Console | None = None) -> None:
    console = console or Console()
    scored = [r for r in rows if not r.error]
    errors = [r for r in rows if r.error]
    # Slim console table (names/notes live in the Markdown memo) so it fits a terminal.
    t = Table(
        title="Cross-sectional skill screen (ranked by out-of-sample skill)", title_justify="left"
    )
    for col, justify in (
        ("ticker", "left"), ("driver", "left"), ("n", "right"), ("lag", "right"),
        ("corr r", "right"), ("skill (oos)", "right"), ("pred YoY", "right"), ("target", "right"),
    ):
        t.add_column(col, justify=justify)
    for r in scored:
        style = "bold green" if (r.skill or 0) > 0 else ""
        corr = "n/a" if r.corr != r.corr else f"{r.corr:+.3f}"
        t.add_row(
            r.ticker, r.driver or "", str(r.n), str(r.lag), corr,
            skill_cell(r.skill), pct(r.predicted_yoy), quarter_label(r.target_period), style=style,
        )
    console.print(t)
    n_pos = sum(1 for r in scored if (r.skill or 0) > 0)
    console.print(
        f"[dim]{n_pos}/{len(scored)} (company, driver) pairs beat naive persistence out-of-sample.[/]"
    )
    for r in errors:
        console.print(f"[dim]skipped {r.ticker}/{r.driver or '?'}: {r.error}[/]")


def build_screen_markdown(rows: list[ScreenRow]) -> str:
    lines: list[str] = []
    A = lines.append
    A("# Cross-sectional skill screen\n")
    A("Ranked by out-of-sample skill vs naive persistence (positive = beats naive).\n")
    A("| ticker | name | driver | n | lag | corr r | skill (oos) | pred YoY | target | note |")
    A("|:--|:--|:--|--:|--:|--:|--:|--:|--:|:--|")
    for r in rows:
        if r.error:
            A(f"| {r.ticker} | {r.name or ''} | {r.driver or ''} | | | | | | | {r.error} |")
            continue
        corr = "n/a" if r.corr != r.corr else f"{r.corr:+.3f}"
        A(
            f"| {r.ticker} | {r.name or ''} | {r.driver} | {r.n} | {r.lag} | {corr} | "
            f"{skill_cell(r.skill)} | {pct(r.predicted_yoy)} | {quarter_label(r.target_period)} | |"
        )
    A("")
    scored = [r for r in rows if not r.error]
    n_pos = sum(1 for r in scored if (r.skill or 0) > 0)
    A(f"_{n_pos}/{len(scored)} (company, driver) pairs beat naive persistence out-of-sample._")
    return "\n".join(lines)


def _skill_verdict(skill: float | None) -> str:
    if skill is None:
        return "n/a"
    return f"{skill * 100:+.0f}% ({'beats' if skill > 0 else 'worse than'} naive)"


def render_multifactor(
    entity: Entity, res: MultiFactorResult, console: Console | None = None
) -> None:
    console = console or Console()
    console.print(Panel(_entity_head(entity), title="Entity", border_style="cyan", expand=False))
    console.print(
        f"[bold]KPI:[/] {res.kpi_source}:{res.kpi_metric}    "
        f"[bold]drivers:[/] {', '.join(res.driver_labels) or 'n/a'}    "
        f"[bold]seasonal:[/] {res.seasonal}    [bold]n:[/] {res.n_obs}"
    )
    if res.features:
        t = Table(title="Coefficients", title_justify="left")
        t.add_column("factor")
        t.add_column("coef", justify="right")
        t.add_column("t", justify="right")
        t.add_column("p", justify="right")
        for f in res.features:
            t.add_row(f.name, f"{f.coef:+.3f}", f"{f.t:+.2f}", f"{f.p:.3f}")
        console.print(t)

    fit = Table(show_header=False, box=None)
    fit.add_row("R^2", f"{res.r2:.3f}")
    fit.add_row("Backtest MAE", f"{pct(res.backtest_mae, signed=False)} over {res.backtest_n} folds")
    fit.add_row("  vs naive MAE", f"{pct(res.backtest_naive_mae, signed=False)}  |  skill: {_skill_verdict(res.skill)}")
    console.print(Panel(fit, title="Fit & backtest", border_style="blue", expand=False))

    fc = Table(show_header=False, box=None)
    fc.add_row("Target quarter", quarter_label(res.target_period))
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

    _print_footnotes(console, res)


def build_multifactor_markdown(entity: Entity, res: MultiFactorResult) -> str:
    lines: list[str] = []
    A = lines.append
    A(f"# Multifactor forecast — {entity.name or entity.query} ({entity.ticker or ''})\n")
    season = " + quarter seasonality" if res.seasonal else ""
    A(f"- **CIK:** {entity.cik or 'n/a'} · **Drivers:** {', '.join(res.driver_labels) or 'n/a'}{season}")
    A(f"- **Aligned quarters:** {res.n_obs} · **R²:** {res.r2:.3f}\n")
    A("## Coefficients\n")
    A("| factor | coef | t | p |")
    A("|:--|--:|--:|--:|")
    for f in res.features:
        A(f"| {f.name} | {f.coef:+.3f} | {f.t:+.2f} | {f.p:.3f} |")
    A("")
    A("## Backtest\n")
    A(f"- **Backtest MAE (YoY):** {pct(res.backtest_mae, signed=False)} over {res.backtest_n} folds")
    A(f"- **Naive MAE (YoY):** {pct(res.backtest_naive_mae, signed=False)} — skill: {_skill_verdict(res.skill)}\n")
    A("## Forecast\n")
    A(f"- **Target quarter:** {quarter_label(res.target_period)}")
    A(f"- **Predicted revenue YoY:** {pct(res.predicted_yoy)}")
    if res.predicted_level is not None:
        A(
            f"- **Predicted revenue:** {money(res.predicted_level)} "
            f"({int((1 - res.alpha) * 100)}% PI {money(res.pi_low_level)} … {money(res.pi_high_level)})"
        )
        A(f"- **Year-ago base:** {money(res.base_level)}")
    A("")
    if res.warnings:
        A("## Caveats\n")
        for w in res.warnings:
            A(f"- ! {w}")
        A("")
    A("---")
    A("_Multiple regression on public-data signals; small samples overfit. Not investment advice._")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Combined dossier: one report blending triangulation + multifactor + panel    #
# --------------------------------------------------------------------------- #
def _consensus_verdict(tri: TriangulationResult, mf: MultiFactorResult) -> str:
    """One-line read combining the ensemble and the regression."""
    vals = [v for v in (tri.ensemble_yoy, mf.predicted_yoy) if v is not None]
    if not vals:
        return "No forecast could be produced from the available signals."
    direction = "growth" if sum(vals) / len(vals) >= 0 else "decline"
    agree = "agree" if all((v >= 0) == (vals[0] >= 0) for v in vals) else "disagree on direction"
    return (
        f"Ensemble {pct(tri.ensemble_yoy)} and multifactor {pct(mf.predicted_yoy)} "
        f"YoY — the two methods {agree} ({direction})."
    )


def build_dossier_markdown(
    entity: Entity,
    tri: TriangulationResult,
    mf: MultiFactorResult,
    panel_rows: list[dict],
) -> str:
    """Compose a single company dossier from the triangulation and multifactor
    results plus the company's point-in-time panel coverage."""
    lines: list[str] = []
    A = lines.append
    A(f"# Signal dossier — {entity.name or entity.query} ({entity.ticker or ''})\n")
    A(f"- **CIK:** {entity.cik or 'n/a'} · **Sector:** {entity.sector or entity.sic_description or 'n/a'}"
      f" (SIC {entity.sic or 'n/a'})")
    A(f"- **Fiscal year end:** {entity.fiscal_year_end or 'n/a'} · **Seed terms:** "
      f"{', '.join(entity.seed_terms) or 'n/a'}")
    A(f"- **Target quarter:** {quarter_label(tri.target_period or mf.target_period)}\n")

    A("## Bottom line\n")
    A(f"> {_consensus_verdict(tri, mf)}\n")

    # --- ensemble nowcast (triangulation) ---
    A("## Triangulated nowcast\n")
    A("Independent demand signals, each weighted by out-of-sample skill.\n")
    A("| driver | n | lag | corr r | skill (oos) | pred YoY | weight |")
    A("|:--|--:|--:|--:|--:|--:|--:|")
    for d in tri.drivers:
        A(f"| {d.label} | {d.n} | {d.lag} | {d.corr:+.3f} | {skill_cell(d.skill)} | "
          f"{pct(d.predicted_yoy)} | {weight_cell(d.weight)} |")
    A("")
    A(f"- **Ensemble revenue YoY:** {pct(tri.ensemble_yoy)}")
    if tri.ensemble_level is not None:
        A(f"- **Ensemble revenue:** {money(tri.ensemble_level)} (year-ago base {money(tri.base_level)})")
    if tri.agreement_stdev is not None:
        A(f"- **Driver spread (sd):** {pct(tri.agreement_stdev, signed=False)}")
    A("")

    # --- combined regression (multifactor) ---
    A("## Multifactor regression\n")
    season = " + quarter seasonality" if mf.seasonal else ""
    A(f"Single regression on {', '.join(mf.driver_labels) or 'n/a'}{season} "
      f"(n={mf.n_obs}, R²={mf.r2:.3f}).\n")
    if mf.features:
        A("| factor | coef | t | p |")
        A("|:--|--:|--:|--:|")
        for f in mf.features:
            A(f"| {f.name} | {f.coef:+.3f} | {f.t:+.2f} | {f.p:.3f} |")
        A("")
    A(f"- **Predicted revenue YoY:** {pct(mf.predicted_yoy)} — skill: {_skill_verdict(mf.skill)}")
    if mf.predicted_level is not None:
        A(f"- **Predicted revenue:** {money(mf.predicted_level)} "
          f"({int((1 - mf.alpha) * 100)}% PI {money(mf.pi_low_level)} … {money(mf.pi_high_level)})")
    A("")

    # --- point-in-time panel coverage ---
    A("## Point-in-time panel coverage\n")
    if panel_rows:
        A("Vintages captured by scheduled `altsignal refresh` runs (more vintages = "
          "more honest backtests).\n")
        A("| source | metric | geo | periods | vintages | capture window |")
        A("|:--|:--|:--|--:|--:|:--|")
        for r in panel_rows:
            window = (
                f"{r['first_capture']} → {r['last_capture']}"
                if r["first_capture"] != r["last_capture"]
                else r["first_capture"]
            )
            A(f"| {r['source']} | {r['metric']} | {r['geo'] or '—'} | {r['n_obs']} | "
              f"{r['n_vintages']} | {window} |")
        A("")
    else:
        A("_No panel history yet for this company. Run `altsignal refresh "
          f"{entity.ticker or entity.query}` (ideally on a schedule) to start building one._\n")

    # --- caveats (deduped across both analyses) ---
    seen: set[str] = set()
    caveats = [w for w in (*tri.warnings, *mf.warnings) if not (w in seen or seen.add(w))]
    if caveats:
        A("## Caveats\n")
        for w in caveats:
            A(f"- ⚠ {w}")
        A("")

    A("---")
    A("_Dossier built from public-data signals (triangulation + multifactor regression). "
      "Correlation is not causation; small samples are noisy. Not investment advice._")
    return "\n".join(lines)
