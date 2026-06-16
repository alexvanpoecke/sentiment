# altsignal

Build **non-proprietary alternative-data signals** for any public company from readily
available sources — Google Trends, SEC filings, web traffic, macro series, job postings,
shipping/imports, forum sentiment — and turn them into **forecastable features**.

The motivating use case: *"Forecast WGO revenue next quarter. Pull Google Trends for
'buy an RV', run a 12-quarter correlation to YoY revenue growth at the right lag, then
forecast the upcoming quarter."* — done generically, for any ticker and any relevant signal,
so you never hard-code one company or one signal.

---

## Why it's built this way

- **Generic, not WGO-specific.** Give it a ticker; it resolves the company, infers the
  sector from the SEC SIC code, and routes to the signals that matter for *that* business
  (cars → dealer/auto-search terms; software → web traffic + job postings; retail →
  shipping/imports). See `configs/industries.toml`.
- **Pluggable connectors.** Every data source implements one small `Connector` interface and
  registers itself. Adding a source is one file.
- **Pure-Python core, heavy stack optional.** The core (stats, OLS, correlation, prediction
  intervals) is implemented in the standard library so it installs instantly on any CPython
  (incl. 3.14) with **no compiled-wheel dependency**. `pandas` / `scikit-learn` / `statsmodels`
  are opt-in extras behind the feature interfaces, for when you want heavier models.
- **Reproducible & polite.** All raw HTTP responses are cached in SQLite with TTLs, so reruns
  are fast and don't hammer rate-limited endpoints. Point-in-time vintages are recorded to
  avoid look-ahead bias in backtests.
- **Free / official APIs first.** SEC EDGAR, Google Trends, Wikipedia pageviews, FRED, Reddit
  API, GDELT. ToS-sensitive scrapers are deferred / opt-in.

## Architecture

```
ticker ─► entities/resolver ─► Entity{cik, sic, sector, seed_terms, peers, fiscal_year_end}
                                   │
                                   ▼
        registry ─► connectors/* ─► Signal[] (normalized time series + provenance)
          edgar (KPI: revenue)         │
          google_trends (driver)       ▼
          wikipedia (driver)     store (SQLite raw-cache + signal store, vintages)
          fred/reddit/... (stub)       │
                                       ▼
        features/{align,transforms,lag,stats} ─► aligned YoY series, best lag, OLS
                                       │
                                       ▼
        workflows/forecast ─► ForecastResult{lag, corr, r2, forecast, PI, backtest}
                                       │
                                       ▼
        reports/render ─► console + Markdown memo      cli.py (typer)
```

## Install

Requires Python 3.11+ (uses stdlib `tomllib`). On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -e .
# optional heavier numerics / dev tooling:
# .\.venv\Scripts\python -m pip install -e ".[pandas,ml,dev]"
```

Copy `.env.example` to `.env` and set at least `ALTSIGNAL_CONTACT_EMAIL` (SEC requires a
contact in the User-Agent). FRED/Reddit keys are optional.

## Usage

```powershell
# What sources are wired up and which are available with current keys?
altsignal sources

# Resolve a company: ticker -> CIK, SIC, sector, seed terms, fiscal year end
altsignal resolve WGO

# Pull one raw signal
altsignal signal WGO --source edgar --metric revenue
altsignal signal WGO --source google_trends --term "buy an RV"

# The headline workflow: correlation + lag + forecast (reproduces the example)
altsignal forecast WGO --term "buy an RV" --driver google_trends --max-lag 4 --quarters 16
altsignal forecast WGO --driver wikipedia        # uses Wikipedia pageviews as the driver
```

Every `forecast` run also writes a Markdown research memo to `reports_out/`.

## Status / roadmap

**Phase 1 (this scaffold):** core framework + working end-to-end slice — EDGAR revenue,
Google Trends + Wikipedia drivers, lagged-correlation forecast with prediction intervals and
a walk-forward backtest, CLI, tests.

**Phase 2 (free breadth):** FRED, Reddit (PRAW), GDELT news, Greenhouse/Lever job boards,
Cloudflare Radar / app-store ranks; LLM-assisted entity enrichment (seed terms, peers).

**Phase 3 (industry packs + premium/opt-in):** auto dealer inventory (Marketcheck), imports /
bill-of-lading (ImportYeti), SimilarWeb traffic; triangulation, beat/miss pre-announcement
screen, cross-sectional ranking workflows.

**Phase 4 (orchestration):** MCP server so Claude can drive every connector/workflow as a
tool; scheduled signal refresh to build a real point-in-time panel; report builder.

## Legal / data posture

This tool collects **publicly available** information for financial research. It prioritizes
official APIs (SEC, FRED, Reddit, Wikipedia, GDELT) and respects robots.txt and rate limits.
Some sources (Google Trends, SimilarWeb, etc.) have terms restricting automated access — use
official/licensed access where required, and review each source's terms before relying on it.
Signals are estimates, not investment advice.
