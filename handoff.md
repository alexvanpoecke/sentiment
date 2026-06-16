# altsignal — Handoff

> Read this first if you're a future session (human or Claude) picking up this project.
> Companion to `README.md` (which is usage/architecture). This doc = **original ask +
> what's built + what to build next + gotchas.**

---

## 0. START HERE (next-session quick orientation)

1. Project = **altsignal**: turn readily-available public data into alternative-data signals
   for any public company, and forecast a KPI from them. Generic — never hard-coded to one
   company or signal.
2. **Phase 1 is DONE and verified live** (see §3). It's a Python CLI/library.
3. Dev setup (Windows PowerShell):
   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\python -m pip install -e ".[dev]"
   .\.venv\Scripts\python -m pytest -q            # 20 tests, all passing
   .\.venv\Scripts\altsignal.exe forecast WGO --term "buy an RV" --max-lag 4
   ```
4. Highest-value next steps are in §5. The two best first moves: **(a) pick the lag by
   out-of-sample backtest skill, not in-sample |r|** (current default picks spurious lags),
   and **(b) a triangulation workflow** that blends multiple signals.
5. Env gotcha that will bite you: this machine MITM-proxies TLS — Python HTTPS needs the OS
   cert store via `truststore` (already wired). See §6.

---

## 1. What the user originally asked for

> "Build a scraper tool with **many workflows** for creating **non-proprietary data signals**
> from readily available information (e.g. Google Trends). Example: *forecast WGO revenue next
> quarter — pull Google Trends for 'buy an RV', do a 12-quarter correlation to YoY revenue
> growth, lag the data appropriately, then pull current Trends and forecast the upcoming
> quarter.* For a given company also pull, where relevant: **public shipping/imports,
> competitor filings for triangulation, SimilarWeb traffic, FRED macro, (if autos) dealer
> inventory & listings, job postings, Reddit/forum sentiment, etc.** Think downstream and far.
> **Don't build for the specific example — build generally.**"

The screenshot was a Claude conversation ("Afternoon, Brett") doing exactly that revenue
forecast for WGO. The point of altsignal is to make that kind of analysis **repeatable and
general** across companies and signal types.

### Decisions the user made (don't re-litigate)
| Question | Choice |
|---|---|
| Primary interface | **CLI / library only** for now. MCP server deferred to Phase 4. |
| Where to start | **Scaffold + one working end-to-end slice**, then broaden. |
| Data posture | **Free / official APIs first.** ToS-sensitive scrapers deferred / opt-in. |
| Stack | Python (3.14 installed). **Pure-Python core on purpose**; numpy/pandas/statsmodels are optional extras. |

The user said they'll **build the rest later** — this handoff is so they (or a future session)
can resume cleanly.

---

## 2. Architecture (the layers)

```
ticker ─► entities/resolver ─► Entity{cik, sic, sector, seed_terms, peers, macro_series, ...}
                                   │   (the "think downstream" routing brain)
                                   ▼
        registry ─► connectors/* ─► Signal[]  (normalized time series + provenance)
                                   │
                                   ▼
                       store (SQLite raw-response cache + signal store, point-in-time vintages)
                                   │
                                   ▼
   features/{align,transforms,lag,stats} ─► quarterly YoY series, best lag, OLS+PI
                                   │
                                   ▼
        workflows/forecast ─► ForecastResult ─► reports/render ─► console + Markdown memo
                                                          ▲
                                                       cli.py (typer)
```

**Extending it = implement one interface.** A new data source subclasses `Connector`
(`connectors/base.py`), decorates with `@register`, gets added to the import line in
`connectors/__init__.py`, and is routed per SIC in `configs/industries.toml`. A new analysis
is a function in `workflows/` + a command in `cli.py`.

---

## 3. What's built (Phase 1) — file by file

### Core
| File | Purpose |
|---|---|
| `src/altsignal/config.py` | `Settings` from env + `.env`; paths; SEC User-Agent; cache TTL. |
| `src/altsignal/models.py` | Dataclasses: `Entity`, `Signal`, `Observation`, `ForecastResult`, `LagStat`. No 3rd-party deps. |
| `src/altsignal/store.py` | SQLite `raw_cache` (get_or_fetch_bytes/json) + normalized signal store. Vintage-aware. |
| `src/altsignal/http.py` | `HttpClient`: retries+backoff (honors Retry-After), per-host rate limiting, **OS-trust TLS via truststore**. |
| `src/altsignal/registry.py` | `@register` decorator + `get_connector`/`all_connectors`. |
| `src/altsignal/cli.py` | `typer` CLI: `sources`, `resolve`, `signal`, `forecast`. Console script = `altsignal`. |

### Connectors (`src/altsignal/connectors/`)
| Source | File | Status |
|---|---|---|
| `edgar` | `edgar.py` | **REAL.** Ticker→CIK resolve, submissions (SIC/fiscal-year), **quarterly revenue from XBRL** (handles cumulative-YTD *and* discrete filers via start-group differencing + latest-filing dedup). |
| `google_trends` | `trends.py` | **REAL.** explore→multiline flow, XSSI-prefix strip, cookie bootstrap, cached, raises `TrendsUnavailable` on 429 so callers can fall back. |
| `wikipedia` | `wikipedia.py` | **REAL.** Monthly pageviews (free, reliable). Good fallback driver. |
| `fred` | `fred.py` | **REAL but needs `FRED_API_KEY`.** Macro series observations. |
| `reddit` | `reddit.py` | **STUB (Phase 2).** Registered; `fetch()` raises NotImplemented. |
| `gdelt` | `gdelt.py` | **STUB (Phase 2).** Registered; free, no key when built. |

### Routing brain
- `src/altsignal/entities/resolver.py` — `resolve(query)` → rich `Entity`. Matches SIC to a
  bucket in `configs/industries.toml` (exact SIC, then 2-digit major-group fallback),
  substitutes `{brand}`/`{ticker}` into seed terms, and lists relevant connectors/macro
  series/subreddits.
- `configs/industries.toml` — routing table for RV/powersports, autos, auto-dealers, software,
  retail/e-commerce, restaurants, airlines, semis, homebuilders + defaults. **This is where you
  teach it new industries.**

### Features (pure-Python, `src/altsignal/features/`)
- `stats.py` — Pearson (+p), simple **OLS with prediction intervals**, Student-t CDF/PPF via the
  regularized incomplete beta. (Swap for `statsmodels` later via the `[ml]` extra.)
- `transforms.py` — `yoy`, `qoq`, `diff`, `rolling_mean`, `zscore`.
- `align.py` — calendar-quarter snapping, `add_quarters`/`prev_quarter`, `to_quarterly`, `align`.
- `lag.py` — `scan_lags` (cross-correlation lag search: driver leads target), `lagged_pairs`.

### Workflow + reporting
- `src/altsignal/workflows/forecast.py` — `run_forecast(query, ...)` orchestrates
  resolve → EDGAR revenue → driver (trends/wikipedia/fred/CSV) → align+YoY → lag scan → OLS →
  forecast next quarter + PI → walk-forward backtest vs naive persistence.
- `src/altsignal/reports/render.py` — `render_console` (ASCII-safe Rich) + `build_markdown`
  (UTF-8 memo written to `reports_out/`).

### Tests (`tests/`, 20 passing)
`test_stats.py`, `test_transforms.py`, `test_align.py`, `test_lag.py`, `test_edgar_periods.py`
(the last covers the cumulative-YTD vs discrete-quarter derivation + restatement dedup). All
**offline** — no network.

### Verified live (this session, 2026-06-15)
- `resolve WGO` → Winnebago, CIK 0000107687, SIC 3716 (Motor Homes), sector "RV / Towables /
  Powersports", auto-derived seed term **"buy an RV"**, FY-end 0830.
- `forecast WGO --term "buy an RV"` (Google Trends) and `--driver wikipedia` both ran on real
  data, 16 quarters. EDGAR returned 30 clean quarters (latest 2026-02-28 = $657.4M).
- **Honest result:** both drivers' max-|r| lag (lag 4) **failed the backtest** (skill −66% to
  −118% vs naive). The tool reports this rather than hiding it — see §4.1.

---

## 4. Known limitations / honest caveats (fix these as you go)

1. **Lag selection is naive.** `scan_lags` picks max |r| among lags with n ≥ `min_n`. For WGO
   this selected a spurious lag-4 (even a *negative* one for Trends) that loses to persistence
   out-of-sample. The lag table is fully transparent, but the *chosen* lag should be picked by
   **backtest skill and/or sign**, and the scan p-values want a **multiple-testing correction**.
   → highest-value fix; see §5.
2. **Single-regressor OLS only.** No seasonality, no multiple drivers, no regularization. That's
   the reason for the `[ml]` extra (statsmodels/scikit-learn) behind the feature interface.
3. **Google Trends is unofficial & relative.** Rate-limits (429), values are a 0-100 index
   normalized within the window (we use YoY-of-index). Cached + Wikipedia fallback in place.
   Consider the official Trends/Glimpse API or `pytrends` swap if it gets blocked.
4. **EDGAR `revenue` only.** Other metrics/segments not exposed yet. Derivation handles the
   common filer styles but mid-year fiscal changes, non-USD, or odd concepts may slip.
5. **Calendar vs fiscal alignment.** Fiscal quarters are snapped to the *containing calendar
   quarter*; for off-calendar FYEs (e.g. WGO's August) the driver-averaging window is slightly
   offset. Acceptable for v1; note it before trusting tight numbers.
6. **Signal-store persistence unused.** `store.save_signal/load_signal` exist but the workflow
   runs in-memory off the raw HTTP cache. Wire it up if you want a queryable panel.

---

## 5. What to build next (prioritized)

### A. Quick, high-value refinements (do first)
- [ ] **Lag selection by out-of-sample skill** — in `workflows/forecast.py`, backtest each
      candidate lag and choose the one that beats naive persistence (and/or prefer the expected
      sign). Add `--lag-by {corr,skill}`. Touches `features/lag.py` + `forecast.py`.
- [ ] **Triangulation workflow** — new `workflows/triangulate.py`: run several drivers
      (Trends + Wikipedia + a FRED macro), produce an **ensemble nowcast**, weight by historical
      skill, and show agreement/divergence. New CLI command `altsignal triangulate TICKER`.

### B. Phase 2 — free breadth (more connectors, same interface)
- [ ] `fred.py` — already real; just set `FRED_API_KEY` and wire macro series into triangulation
      (each `Entity` already carries `macro_series`).
- [ ] `reddit.py` — OAuth (or PRAW), pull seed-term mentions across `entity.subreddits`, score
      sentiment, emit weekly mention-volume + net-sentiment Signal.
- [ ] `gdelt.py` — GDELT 2.0 DOC API `timelinevol`/`tone` (free, no key).
- [ ] **`jobs.py`** (new) — Greenhouse/Lever public board JSON endpoints = clean hiring signal.
- [ ] **traffic** — Cloudflare Radar (free-ish) or app-store rank scraping as a SimilarWeb-free
      web-demand proxy.
- [ ] **Entity enrichment** — LLM-assisted seed terms / peers / domains; peer discovery by
      scanning same-SIC filers.

### C. Phase 3 — industry packs + premium/opt-in (needs keys/budget)
- [ ] Auto dealer inventory & listings (Marketcheck) for the autos/RV buckets.
- [ ] Imports / bill-of-lading (ImportYeti/Panjiva) — the "public shipping" leg.
- [ ] SimilarWeb traffic.
- [ ] Workflows: **beat/miss pre-announcement screen** (signal trajectory vs consensus),
      **cross-sectional ranking** across a peer universe.

### D. Phase 4 — orchestration
- [ ] **MCP server** exposing every connector + workflow as a tool, so Claude can drive it
      conversationally (turns the original screenshot into real tool calls). The `[mcp]` extra is
      reserved in `pyproject.toml`.
- [ ] **Scheduled refresh** to accumulate a real point-in-time panel (Trends is relative &
      current-only, so snapshotting over time builds genuine history).

---

## 6. Environment & gotchas

- **Python 3.14.5** (Microsoft Store build) + `py` launcher + `git`. No `uv`/`node`.
- **Corporate TLS proxy**: this machine intercepts TLS with a custom root CA that's in the
  Windows cert store but NOT in `certifi`. Symptom: `SSL: CERTIFICATE_VERIFY_FAILED`. Fix is
  already in `http.py` via **`truststore`** (OS cert store). Any new HTTP code must go through
  `HttpClient` (or otherwise use truststore). **Never disable verification.**
- **SEC requires a contact UA.** `.env` sets `ALTSIGNAL_CONTACT_EMAIL`
  (alexander.vanpoecke@readyx.com). `.env` is gitignored.
- **Cache**: `data/altsignal.sqlite` (gitignored). Delete it to force re-fetch. Memos land in
  `reports_out/` (gitignored).
- **Console output is ASCII-safe** on purpose (legacy Windows codepage mojibakes `·`/`²`/`—`);
  the saved Markdown keeps full UTF-8 formatting.

---

## 7. Command reference

```powershell
altsignal sources                                   # connectors + availability
altsignal resolve WGO                               # ticker -> CIK/SIC/sector/seed-terms
altsignal signal WGO --source edgar --limit 8       # raw signal dump
altsignal signal NKE --source google_trends --term "nike shoes"
altsignal forecast WGO --term "buy an RV" --driver google_trends --max-lag 4 --quarters 16
altsignal forecast TSLA --driver wikipedia --page "Tesla, Inc."
altsignal forecast WGO --driver csv --driver-csv mydata.csv   # bring-your-own driver (date,value)
```
Run via `.\.venv\Scripts\altsignal.exe ...` if the venv isn't activated.
