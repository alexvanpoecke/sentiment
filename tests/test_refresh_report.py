"""Refresh + report workflows, exercised with stubbed resolution/fetching (no network)."""

from datetime import date

import pytest

from altsignal.models import Entity, Observation, Signal
from altsignal.store import Store
from altsignal.workflows import refresh as refresh_mod


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "panel.sqlite")
    yield s
    s.close()


def _rev(ticker):
    return Signal(
        entity_key=ticker, source="edgar", metric="revenue", freq="Q",
        observations=[Observation(ts=date(2025, 3, 31), value=1000.0),
                      Observation(ts=date(2025, 6, 30), value=1100.0)],
    )


def _driver(ticker, source):
    return Signal(
        entity_key=ticker, source=source, metric="pageviews", freq="M",
        observations=[Observation(ts=date(2025, 5, 31), value=42.0)],
    )


@pytest.fixture
def stub_fetching(monkeypatch):
    """Replace resolution + driver loading with deterministic, network-free fakes."""
    def fake_resolve_and_revenue(ticker, *, quarters=16, store=None, settings=None):
        if ticker == "BAD":
            raise RuntimeError("not a SEC filer")
        ent = Entity(query=ticker, ticker=ticker, cik="0000000001", name=f"{ticker} Inc")
        return ent, _rev(ticker)

    def fake_load_driver(entity, driver, term, page, geo, quarters, store, settings):
        if driver == "explode":
            raise RuntimeError("driver boom")
        return _driver(entity.ticker, driver), f"{driver} label"

    monkeypatch.setattr(refresh_mod, "resolve_and_revenue", fake_resolve_and_revenue)
    monkeypatch.setattr(refresh_mod, "_load_driver", fake_load_driver)
    # Default: empty watchlist so tests don't depend on configs/watchlist.toml.
    monkeypatch.setattr(refresh_mod, "load_watchlist", lambda settings=None: {})


def test_refresh_records_revenue_and_drivers(store, stub_fetching):
    res = refresh_mod.refresh(
        ["WGO", "THO"], drivers=["wikipedia", "gdelt"],
        captured_at=date(2025, 7, 1), store=store, settings=object(),
    )
    # 2 tickers x (1 revenue + 2 drivers) = 6 successful captures, no failures.
    assert res.n_ok == 6 and res.n_failed == 0
    assert res.n_obs == 2 * (2 + 1 + 1)  # revenue has 2 obs, each driver 1 obs

    summary = {(r["source"], r["metric"]) for r in store.panel_summary("WGO")}
    assert ("edgar", "revenue") in summary
    assert ("wikipedia", "pageviews") in summary
    assert ("gdelt", "pageviews") in summary


def test_refresh_isolates_failures(store, stub_fetching):
    res = refresh_mod.refresh(
        ["WGO", "BAD"], drivers=["wikipedia", "explode"],
        captured_at=date(2025, 7, 1), store=store, settings=object(),
    )
    errs = [c for c in res.captures if c.error]
    # BAD fails to resolve (1) and WGO's 'explode' driver fails (1).
    assert res.n_failed == 2
    assert any(c.entity_key == "BAD" for c in errs)
    assert any(c.source == "explode" for c in errs)
    # WGO's revenue + wikipedia still captured despite the bad driver.
    assert res.n_ok == 2


def test_refresh_empty_watchlist_warns(store, monkeypatch, stub_fetching):
    monkeypatch.setattr(refresh_mod, "load_watchlist", lambda settings=None: {})
    res = refresh_mod.refresh([], captured_at=date(2025, 7, 1), store=store, settings=object())
    assert res.captures == [] and res.warnings


def test_refresh_reads_watchlist_when_no_tickers(store, monkeypatch, stub_fetching):
    monkeypatch.setattr(
        refresh_mod, "load_watchlist",
        lambda settings=None: {"tickers": ["LCII"], "drivers": ["wikipedia"]},
    )
    res = refresh_mod.refresh(None, captured_at=date(2025, 7, 1), store=store, settings=object())
    assert res.tickers == ["LCII"]
    assert res.n_ok == 2  # revenue + 1 driver


def test_dossier_markdown_composes_sections():
    from altsignal.models import MultiFactorResult, TriangulationResult
    from altsignal.reports.render import build_dossier_markdown

    ent = Entity(query="WGO", ticker="WGO", name="Winnebago", cik="0001")
    tri = TriangulationResult(entity_key="WGO", target_period=date(2025, 9, 30), ensemble_yoy=0.05)
    mf = MultiFactorResult(entity_key="WGO", predicted_yoy=0.04, r2=0.6)
    panel_rows = [
        {"entity_key": "WGO", "source": "gdelt", "metric": "tone", "geo": "",
         "n_obs": 8, "n_vintages": 3, "first_ts": "2024-03-31", "last_ts": "2025-06-30",
         "first_capture": "2025-05-01", "last_capture": "2025-07-01"},
    ]
    md = build_dossier_markdown(ent, tri, mf, panel_rows)
    assert "# Signal dossier — Winnebago (WGO)" in md
    assert "Triangulated nowcast" in md
    assert "Multifactor regression" in md
    assert "Point-in-time panel coverage" in md
    assert "2025-05-01 → 2025-07-01" in md  # capture window rendered


def test_dossier_handles_empty_panel():
    from altsignal.models import MultiFactorResult, TriangulationResult
    from altsignal.reports.render import build_dossier_markdown

    ent = Entity(query="NEW", ticker="NEW", name="NewCo")
    md = build_dossier_markdown(ent, TriangulationResult(entity_key="NEW"),
                                MultiFactorResult(entity_key="NEW"), [])
    assert "No panel history yet" in md
    assert "altsignal refresh NEW" in md
