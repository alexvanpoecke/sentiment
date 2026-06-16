"""Point-in-time panel store: vintage recording + look-ahead-free reconstruction."""

from datetime import date

import pytest

from altsignal.models import Observation, Signal
from altsignal.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "panel.sqlite")
    yield s
    s.close()


def _sig(source, metric, obs, geo=None):
    return Signal(
        entity_key="ignored",  # record_panel keys by the entity_key argument, not this
        source=source,
        metric=metric,
        geo=geo,
        observations=[Observation(ts=d, value=v) for d, v in obs],
    )


def test_record_and_summary(store):
    sig = _sig("wikipedia", "pageviews", [(date(2025, 3, 31), 100.0), (date(2025, 6, 30), 110.0)])
    assert store.record_panel("WGO", sig, date(2025, 7, 1)) == 2

    rows = store.panel_summary()
    assert len(rows) == 1
    r = rows[0]
    assert (r["entity_key"], r["source"], r["metric"]) == ("WGO", "wikipedia", "pageviews")
    assert r["n_obs"] == 2 and r["n_vintages"] == 1
    assert r["first_capture"] == r["last_capture"] == "2025-07-01"


def test_reconstruction_uses_latest_vintage_at_or_before_as_of(store):
    # Vintage 1 (captured 2025-07-01): Q2 reads 110.
    store.record_panel("WGO", _sig("gdelt", "tone", [(date(2025, 6, 30), 110.0)]), date(2025, 7, 1))
    # Vintage 2 (captured 2025-08-01): Q2 was revised up to 125 (backfill).
    store.record_panel("WGO", _sig("gdelt", "tone", [(date(2025, 6, 30), 125.0)]), date(2025, 8, 1))

    # As of the day before the revision, a backtest must see the ORIGINAL 110.
    early = store.load_panel_as_of("WGO", "gdelt", "tone", None, date(2025, 7, 15))
    assert early is not None and early.as_dict()[date(2025, 6, 30)] == 110.0

    # After the revision, it sees the corrected 125.
    late = store.load_panel_as_of("WGO", "gdelt", "tone", None, date(2025, 8, 15))
    assert late.as_dict()[date(2025, 6, 30)] == 125.0

    # Before any capture, nothing is known.
    assert store.load_panel_as_of("WGO", "gdelt", "tone", None, date(2025, 6, 1)) is None

    # Two vintages of one period -> two distinct captures, still one observed period.
    summ = store.panel_summary("WGO")[0]
    assert summ["n_obs"] == 1 and summ["n_vintages"] == 2


def test_reconstruction_mixes_periods_from_different_vintages(store):
    # Older period captured early; a newer period only appears in a later vintage.
    store.record_panel("THO", _sig("wikipedia", "pageviews", [(date(2025, 3, 31), 50.0)]), date(2025, 4, 1))
    store.record_panel(
        "THO",
        _sig("wikipedia", "pageviews", [(date(2025, 3, 31), 50.0), (date(2025, 6, 30), 60.0)]),
        date(2025, 7, 1),
    )
    # As of 2025-05-01 only Q1 was known.
    mid = store.load_panel_as_of("THO", "wikipedia", "pageviews", None, date(2025, 5, 1))
    assert list(mid.as_dict()) == [date(2025, 3, 31)]
    # As of 2025-07-15 both quarters are known.
    full = store.load_panel_as_of("THO", "wikipedia", "pageviews", None, date(2025, 7, 15))
    assert set(full.as_dict()) == {date(2025, 3, 31), date(2025, 6, 30)}


def test_same_day_rerun_is_idempotent(store):
    sig = _sig("edgar", "revenue", [(date(2025, 3, 31), 1000.0)])
    store.record_panel("WGO", sig, date(2025, 7, 1))
    # Re-run same day with a corrected value: overwrite, not duplicate.
    store.record_panel("WGO", _sig("edgar", "revenue", [(date(2025, 3, 31), 1010.0)]), date(2025, 7, 1))
    summ = store.panel_summary("WGO")[0]
    assert summ["n_vintages"] == 1
    asof = store.load_panel_as_of("WGO", "edgar", "revenue", None, date(2025, 7, 1))
    assert asof.as_dict()[date(2025, 3, 31)] == 1010.0


def test_as_of_filing_date_beats_capture_date(store):
    # Revenue became public on its filing date (2025-08-05) but we only captured it
    # on a later refresh (2025-09-01). A backtest "as of 2025-08-10" must SEE it,
    # because it was knowable from the filing date — not gated on when we ran refresh.
    sig = Signal(
        entity_key="WGO", source="edgar", metric="revenue",
        observations=[Observation(ts=date(2025, 6, 30), value=1000.0, as_of=date(2025, 8, 5))],
    )
    store.record_panel("WGO", sig, captured_at=date(2025, 9, 1))

    known = store.load_panel_as_of("WGO", "edgar", "revenue", None, date(2025, 8, 10))
    assert known is not None and known.as_dict()[date(2025, 6, 30)] == 1000.0
    # Before the filing date it was not yet public, even though it predates capture.
    assert store.load_panel_as_of("WGO", "edgar", "revenue", None, date(2025, 8, 1)) is None


def test_geo_is_distinguished(store):
    store.record_panel("WGO", _sig("google_trends", "interest", [(date(2025, 3, 31), 70.0)], geo="US"), date(2025, 7, 1))
    store.record_panel("WGO", _sig("google_trends", "interest", [(date(2025, 3, 31), 40.0)], geo="GB"), date(2025, 7, 1))
    us = store.load_panel_as_of("WGO", "google_trends", "interest", "US", date(2025, 7, 1))
    gb = store.load_panel_as_of("WGO", "google_trends", "interest", "GB", date(2025, 7, 1))
    assert us.as_dict()[date(2025, 3, 31)] == 70.0
    assert gb.as_dict()[date(2025, 3, 31)] == 40.0
    assert len(store.panel_summary("WGO")) == 2
