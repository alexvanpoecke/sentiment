"""Parsing/logic tests for the new connectors + triangulation weighting.

These avoid the network by exercising static parse helpers and monkeypatching
the cached fetch, so they run offline.
"""

from datetime import date

import pytest

from altsignal.connectors.base import ConnectorError
from altsignal.connectors.gdelt import GdeltConnector
from altsignal.connectors.jobs import GreenhouseConnector
from altsignal.connectors.reddit import RedditConnector
from altsignal.store import Store
from altsignal.workflows.triangulate import skill_weights


def _mem_store() -> Store:
    return Store(":memory:")


def test_gdelt_parses_timeline(monkeypatch):
    conn = GdeltConnector(store=_mem_store())
    monkeypatch.setattr(
        conn,
        "get_json",
        lambda *a, **k: {
            "timeline": [
                {
                    "series": "Volume Intensity",
                    "data": [
                        {"date": "20230101T000000Z", "value": 1.5},
                        {"date": "20230401000000", "value": 2.0},
                    ],
                }
            ]
        },
    )
    sig = conn.timeline("winnebago industries", start=date(2022, 1, 1), end=date(2023, 6, 1))
    assert sig.metric == "news_volume" and sig.unit == "intensity(%)"
    assert [o.ts for o in sig.observations] == [date(2023, 1, 1), date(2023, 4, 1)]
    assert sig.observations[1].value == 2.0


def test_gdelt_empty_timeline_raises(monkeypatch):
    conn = GdeltConnector(store=_mem_store())
    monkeypatch.setattr(conn, "get_json", lambda *a, **k: {})
    with pytest.raises(ConnectorError):
        conn.timeline("no results here", start=date(2022, 1, 1), end=date(2023, 1, 1))


def test_reddit_bucket_monthly():
    children = [
        {"data": {"created_utc": 1672531200}},  # 2023-01-01 UTC
        {"data": {"created_utc": 1675209600}},  # 2023-02-01 UTC
        {"data": {"created_utc": 1675900800}},  # 2023-02-09 UTC -> same Feb bucket
        {"data": {}},  # no timestamp -> skipped
    ]
    buckets = RedditConnector._bucket_monthly(children)
    assert buckets == {date(2023, 1, 1): 1, date(2023, 2, 1): 2}


def test_greenhouse_count_jobs():
    assert GreenhouseConnector._count_jobs({"jobs": [1, 2, 3]}) == 3
    assert GreenhouseConnector._count_jobs({"jobs": None}) == 0
    assert GreenhouseConnector._count_jobs({}) == 0


def test_skill_weights_proportional_and_fallback():
    w = skill_weights([0.5, 0.1, None])
    assert abs(sum(w) - 1.0) < 1e-9
    assert w[0] > w[1] > 0 and w[2] == 0.0  # None -> zero weight
    # no positive skill anywhere -> equal weights
    eq = skill_weights([-0.2, None, -0.5])
    assert all(abs(x - 1 / 3) < 1e-9 for x in eq)
    assert skill_weights([]) == []
