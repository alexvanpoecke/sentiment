"""FRED macro connector (Federal Reserve Bank of St. Louis).

Official API; free with a key (FRED_API_KEY). Useful for the macro leg of a
triangulation (consumer sentiment, disposable income, gas prices, housing, ...).
"""

from __future__ import annotations

from datetime import date

from ..features.align import fetch_window_start
from ..models import Observation, Signal
from ..registry import register
from .base import Connector

OBS_URL = "https://api.stlouisfed.org/fred/series/observations"


def _infer_freq(dates: list[date]) -> str:
    if len(dates) < 2:
        return "U"
    gaps = sorted((dates[i] - dates[i - 1]).days for i in range(1, len(dates)))
    med = gaps[len(gaps) // 2]
    if med <= 8:
        return "W"
    if med <= 35:
        return "M"
    if med <= 100:
        return "Q"
    return "A"


@register
class FredConnector(Connector):
    source = "fred"
    title = "FRED macro series"
    free = True
    requires = ("fred_api_key",)
    min_interval = 0.3

    def series(self, series_id: str, *, observation_start: str | None = None) -> Signal:
        params = {
            "series_id": series_id,
            "api_key": self.settings.fred_api_key,
            "file_type": "json",
        }
        if observation_start:
            params["observation_start"] = observation_start
        key = f"fred:{series_id}:{observation_start or 'all'}"
        obj = self.get_json(key, OBS_URL, params=params)
        obs: list[Observation] = []
        for row in obj.get("observations", []):
            if row.get("value") in (None, ".", ""):
                continue
            obs.append(Observation(ts=date.fromisoformat(row["date"]), value=float(row["value"])))
        if not obs:
            raise RuntimeError(f"no observations for FRED series {series_id}")
        return Signal(
            entity_key=f"macro:{series_id}",
            source=self.source,
            metric=series_id,
            freq=_infer_freq([o.ts for o in obs]),
            observations=obs,
            meta={"series_id": series_id},
        )

    def fetch(
        self,
        *,
        series_id: str | None = None,
        observation_start: str | None = None,
        quarters: int = 16,
        **_,
    ):
        if not series_id:
            raise ValueError("fred.fetch needs `series_id`")
        if not self.available():
            raise RuntimeError("FRED connector unavailable: set FRED_API_KEY")
        if observation_start is None:
            observation_start = fetch_window_start(date.today(), quarters).isoformat()
        return [self.series(series_id, observation_start=observation_start)]
