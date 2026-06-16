"""GDELT news-volume / tone connector (GDELT 2.0 DOC API).

Free, no key. Returns a historical time series of how intensely the global news
media covers a query — a real "attention" driver that plugs straight into the
forecast and triangulation workflows. ``timelinevol`` is coverage *intensity*
(% of all monitored articles), which is normalized over time and so behaves like
a relative index (good for YoY); ``timelinetone`` is average sentiment.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from ..features.align import fetch_window_start, lookback_start
from ..models import Observation, Signal
from ..registry import register
from .base import Connector, ConnectorError

DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_MODES = {"timelinevol", "timelinevolraw", "timelinetone"}


@register
class GdeltConnector(Connector):
    source = "gdelt"
    title = "GDELT news volume & tone"
    free = True
    min_interval = 1.0  # be gentle with the public API
    note = "Free, no key; news coverage intensity since 2017."

    def timeline(
        self,
        query: str,
        *,
        mode: str = "timelinevol",
        start: date | None = None,
        end: date | None = None,
    ) -> Signal:
        if mode not in _MODES:
            raise ValueError(f"gdelt mode must be one of {sorted(_MODES)}, not {mode!r}")
        end = end or datetime.now(timezone.utc).date()
        start = start or lookback_start(end, 24)
        # GDELT expects a phrase query to be quoted.
        q = f'"{query}"' if " " in query else query
        params = {
            "query": q,
            "mode": mode,
            "format": "json",
            "startdatetime": start.strftime("%Y%m%d000000"),
            "enddatetime": end.strftime("%Y%m%d000000"),
        }
        key = f"gdelt:{mode}:{query}:{start:%Y%m%d}:{end:%Y%m%d}"
        obj = self.get_json(key, DOC_URL, params=params)

        if not isinstance(obj, dict):
            raise ConnectorError(
                f"gdelt: unexpected (non-object) response for {query!r}",
                source=self.source, url=DOC_URL,
            )
        timeline = obj.get("timeline") or []
        series = timeline[0] if timeline and isinstance(timeline[0], dict) else {}
        obs: list[Observation] = []
        for pt in series.get("data", []):
            digits = "".join(ch for ch in str(pt.get("date", "")) if ch.isdigit())[:8]
            if len(digits) != 8:
                continue
            try:
                d = date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
                obs.append(Observation(ts=d, value=float(pt["value"])))
            except (ValueError, KeyError, TypeError):
                continue
        if not obs:
            raise ConnectorError(
                f"gdelt: no timeline data for {query!r}", source=self.source, url=DOC_URL
            )
        unit = "tone" if mode == "timelinetone" else "intensity(%)"
        metric = "news_tone" if mode == "timelinetone" else "news_volume"
        return Signal(
            entity_key=f"gdelt:{query}",
            source=self.source,
            metric=metric,
            freq="D",  # daily points; aggregated to quarters downstream
            unit=unit,
            observations=obs,
            meta={"query": query, "mode": mode},
        )

    def fetch(
        self,
        *,
        query: str | None = None,
        term: str | None = None,
        mode: str = "timelinevol",
        quarters: int = 16,
        **_,
    ):
        q = query or term
        if not q:
            raise ValueError("gdelt.fetch needs `query` (or `term`)")
        end = datetime.now(timezone.utc).date()
        start = fetch_window_start(end, quarters)
        return [self.timeline(q, mode=mode, start=start, end=end)]
