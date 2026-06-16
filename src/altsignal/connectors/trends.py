"""Google Trends connector.

Replicates the public web flow: bootstrap cookies -> /api/explore (get the
TIMESERIES widget token) -> /api/widgetdata/multiline (the actual series).
Google rate-limits aggressively and may return HTTP 429; on failure this
raises TrendsUnavailable so callers can degrade to another driver. Note that
Trends values are a *relative* 0-100 index normalized within the requested
window, so we work in YoY-of-the-index downstream.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import date, datetime, timezone

from ..features.align import lookback_start
from ..models import Observation, Signal
from ..registry import register
from .base import Connector, ConnectorError

HOME = "https://trends.google.com/?geo={geo}"
EXPLORE = "https://trends.google.com/trends/api/explore"
MULTILINE = "https://trends.google.com/trends/api/widgetdata/multiline"


class TrendsUnavailable(ConnectorError):
    pass


def _strip_xssi(text: str) -> str:
    # Responses are prefixed with )]}' to defeat JSON hijacking.
    idx = text.find("{")
    if idx == -1:
        raise TrendsUnavailable("unexpected Trends response (no JSON body)")
    return text[idx:]


@register
class GoogleTrendsConnector(Connector):
    source = "google_trends"
    title = "Google Trends (search interest)"
    free = True
    min_interval = 1.0  # be gentle
    note = "Unofficial endpoint; relative 0-100 index; may rate-limit (HTTP 429)."

    def _user_agent(self) -> str:
        return self.settings.browser_user_agent

    def _extra_headers(self) -> dict[str, str]:
        return {"Accept-Language": "en-US,en;q=0.9", "Referer": "https://trends.google.com/"}

    def interest_over_time(
        self,
        term: str,
        *,
        geo: str = "US",
        start: date | None = None,
        end: date | None = None,
    ) -> Signal:
        end = end or datetime.now(timezone.utc).date()
        start = start or lookback_start(end, 24)
        timeframe = f"{start.isoformat()} {end.isoformat()}"
        key = f"trends:{geo}:{term}:{timeframe}"

        def _fetch() -> tuple[bytes, str]:
            # Reuse the connector's pooled client (cookies persist across the 3 calls);
            # nullcontext avoids closing the shared client after the request.
            with nullcontext(self.http()) as c:
                # 1) bootstrap cookies
                c.get(HOME.format(geo=geo))
                # 2) explore -> widget token + request
                req = json.dumps(
                    {
                        "comparisonItem": [{"keyword": term, "geo": geo, "time": timeframe}],
                        "category": 0,
                        "property": "",
                    }
                )
                er = c.get(EXPLORE, params={"hl": "en-US", "tz": "0", "req": req})
                if er.status_code != 200:
                    raise TrendsUnavailable(f"explore HTTP {er.status_code}")
                widgets = json.loads(_strip_xssi(er.text)).get("widgets", [])
                ts_widget = next((w for w in widgets if w.get("id") == "TIMESERIES"), None)
                if not ts_widget:
                    raise TrendsUnavailable("no TIMESERIES widget in explore response")
                # 3) multiline -> the series
                mr = c.get(
                    MULTILINE,
                    params={
                        "hl": "en-US",
                        "tz": "0",
                        "req": json.dumps(ts_widget["request"]),
                        "token": ts_widget["token"],
                    },
                )
                if mr.status_code != 200:
                    raise TrendsUnavailable(f"multiline HTTP {mr.status_code}")
                # Strip the )]}' XSSI guard so the generic JSON cache can parse it.
                return _strip_xssi(mr.text).encode("utf-8"), "application/json"

        obj, _ = self.store.get_or_fetch_json(key, self.settings.cache_ttl, _fetch)
        timeline = obj.get("default", {}).get("timelineData", [])
        obs: list[Observation] = []
        for pt in timeline:
            # Skip malformed points instead of letting a bare KeyError/ValueError
            # escape — callers rely on TrendsUnavailable to degrade to another driver.
            try:
                if not pt.get("value"):
                    continue
                ts = datetime.fromtimestamp(int(pt["time"]), tz=timezone.utc).date()
                obs.append(Observation(ts=ts, value=float(pt["value"][0])))
            except (KeyError, ValueError, TypeError, IndexError):
                continue
        if not obs:
            raise TrendsUnavailable(f"no timeline data returned for {term!r}")
        return Signal(
            entity_key=f"trends:{geo}:{term}",
            source=self.source,
            metric="search_interest",
            freq="W",  # typically weekly for multi-year ranges; aggregated to Q downstream
            geo=geo,
            unit="index(0-100)",
            observations=obs,
            meta={"term": term, "geo": geo, "timeframe": timeframe},
        )

    def fetch(self, *, term: str | None = None, geo: str = "US", quarters: int = 16, **_):
        if not term:
            raise ValueError("google_trends.fetch needs `term`")
        end = datetime.now(timezone.utc).date()
        start = lookback_start(end, quarters + 8)
        return [self.interest_over_time(term, geo=geo, start=start, end=end)]
