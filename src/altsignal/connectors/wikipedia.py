"""Wikipedia pageviews connector (Wikimedia REST API).

Free, official, no key. A solid, reliable "attention" proxy and a good fallback
driver when Google Trends is rate-limited. Wikimedia asks for a descriptive
User-Agent with contact info (we use the SEC contact UA).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from urllib.parse import quote

from ..features.align import lookback_start
from ..models import Observation, Signal
from ..registry import register
from .base import Connector

REST = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "{project}/all-access/user/{article}/monthly/{start}/{end}"
)


@register
class WikipediaConnector(Connector):
    source = "wikipedia"
    title = "Wikipedia pageviews (attention proxy)"
    free = True
    min_interval = 0.2

    def pageviews(
        self,
        page: str,
        *,
        project: str = "en.wikipedia.org",
        start: date | None = None,
        end: date | None = None,
    ) -> Signal:
        end = end or datetime.now(timezone.utc).date()
        start = start or lookback_start(end, 24)
        article = quote(page.replace(" ", "_"), safe="")
        url = REST.format(
            project=project,
            article=article,
            start=start.strftime("%Y%m%d"),
            end=end.strftime("%Y%m%d"),
        )
        key = f"wiki:{project}:{page}:{start:%Y%m%d}:{end:%Y%m%d}"
        obj = self.get_json(key, url)
        obs: list[Observation] = []
        for item in obj.get("items", []):
            ts = datetime.strptime(item["timestamp"][:8], "%Y%m%d").date()
            obs.append(Observation(ts=ts, value=float(item["views"])))
        if not obs:
            raise RuntimeError(f"no pageview data for {page!r}")
        return Signal(
            entity_key=f"wiki:{project}:{page}",
            source=self.source,
            metric="pageviews",
            freq="M",
            unit="views",
            observations=obs,
            meta={"page": page, "project": project},
        )

    def fetch(
        self,
        *,
        page: str | None = None,
        project: str = "en.wikipedia.org",
        quarters: int = 16,
        **_,
    ):
        if not page:
            raise ValueError("wikipedia.fetch needs `page` (article title)")
        end = datetime.now(timezone.utc).date()
        start = lookback_start(end, quarters + 8)
        return [self.pageviews(page, project=project, start=start, end=end)]
