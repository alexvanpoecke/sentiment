"""Reddit mention-volume connector (official OAuth API).

Client-credentials OAuth, then a search for the query, bucketing the returned
submissions into monthly mention counts. NOTE: the official search returns only
a recent window of results (Pushshift-style deep history is no longer public),
so this is a recent-buzz signal, not a multi-year driver — useful for nowcasting
once accumulated via scheduled runs. Requires REDDIT_CLIENT_ID/SECRET.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from ..models import Observation, Signal
from ..registry import register
from .base import Connector, ConnectorError

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
SEARCH_URL = "https://oauth.reddit.com/search"


@register
class RedditConnector(Connector):
    source = "reddit"
    title = "Reddit mentions & sentiment"
    free = True
    requires = ("reddit_client_id", "reddit_client_secret")
    min_interval = 1.0
    note = "Recent mention volume (official API; history limited)."

    def _user_agent(self) -> str:
        return self.settings.reddit_user_agent or self.settings.sec_user_agent

    @staticmethod
    def _bucket_monthly(children: list[dict]) -> dict[date, int]:
        """Count submissions per calendar month from a search listing's children."""
        buckets: dict[date, int] = {}
        for child in children:
            data = child.get("data", {})
            ts = data.get("created_utc")
            if ts is None:
                continue
            try:
                d = datetime.fromtimestamp(float(ts), tz=timezone.utc).date().replace(day=1)
            except (ValueError, OverflowError, OSError):
                continue
            buckets[d] = buckets.get(d, 0) + 1
        return dict(sorted(buckets.items()))

    def _token(self) -> str:
        resp = self.http().request(
            "POST",
            TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(self.settings.reddit_client_id, self.settings.reddit_client_secret),
        )
        if resp.status_code != 200:
            raise ConnectorError(
                f"reddit token HTTP {resp.status_code}", source=self.source, url=TOKEN_URL,
                status=resp.status_code,
            )
        body = resp.json()
        token = body.get("access_token")
        if not token:  # Reddit returns 200 + {"error": ...} for OAuth failures
            raise ConnectorError(
                f"reddit token response had no access_token (error: {body.get('error', 'unknown')})",
                source=self.source, url=TOKEN_URL,
            )
        return token

    def mentions(self, query: str, *, limit: int = 100) -> Signal:
        key = f"reddit:mentions:{query}:{limit}"

        def _fetch() -> tuple[bytes, str]:
            token = self._token()
            resp = self.http().request(
                "GET",
                SEARCH_URL,
                params={"q": query, "sort": "new", "limit": str(limit), "type": "link", "t": "all"},
                headers={"Authorization": f"bearer {token}"},
            )
            if resp.status_code != 200:
                raise ConnectorError(
                    f"reddit search HTTP {resp.status_code}", source=self.source, url=SEARCH_URL,
                    status=resp.status_code,
                )
            return resp.content, "application/json"

        obj, _ = self.store.get_or_fetch_json(key, self.settings.cache_ttl, _fetch)
        children = obj.get("data", {}).get("children", [])
        obs = [Observation(ts=d, value=float(c)) for d, c in self._bucket_monthly(children).items()]
        if not obs:
            raise ConnectorError(
                f"reddit: no submissions found for {query!r}", source=self.source, url=SEARCH_URL
            )
        return Signal(
            entity_key=f"reddit:{query}",
            source=self.source,
            metric="mentions",
            freq="M",
            unit="posts",
            observations=obs,
            meta={"query": query, "n_posts": len(children)},
        )

    def fetch(self, *, query: str | None = None, term: str | None = None, **_):
        q = query or term
        if not q:
            raise ValueError("reddit.fetch needs `query` (or `term`)")
        if not self.available():
            raise RuntimeError(
                "Reddit connector unavailable: set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET"
            )
        return [self.mentions(q)]
