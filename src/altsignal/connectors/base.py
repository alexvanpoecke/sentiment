"""Base class every data-source connector implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..http import HttpClient
from ..models import Signal


class ConnectorError(RuntimeError):
    """An upstream fetch failed (e.g. a non-OK HTTP status)."""

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        url: str | None = None,
        status: int | None = None,
    ):
        self.source, self.url, self.status = source, url, status
        super().__init__(message)


class Connector(ABC):
    #: unique source id, e.g. "edgar" (set by subclass)
    source: str = ""
    #: human-friendly name for `altsignal sources`
    title: str = ""
    #: settings attributes that must be truthy for this connector to work
    requires: tuple[str, ...] = ()
    #: is this a free / officially-sanctioned source?
    free: bool = True
    #: default per-host min interval (seconds) between requests
    min_interval: float = 0.5
    #: short note for `altsignal sources`
    note: str = ""

    def __init__(self, store=None, settings=None):
        from ..config import get_settings
        from ..store import get_store

        self.settings = settings or get_settings()
        self.store = store if store is not None else get_store()
        self._http: HttpClient | None = None

    # -- availability --------------------------------------------------------
    def available(self) -> bool:
        return all(getattr(self.settings, attr, None) for attr in self.requires)

    def availability_note(self) -> str:
        if self.available():
            return "ready"
        missing = [a for a in self.requires if not getattr(self.settings, a, None)]
        return "needs: " + ", ".join(missing) if missing else "unavailable"

    # -- http + cache helpers ------------------------------------------------
    def _user_agent(self) -> str:
        """User-Agent for this connector's requests (override per source)."""
        return self.settings.sec_user_agent

    def _extra_headers(self) -> dict[str, str] | None:
        """Extra default headers for this connector (override per source)."""
        return None

    def http(self) -> HttpClient:
        """One reusable, connection-pooling client per connector instance."""
        if self._http is None:
            self._http = HttpClient(
                user_agent=self._user_agent(),
                headers=self._extra_headers(),
                min_interval=self.min_interval,
            )
        return self._http

    def get_json(self, key: str, url: str, *, ttl: int | None = None, params: dict | None = None):
        """Fetch + cache JSON with uniform status handling; returns the parsed object."""
        ttl = self.settings.cache_ttl if ttl is None else ttl

        def _fetch() -> tuple[bytes, str]:
            resp = self.http().get(url, params=params)
            if resp.status_code != 200:
                raise ConnectorError(
                    f"{self.source}: {url} returned HTTP {resp.status_code}",
                    source=self.source,
                    url=url,
                    status=resp.status_code,
                )
            return resp.content, resp.headers.get("content-type", "application/json")

        obj, _ = self.store.get_or_fetch_json(key, ttl, _fetch)
        return obj

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def __enter__(self) -> "Connector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- the one method connectors must implement ----------------------------
    @abstractmethod
    def fetch(self, **kwargs) -> list[Signal]:
        """Return a list of normalized Signals. See each connector for kwargs."""
        raise NotImplementedError
