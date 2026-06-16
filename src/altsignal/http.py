"""Shared HTTP client: sane defaults, per-host rate limiting, retries with backoff.

Hand-rolled retry loop (instead of leaning on a decorator) so we can honor
``Retry-After`` and still hand the final response back to the caller for inspection.
"""

from __future__ import annotations

import logging
import ssl
import threading
import time

import httpx

log = logging.getLogger("altsignal.http")

RETRY_STATUS = {429, 500, 502, 503, 504}

# Verify TLS against the OS trust store (picks up corporate root CAs that
# certifi's bundle lacks). Falls back to httpx's default if truststore is absent.
try:  # pragma: no cover - environment dependent
    import truststore

    _SSL_CONTEXT: ssl.SSLContext | bool = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
except Exception:  # pragma: no cover
    _SSL_CONTEXT = True

_last_request: dict[str, float] = {}
_rate_lock = threading.Lock()


def _respect_rate_limit(host: str, min_interval: float) -> None:
    if min_interval <= 0:
        return
    with _rate_lock:
        now = time.monotonic()
        wait = min_interval - (now - _last_request.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        _last_request[host] = time.monotonic()


class HttpError(RuntimeError):
    """Raised when a request ultimately fails after retries."""


class HttpClient:
    def __init__(
        self,
        *,
        user_agent: str,
        headers: dict[str, str] | None = None,
        min_interval: float = 0.5,
        timeout: float = 30.0,
        max_attempts: int = 4,
    ):
        base_headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        if headers:
            base_headers.update(headers)
        self._client = httpx.Client(
            headers=base_headers, timeout=timeout, follow_redirects=True, verify=_SSL_CONTEXT
        )
        self.min_interval = min_interval
        self.max_attempts = max_attempts

    # context manager sugar
    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @property
    def cookies(self) -> httpx.Cookies:
        return self._client.cookies

    def get(self, url: str, params=None, headers=None) -> httpx.Response:
        return self.request("GET", url, params=params, headers=headers)

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        host = httpx.URL(url).host or ""
        last_exc: Exception | None = None
        last_resp: httpx.Response | None = None
        for attempt in range(1, self.max_attempts + 1):
            _respect_rate_limit(host, self.min_interval)
            try:
                resp = self._client.request(method, url, **kwargs)
            except httpx.TransportError as exc:
                last_exc = exc
                log.debug("%s %s transport error: %s (attempt %d)", method, url, exc, attempt)
                if attempt < self.max_attempts:
                    self._backoff(attempt)
                continue

            if resp.status_code in RETRY_STATUS:
                last_resp = resp
                log.debug("%s %s -> %d (attempt %d)", method, url, resp.status_code, attempt)
                if attempt < self.max_attempts:
                    self._backoff(attempt, resp)
                continue
            return resp

        if last_resp is not None:
            # Exhausted retries on a retryable status: hand it back so the caller
            # can decide (e.g. degrade gracefully) rather than always exploding.
            return last_resp
        raise HttpError(f"{method} {url} failed after {self.max_attempts} attempts: {last_exc}")

    def _backoff(self, attempt: int, resp: httpx.Response | None = None) -> None:
        delay = min(2.0**attempt, 30.0)
        if resp is not None:
            ra = resp.headers.get("Retry-After", "")
            if ra.isdigit():
                delay = min(float(ra), 60.0)
        time.sleep(delay)
