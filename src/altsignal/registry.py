"""Connector registry. Sources register themselves via the @register decorator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .connectors.base import Connector
    from .models import Entity, Signal

CONNECTORS: dict[str, type["Connector"]] = {}


def register(cls: type["Connector"]) -> type["Connector"]:
    if not getattr(cls, "source", ""):
        raise ValueError(f"{cls.__name__} must set a non-empty `source`")
    CONNECTORS[cls.source] = cls
    return cls


def _ensure_loaded() -> None:
    # Importing the package runs each module, triggering @register.
    import altsignal.connectors  # noqa: F401


def get_connector(source: str, store=None, settings=None) -> "Connector":
    _ensure_loaded()
    cls = CONNECTORS.get(source)
    if cls is None:
        raise KeyError(f"unknown source {source!r}; known: {', '.join(sorted(CONNECTORS))}")
    return cls(store=store, settings=settings)


def all_connectors() -> dict[str, type["Connector"]]:
    _ensure_loaded()
    return dict(sorted(CONNECTORS.items()))


def fetch_entity_signal(
    source: str,
    entity: "Entity",
    *,
    store=None,
    settings=None,
    term: str | None = None,
    page: str | None = None,
    board: str | None = None,
    geo: str = "US",
    quarters: int = 16,
) -> list["Signal"]:
    """Fetch raw signals for a resolved entity from one connector, applying the
    per-source argument conventions (seed-term / short_name / macro fallbacks).

    Single source of truth for the ``signal`` CLI command and the MCP
    ``get_signal`` tool, so the two presentation layers can never drift apart.
    """
    conn = get_connector(source, store=store, settings=settings)
    if source == "edgar":
        return conn.fetch(cik=entity.cik, query=entity.query, metric="revenue")
    if source == "google_trends":
        t = term or (entity.seed_terms[0] if entity.seed_terms else entity.short_name)
        return conn.fetch(term=t, geo=geo, quarters=quarters)
    if source == "wikipedia":
        return conn.fetch(page=page or entity.name or entity.short_name)
    if source == "fred":
        return conn.fetch(series_id=term)
    if source == "gdelt":
        return conn.fetch(query=term or entity.short_name, quarters=quarters)
    if source == "reddit":
        return conn.fetch(query=term or entity.short_name)
    if source == "greenhouse":
        return conn.fetch(board=board or term)
    return conn.fetch(term=term, page=page, query=entity.query)
