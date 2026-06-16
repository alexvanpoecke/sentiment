"""Connector registry. Sources register themselves via the @register decorator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .connectors.base import Connector

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
