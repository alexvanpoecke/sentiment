"""Connectors package. Importing it registers every built-in source."""

from __future__ import annotations

# Import order doesn't matter; each module self-registers via @register.
from . import edgar, fred, gdelt, jobs, reddit, trends, wikipedia  # noqa: F401

__all__ = ["edgar", "fred", "gdelt", "jobs", "reddit", "trends", "wikipedia"]
