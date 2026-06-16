"""GDELT news-volume / tone connector — PLACEHOLDER (Phase 2).

Planned: query the free GDELT 2.0 DOC API for article volume and average tone
about the entity over time, emitting a daily/weekly news-attention + tone Signal.
No key required. Registered now for visibility and routing.
"""

from __future__ import annotations

from ..registry import register
from .base import Connector


@register
class GdeltConnector(Connector):
    source = "gdelt"
    title = "GDELT news volume & tone"
    free = True
    note = "Phase 2 - not yet implemented."

    def fetch(self, **_):
        raise NotImplementedError(
            "GDELT connector is a Phase 2 stub. Implement the DOC 2.0 timelinevol/tone query here."
        )
