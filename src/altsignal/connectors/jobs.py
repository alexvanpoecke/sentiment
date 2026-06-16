"""Job-postings connector (Greenhouse public boards).

Greenhouse exposes each company's open requisitions as public JSON, no key.
This returns the *current* open-req count — a clean hiring-intensity snapshot.
It's a single point in time, not a history: accumulate it via scheduled runs
(Phase 4) to build a trend. The board token is the company's Greenhouse slug
(e.g. "stripe" for boards.greenhouse.io/stripe), passed explicitly.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..models import Observation, Signal
from ..registry import register
from .base import Connector

BOARD_URL = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs"


@register
class GreenhouseConnector(Connector):
    source = "greenhouse"
    title = "Greenhouse job postings (hiring snapshot)"
    free = True
    min_interval = 0.5
    note = "Current open-req count; snapshot (schedule runs to build history)."

    @staticmethod
    def _count_jobs(obj: dict) -> int:
        return len(obj.get("jobs", []) or [])

    def open_reqs(self, board: str) -> Signal:
        obj = self.get_json(f"greenhouse:{board}", BOARD_URL.format(board=board))
        count = self._count_jobs(obj)
        today = datetime.now(timezone.utc).date()
        return Signal(
            entity_key=f"greenhouse:{board}",
            source=self.source,
            metric="open_reqs",
            freq="D",
            unit="postings",
            observations=[Observation(ts=today, value=float(count))],
            meta={"board": board, "total": count},
        )

    def fetch(self, *, board: str | None = None, **_):
        if not board:
            raise ValueError(
                "greenhouse.fetch needs `board` (the Greenhouse board token, e.g. 'stripe')"
            )
        return [self.open_reqs(board)]
