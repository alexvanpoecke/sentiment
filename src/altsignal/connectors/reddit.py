"""Reddit forum-sentiment connector — PLACEHOLDER (Phase 2).

Planned: OAuth via the official API (or PRAW), pull submissions/comments for the
entity's seed terms across configured subreddits, score sentiment, and emit a
weekly/monthly mention-volume + net-sentiment Signal. Registered now so it
shows up in `altsignal sources` and the routing config can reference it.
"""

from __future__ import annotations

from ..registry import register
from .base import Connector


@register
class RedditConnector(Connector):
    source = "reddit"
    title = "Reddit mentions & sentiment"
    free = True
    requires = ("reddit_client_id", "reddit_client_secret")
    note = "Phase 2 - not yet implemented."

    def fetch(self, **_):
        raise NotImplementedError(
            "Reddit connector is a Phase 2 stub. Implement OAuth + sentiment scoring here."
        )
