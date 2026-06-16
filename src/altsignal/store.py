"""SQLite-backed raw-response cache + normalized signal store.

The raw cache is the important part: it makes reruns fast and keeps us from
hammering rate-limited endpoints. Normalized signal persistence is a convenience
for inspection/backtesting.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import date
from pathlib import Path
from typing import Callable

from .models import Observation, Signal

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_cache (
    key          TEXT PRIMARY KEY,
    fetched_at   REAL NOT NULL,
    ttl          REAL NOT NULL,
    content      BLOB NOT NULL,
    content_type TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_key  TEXT NOT NULL,
    source      TEXT NOT NULL,
    metric      TEXT NOT NULL,
    geo         TEXT,
    freq        TEXT,
    unit        TEXT,
    meta_json   TEXT,
    created_at  REAL NOT NULL,
    UNIQUE(entity_key, source, metric, geo)
);

CREATE TABLE IF NOT EXISTS observations (
    signal_id   INTEGER NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    ts          TEXT NOT NULL,
    value       REAL NOT NULL,
    as_of       TEXT,
    PRIMARY KEY (signal_id, ts)
);

-- Append-only point-in-time panel: every refresh stamps each observation with
-- the date it was captured, so backtests can reconstruct exactly what a signal
-- looked like "as of" a past date (avoiding look-ahead bias from later revisions
-- — Google Trends rescaling, GDELT backfill, EDGAR restatements, etc.).
-- `geo` is '' (not NULL) so it participates cleanly in the primary key.
-- `as_of` is the date the observation actually became public (EDGAR filing date,
-- article publish date) when the connector knows it; `captured_at` is when *we*
-- recorded it. `knowable_at` = COALESCE(as_of, captured_at) is the one canonical
-- "date this became knowable" used for reconstruction — defined once, here, and
-- indexed so the latest-vintage lookup is index-served. NULL as_of falls back to
-- captured_at, so history stays honest even for data predating our first refresh.
CREATE TABLE IF NOT EXISTS panel (
    entity_key  TEXT NOT NULL,
    source      TEXT NOT NULL,
    metric      TEXT NOT NULL,
    geo         TEXT NOT NULL DEFAULT '',
    ts          TEXT NOT NULL,
    value       REAL NOT NULL,
    as_of       TEXT,
    captured_at TEXT NOT NULL,
    knowable_at TEXT GENERATED ALWAYS AS (COALESCE(as_of, captured_at)) VIRTUAL,
    PRIMARY KEY (entity_key, source, metric, geo, ts, captured_at)
);
CREATE INDEX IF NOT EXISTS panel_lookup ON panel(entity_key, source, metric, geo, ts, knowable_at);
"""

# Bump when adding a _migrate() step; gates migration so it runs once, not per process.
_SCHEMA_VERSION = 1


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """One-time, version-gated upgrades for DBs created by an earlier schema.

        Gated on PRAGMA user_version (a single header read), so an up-to-date DB
        skips the catalog scan entirely instead of paying it every process start.
        """
        if self.conn.execute("PRAGMA user_version").fetchone()[0] >= _SCHEMA_VERSION:
            return
        # v0 -> v1: panel gained `as_of` and the `knowable_at` generated column, and
        # its index moved onto knowable_at. CREATE TABLE/INDEX IF NOT EXISTS won't
        # alter a pre-existing panel, so backfill here (no-op on a fresh schema).
        # table_xinfo (not table_info) so the VIRTUAL generated column is listed.
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_xinfo(panel)").fetchall()}
        if cols and "as_of" not in cols:
            self.conn.execute("ALTER TABLE panel ADD COLUMN as_of TEXT")
        if cols and "knowable_at" not in cols:
            self.conn.execute(
                "ALTER TABLE panel ADD COLUMN knowable_at TEXT "
                "GENERATED ALWAYS AS (COALESCE(as_of, captured_at)) VIRTUAL"
            )
            self.conn.execute("DROP INDEX IF EXISTS panel_lookup")  # was on raw captured_at
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS panel_lookup ON panel"
                "(entity_key, source, metric, geo, ts, knowable_at)"
            )
        self.conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    # ------------------------------------------------------------------ cache
    def get_or_fetch_bytes(
        self, key: str, ttl: int, fetch: Callable[[], tuple[bytes, str]]
    ) -> tuple[bytes, str, bool]:
        """Return (content, content_type, from_cache). ``fetch`` is called only on miss/expiry."""
        row = self.conn.execute(
            "SELECT content, content_type, fetched_at FROM raw_cache WHERE key=?", (key,)
        ).fetchone()
        now = time.time()
        if row is not None and (now - row["fetched_at"]) < ttl:
            return row["content"], row["content_type"], True

        content, content_type = fetch()
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO raw_cache(key, fetched_at, ttl, content, content_type) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, now, float(ttl), content, content_type),
            )
            self.conn.commit()
        return content, content_type, False

    def get_or_fetch_json(
        self, key: str, ttl: int, fetch: Callable[[], tuple[bytes, str]]
    ) -> tuple[object, bool]:
        content, _ctype, from_cache = self.get_or_fetch_bytes(key, ttl, fetch)
        return json.loads(content.decode("utf-8")), from_cache

    def cache_stats(self) -> dict[str, int]:
        n = self.conn.execute("SELECT COUNT(*) AS n FROM raw_cache").fetchone()["n"]
        size = self.conn.execute(
            "SELECT COALESCE(SUM(LENGTH(content)),0) AS s FROM raw_cache"
        ).fetchone()["s"]
        return {"entries": int(n), "bytes": int(size)}

    # ----------------------------------------------------------- signal store
    def save_signal(self, sig: Signal) -> int:
        with self._lock:
            # Explicit upsert: ON CONFLICT can't be relied on here because SQLite
            # treats NULL `geo` as distinct in the UNIQUE index, so the conflict
            # would never fire for (common) geo-less signals and duplicate rows
            # would accumulate. `geo IS ?` matches both NULL and non-NULL.
            row = self.conn.execute(
                "SELECT id FROM signals WHERE entity_key=? AND source=? AND metric=? AND geo IS ?",
                (sig.entity_key, sig.source, sig.metric, sig.geo),
            ).fetchone()
            meta_json = json.dumps(sig.meta, default=str)
            now = time.time()
            if row is not None:
                sid = int(row["id"])
                self.conn.execute(
                    "UPDATE signals SET freq=?, unit=?, meta_json=?, created_at=? WHERE id=?",
                    (sig.freq, sig.unit, meta_json, now, sid),
                )
            else:
                cur = self.conn.execute(
                    "INSERT INTO signals(entity_key, source, metric, geo, freq, unit, meta_json, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (sig.entity_key, sig.source, sig.metric, sig.geo, sig.freq, sig.unit, meta_json, now),
                )
                sid = int(cur.lastrowid)
            self.conn.execute("DELETE FROM observations WHERE signal_id=?", (sid,))
            self.conn.executemany(
                "INSERT OR REPLACE INTO observations(signal_id, ts, value, as_of) VALUES (?,?,?,?)",
                [
                    (sid, o.ts.isoformat(), float(o.value), o.as_of.isoformat() if o.as_of else None)
                    for o in sig.observations
                ],
            )
            self.conn.commit()
            return sid

    def load_signal(
        self, entity_key: str, source: str, metric: str, geo: str | None = None
    ) -> Signal | None:
        row = self.conn.execute(
            "SELECT * FROM signals WHERE entity_key=? AND source=? AND metric=? AND geo IS ?",
            (entity_key, source, metric, geo),
        ).fetchone()
        if row is None:
            return None
        obs_rows = self.conn.execute(
            "SELECT ts, value, as_of FROM observations WHERE signal_id=? ORDER BY ts", (row["id"],)
        ).fetchall()
        obs = [
            Observation(
                ts=date.fromisoformat(r["ts"]),
                value=r["value"],
                as_of=date.fromisoformat(r["as_of"]) if r["as_of"] else None,
            )
            for r in obs_rows
        ]
        return Signal(
            entity_key=row["entity_key"],
            source=row["source"],
            metric=row["metric"],
            geo=row["geo"],
            freq=row["freq"] or "Q",
            unit=row["unit"],
            observations=obs,
            meta=json.loads(row["meta_json"]) if row["meta_json"] else {},
        )

    # ------------------------------------------------------- point-in-time panel
    def record_panel(self, entity_key: str, sig: Signal, captured_at: date) -> int:
        """Append a vintage of ``sig`` under ``entity_key``, stamped ``captured_at``.

        Each observation also carries its own ``as_of`` (the date it became public,
        e.g. an EDGAR filing date) when the connector provides one; reconstruction
        prefers it over ``captured_at``. Idempotent within a
        (entity, source, metric, geo, ts, captured_at) key: re-running on the same
        day overwrites that day's value rather than duplicating it. Returns the
        number of observations recorded.
        """
        cap = captured_at.isoformat()
        geo = sig.geo or ""
        rows = [
            (entity_key, sig.source, sig.metric, geo, o.ts.isoformat(), float(o.value),
             o.as_of.isoformat() if o.as_of else None, cap)
            for o in sig.observations
        ]
        if not rows:
            return 0
        with self._lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO panel"
                "(entity_key, source, metric, geo, ts, value, as_of, captured_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )
            self.conn.commit()
        return len(rows)

    def load_panel_as_of(
        self,
        entity_key: str,
        source: str,
        metric: str,
        geo: str | None,
        as_of: date,
    ) -> Signal | None:
        """Reconstruct a signal as it was known on ``as_of``.

        For each observation period, take the value from the latest vintage whose
        knowable-by date ``COALESCE(as_of, captured_at) <= as_of`` — i.e. the most
        recent reading public on that date, preferring the observation's true
        publish/filing date and falling back to when we captured it. Revisions
        published later are ignored. This is the point-in-time view a backtest must
        use to avoid look-ahead bias. Returns None if nothing was knowable on or
        before ``as_of``.
        """
        g = geo or ""
        cap = as_of.isoformat()
        rows = self.conn.execute(
            "SELECT ts, value, knowable_at FROM panel p "
            "WHERE entity_key=? AND source=? AND metric=? AND geo=? "
            "AND knowable_at<=? "
            "AND knowable_at=("
            "  SELECT MAX(knowable_at) FROM panel "
            "  WHERE entity_key=p.entity_key AND source=p.source AND metric=p.metric "
            "  AND geo=p.geo AND ts=p.ts AND knowable_at<=?"
            ") ORDER BY ts",
            (entity_key, source, metric, g, cap, cap),
        ).fetchall()
        if not rows:
            return None
        # Preserve the true vintage (filing/publish or capture date) as each obs's
        # as_of, rather than flattening it to the query date.
        obs = [
            Observation(
                ts=date.fromisoformat(r["ts"]), value=r["value"],
                as_of=date.fromisoformat(r["knowable_at"]),
            )
            for r in rows
        ]
        return Signal(
            entity_key=entity_key, source=source, metric=metric,
            geo=geo, observations=obs,
        )

    def panel_summary(self, entity_key: str | None = None) -> list[dict]:
        """Per-series coverage of the panel: observation span and vintage count.

        Optionally filtered to one ``entity_key``. Each row reports how many
        distinct periods and vintages have been captured and the capture window —
        i.e. how much point-in-time history has accumulated.
        """
        sql = (
            "SELECT entity_key, source, metric, geo, "
            "COUNT(DISTINCT ts) AS n_obs, COUNT(DISTINCT captured_at) AS n_vintages, "
            "MIN(ts) AS first_ts, MAX(ts) AS last_ts, "
            "MIN(captured_at) AS first_capture, MAX(captured_at) AS last_capture "
            "FROM panel "
        )
        params: tuple = ()
        if entity_key is not None:
            sql += "WHERE entity_key=? "
            params = (entity_key,)
        sql += "GROUP BY entity_key, source, metric, geo ORDER BY entity_key, source, metric, geo"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def close(self) -> None:
        self.conn.close()


_default_store: Store | None = None


def get_store(db_path: str | Path | None = None) -> Store:
    """Process-wide default store (created lazily)."""
    global _default_store
    if db_path is not None:
        return Store(db_path)
    if _default_store is None:
        from .config import get_settings

        _default_store = Store(get_settings().db_path)
    return _default_store
