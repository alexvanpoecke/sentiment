"""Calendar-quarter alignment utilities.

Different sources arrive at different frequencies and on different fiscal
calendars. We snap everything to *calendar* quarter-ends so a fiscal-quarterly
KPI (revenue) can be aligned against a monthly driver (e.g. search interest).
"""

from __future__ import annotations

from datetime import date

Series = list[tuple[date, float]]


def calendar_quarter_end(d: date) -> date:
    q = (d.month - 1) // 3  # 0..3
    end_month = q * 3 + 3
    day = 31 if end_month in (3, 12) else 30
    return date(d.year, end_month, day)


def _qindex(qend: date) -> int:
    return qend.year * 4 + (qend.month - 1) // 3


def add_quarters(qend: date, n: int) -> date:
    idx = _qindex(qend) + n
    year, q = divmod(idx, 4)
    end_month = q * 3 + 3
    day = 31 if end_month in (3, 12) else 30
    return date(year, end_month, day)


def prev_quarter(qend: date, n: int = 1) -> date:
    return add_quarters(qend, -n)


def to_quarterly(series: Series, agg: str = "mean") -> dict[date, float]:
    """Aggregate an (date, value) series to calendar quarters."""
    buckets: dict[date, list[float]] = {}
    for d, v in series:
        buckets.setdefault(calendar_quarter_end(d), []).append(v)
    out: dict[date, float] = {}
    for q, vals in buckets.items():
        if agg == "sum":
            out[q] = sum(vals)
        elif agg == "last":
            out[q] = vals[-1]
        else:  # mean
            out[q] = sum(vals) / len(vals)
    return dict(sorted(out.items()))


def quarter_of(d: date) -> int:
    """Calendar quarter number (1..4) of a date."""
    return (d.month - 1) // 3 + 1


def lookback_start(end: date, quarters: int) -> date:
    """The calendar quarter-end ``quarters`` quarters before the quarter containing ``end``.
    Centralizes driver look-back windows so connectors pull a comparable span."""
    return add_quarters(calendar_quarter_end(end), -quarters)


def yoy_quarterly(qmap: dict[date, float]) -> dict[date, float]:
    """Gap-safe YoY for a calendar-quarter-keyed map: value[q] / value[q - 4 quarters] - 1,
    looked up by *date* (4 calendar quarters back), not by list position, so a missing
    quarter never mis-bases the result. Skips a quarter whose year-ago value is missing
    or zero (divide-by-zero)."""
    out: dict[date, float] = {}
    for q, v in qmap.items():
        base = qmap.get(add_quarters(q, -4))
        if base is not None and base != 0:
            out[q] = v / base - 1.0
    return dict(sorted(out.items()))


def align(a: dict[date, float], b: dict[date, float]) -> tuple[list[date], list[float], list[float]]:
    """Inner-join two date-keyed maps, returning aligned, sorted parallel lists."""
    keys = sorted(set(a) & set(b))
    return keys, [a[k] for k in keys], [b[k] for k in keys]
