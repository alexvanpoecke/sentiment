"""Series transforms. A series is an ordered list of (date, value) tuples."""

from __future__ import annotations

from datetime import date

from .stats import mean

Series = list[tuple[date, float]]


def yoy(series: Series, periods: int = 4) -> Series:
    """Year-over-year growth (fraction) by list position. periods=4 quarterly, 12 monthly.

    NOTE: assumes a regularly-spaced series with NO gaps. For quarterly data that may
    have missing quarters, use ``align.yoy_quarterly`` (date-based) instead.
    Skips points whose year-ago base is zero (divide-by-zero).
    """
    out: Series = []
    for i in range(periods, len(series)):
        d, v = series[i]
        base = series[i - periods][1]
        if base != 0:
            out.append((d, v / base - 1.0))
    return out


def qoq(series: Series, periods: int = 1) -> Series:
    """Period-over-period growth (fraction)."""
    out: Series = []
    for i in range(periods, len(series)):
        d, v = series[i]
        base = series[i - periods][1]
        if base and base != 0:
            out.append((d, v / base - 1.0))
    return out


def diff(series: Series, periods: int = 1) -> Series:
    return [
        (series[i][0], series[i][1] - series[i - periods][1]) for i in range(periods, len(series))
    ]


def rolling_mean(series: Series, window: int) -> Series:
    out: Series = []
    for i in range(window - 1, len(series)):
        vals = [series[j][1] for j in range(i - window + 1, i + 1)]
        out.append((series[i][0], mean(vals)))
    return out


def zscore(series: Series) -> Series:
    vals = [v for _, v in series]
    if len(vals) < 2:
        return [(d, 0.0) for d, _ in series]
    mu = mean(vals)
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    sd = var**0.5
    if sd == 0:
        return [(d, 0.0) for d, _ in series]
    return [(d, (v - mu) / sd) for d, v in series]
