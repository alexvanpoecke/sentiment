"""Cross-correlation lag search.

We want the lag L (in quarters) at which the *driver* best leads the *target*:
    target_yoy[q]  ~  driver_yoy[q - L]
i.e. an earlier driver reading predicts a later target. This is the
"lag the data an appropriate amount" step.
"""

from __future__ import annotations

from datetime import date

from ..models import LagStat
from .align import add_quarters
from .stats import pearson


def lagged_pairs(
    driver_yoy: dict[date, float], target_yoy: dict[date, float], lag: int
) -> tuple[list[float], list[float], list[date]]:
    xs: list[float] = []
    ys: list[float] = []
    qs: list[date] = []
    for q in sorted(target_yoy):
        dk = add_quarters(q, -lag)
        if dk in driver_yoy:
            xs.append(driver_yoy[dk])
            ys.append(target_yoy[q])
            qs.append(q)
    return xs, ys, qs


def scan_lags(
    driver_yoy: dict[date, float],
    target_yoy: dict[date, float],
    max_lag: int = 4,
    min_n: int = 6,
) -> tuple[int, list[LagStat]]:
    """Return (best_lag, table). Best = max |r| among lags with n >= min_n
    (falls back to the lag with the most overlap if none qualify)."""
    table: list[LagStat] = []
    for lag in range(0, max_lag + 1):
        xs, ys, _ = lagged_pairs(driver_yoy, target_yoy, lag)
        r, p, n = pearson(xs, ys)
        table.append(LagStat(lag=lag, r=r, p_value=p, n=n))

    def _abs_r(ls: LagStat) -> float:
        return abs(ls.r) if ls.r == ls.r else -1.0  # NaN-safe

    qualified = [ls for ls in table if ls.n >= min_n and ls.r == ls.r]
    if qualified:
        best = max(qualified, key=_abs_r)
    else:
        # not enough overlap anywhere: prefer most data, then strongest corr
        best = max(table, key=lambda ls: (ls.n, _abs_r(ls)))
    return best.lag, table
