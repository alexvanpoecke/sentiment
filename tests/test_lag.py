import math
from datetime import date

from altsignal.features.align import add_quarters
from altsignal.features.lag import lagged_pairs, scan_lags


def _build():
    q0 = date(2016, 3, 31)
    driver = {add_quarters(q0, i): math.sin(i * 0.7) + 0.05 * i for i in range(24)}
    # target perfectly tracks the driver from 2 quarters earlier (driver leads by 2)
    target = {}
    for i in range(24):
        q = add_quarters(q0, i)
        src = add_quarters(q, -2)
        if src in driver:
            target[q] = 2.0 * driver[src] + 0.5
    return driver, target


def test_scan_lags_recovers_known_lag():
    driver, target = _build()
    best, table = scan_lags(driver, target, max_lag=4, min_n=5)
    assert best == 2
    row2 = next(r for r in table if r.lag == 2)
    assert abs(row2.r - 1.0) < 1e-9  # perfect linear relationship at the true lag


def test_lagged_pairs_alignment():
    driver, target = _build()
    xs, ys, qs = lagged_pairs(driver, target, 2)
    assert len(xs) == len(ys) == len(qs) > 5
    # ys should equal 2*xs + 0.5 by construction
    assert all(abs(y - (2.0 * x + 0.5)) < 1e-9 for x, y in zip(xs, ys))
