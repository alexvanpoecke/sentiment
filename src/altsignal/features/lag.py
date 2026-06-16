"""Cross-correlation lag search with out-of-sample skill scoring.

We want the lag L (in quarters) at which the *driver* best leads the *target*:
    target_yoy[q]  ~  driver_yoy[q - L]

By default the winning lag is chosen by **out-of-sample skill** — a walk-forward
backtest against a naive persistence benchmark — not by the highest in-sample
|r|. Max-|r| selection readily picks spurious, overfit lags (and even the wrong
sign); skill selection picks the lag that actually generalizes. An optional
``sign`` constraint restricts candidates to the economically-expected direction
(e.g. a demand driver should correlate *positively* with revenue).
"""

from __future__ import annotations

from datetime import date

from ..models import LagStat
from .align import add_quarters
from .stats import fit_ols, mean, pearson


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


def walk_forward(
    xs: list[float], ys: list[float], qs: list[date], target_yoy: dict[date, float]
) -> tuple[int, float | None, float | None]:
    """One-step expanding-window backtest vs naive persistence.

    Naive = the prior quarter's YoY carried forward (persistence), looked up by
    DATE (the true prior calendar quarter), since ``qs`` can skip quarters lacking
    a driver reading. Returns (n_folds, model_mae, naive_mae); MAEs are None if no
    fold could be scored.
    """
    errs: list[float] = []
    naive: list[float] = []
    for i in range(4, len(xs)):
        prev_q = add_quarters(qs[i], -1)
        if prev_q not in target_yoy:
            continue
        try:
            reg = fit_ols(xs[:i], ys[:i])
        except ValueError:
            continue
        errs.append(abs(reg.predict(xs[i])[0] - ys[i]))
        naive.append(abs(target_yoy[prev_q] - ys[i]))
    if not errs:
        return 0, None, None
    return len(errs), mean(errs), mean(naive)


def skill_score(model_mae: float | None, naive_mae: float | None) -> float | None:
    """Fraction by which the model beats naive persistence (>0 = better)."""
    if model_mae is None or naive_mae is None or naive_mae <= 0:
        return None
    return (naive_mae - model_mae) / naive_mae


def _abs_r(ls: LagStat) -> float:
    return abs(ls.r) if ls.r == ls.r else -1.0  # NaN-safe


def scan_lags(
    driver_yoy: dict[date, float],
    target_yoy: dict[date, float],
    *,
    max_lag: int = 4,
    min_n: int = 6,
    lag_by: str = "skill",
    sign: str = "any",
) -> tuple[int, list[LagStat]]:
    """Score every lag 0..max_lag and return (best_lag, full_table).

    ``lag_by``: "skill" (default, max out-of-sample skill) or "corr" (max |r|).
    ``sign``:   "any" (default), "positive", or "negative" — restricts the
                candidate pool to that correlation direction.
    Falls back gracefully (relax sign, then min_n) when nothing qualifies.
    """
    table: list[LagStat] = []
    for lag in range(0, max_lag + 1):
        xs, ys, qs = lagged_pairs(driver_yoy, target_yoy, lag)
        r, p, n = pearson(xs, ys)
        folds, model_mae, naive_mae = walk_forward(xs, ys, qs, target_yoy)
        table.append(
            LagStat(
                lag=lag,
                r=r,
                p_value=p,
                n=n,
                folds=folds,
                model_mae=model_mae,
                naive_mae=naive_mae,
                skill=skill_score(model_mae, naive_mae),
            )
        )

    def sign_ok(ls: LagStat) -> bool:
        if sign == "positive":
            return ls.r > 0
        if sign == "negative":
            return ls.r < 0
        return True

    has_r = [ls for ls in table if ls.n >= min_n and ls.r == ls.r]
    pool = [ls for ls in has_r if sign_ok(ls)] or has_r or table

    if lag_by == "skill":
        scored = [ls for ls in pool if ls.skill is not None and ls.skill == ls.skill]
        if scored:
            best = max(scored, key=lambda ls: ls.skill)
        else:  # no fold could be scored anywhere: fall back to most data, then |r|
            best = max(pool, key=lambda ls: (ls.n, _abs_r(ls)))
    else:  # "corr"
        best = max(pool, key=_abs_r)
    return best.lag, table
