"""Pure-Python statistics: Pearson correlation, simple OLS with prediction
intervals, and the Student-t distribution (via the regularized incomplete beta).

No numpy/scipy required. Swap these out for the ``[ml]`` extra later if you want
heavier models — the workflow only depends on the small surface here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_FPMIN = 1e-300
_EPS = 3e-12


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


# --------------------------------------------------------------------------- #
# Regularized incomplete beta  I_x(a, b)  -> Student-t CDF / p-values          #
# --------------------------------------------------------------------------- #
def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    if df <= 0:
        raise ValueError("df must be > 0")
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)  # P(T > |t|)
    return 1.0 - tail if t >= 0 else tail


def student_t_ppf(p: float, df: float) -> float:
    """Inverse CDF (quantile) via bisection — plenty accurate for interval bounds."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    lo, hi = -1000.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _two_sided_t_p(t: float, df: float) -> float:
    # two-sided p-value = I_x(df/2, 1/2) with x = df/(df+t^2)
    return betainc(df / 2.0, 0.5, df / (df + t * t))


# --------------------------------------------------------------------------- #
# Correlation                                                                 #
# --------------------------------------------------------------------------- #
def pearson(xs: list[float], ys: list[float]) -> tuple[float, float, int]:
    """Return (r, two-sided p-value, n). NaN r if undefined."""
    n = len(xs)
    if n != len(ys) or n < 3:
        return (math.nan, 1.0, n)
    mx, my = mean(xs), mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return (math.nan, 1.0, n)
    r = max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))
    df = n - 2
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t = r * math.sqrt(df / (1.0 - r * r))
        p = _two_sided_t_p(t, df)
    return (r, p, n)


# --------------------------------------------------------------------------- #
# Simple linear regression                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class Regression:
    slope: float
    intercept: float
    n: int
    df: int
    r2: float
    resid_std: float  # s, residual standard error
    se_slope: float
    t_slope: float
    p_slope: float
    xbar: float
    sxx: float

    def predict(self, x0: float, alpha: float = 0.20) -> tuple[float, float, float]:
        """Point forecast + (1-alpha) prediction interval for a NEW observation."""
        yhat = self.intercept + self.slope * x0
        if self.df <= 0 or self.sxx <= 0:
            return (yhat, yhat, yhat)
        tcrit = student_t_ppf(1.0 - alpha / 2.0, self.df)
        se_pred = self.resid_std * math.sqrt(1.0 + 1.0 / self.n + (x0 - self.xbar) ** 2 / self.sxx)
        half = tcrit * se_pred
        return (yhat, yhat - half, yhat + half)


def fit_ols(xs: list[float], ys: list[float]) -> Regression:
    n = len(xs)
    if n != len(ys) or n < 3:
        raise ValueError("need at least 3 aligned points to fit a regression")
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0:
        raise ValueError("driver has zero variance; cannot regress")
    slope = sxy / sxx
    intercept = my - slope * mx
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    df = n - 2
    r2 = 1.0 - sse / syy if syy > 0 else 0.0
    resid_std = math.sqrt(sse / df) if df > 0 else 0.0
    se_slope = resid_std / math.sqrt(sxx) if sxx > 0 else math.inf
    t_slope = slope / se_slope if se_slope > 0 else math.inf
    p_slope = _two_sided_t_p(t_slope, df) if df > 0 and math.isfinite(t_slope) else 0.0
    return Regression(
        slope=slope,
        intercept=intercept,
        n=n,
        df=df,
        r2=r2,
        resid_std=resid_std,
        se_slope=se_slope,
        t_slope=t_slope,
        p_slope=p_slope,
        xbar=mx,
        sxx=sxx,
    )


# --------------------------------------------------------------------------- #
# Multiple linear regression (pure Python, normal equations)                  #
# --------------------------------------------------------------------------- #
def _mat_inv(m: list[list[float]]) -> list[list[float]]:
    """Invert a square matrix via Gauss-Jordan with partial pivoting."""
    n = len(m)
    a = [list(m[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            raise ValueError("singular matrix (collinear features?)")
        a[col], a[piv] = a[piv], a[col]
        pivot = a[col][col]
        a[col] = [x / pivot for x in a[col]]
        for r in range(n):
            if r != col and a[r][col] != 0.0:
                factor = a[r][col]
                a[r] = [x - factor * y for x, y in zip(a[r], a[col])]
    return [row[n:] for row in a]


@dataclass
class MultiRegression:
    coef: list[float]  # [intercept, b1, b2, ...]
    n: int
    df: int
    r2: float
    resid_std: float
    se: list[float]  # std error per coefficient
    t: list[float]
    p: list[float]
    _xtx_inv: list[list[float]]

    def predict(self, x0: list[float], alpha: float = 0.20) -> tuple[float, float, float]:
        """Point forecast + (1-alpha) prediction interval for a NEW observation."""
        a0 = [1.0, *x0]
        if len(a0) != len(self.coef):
            raise ValueError(f"predict expected {len(self.coef) - 1} features, got {len(x0)}")
        yhat = sum(c * a for c, a in zip(self.coef, a0))
        if self.df <= 0:
            return (yhat, yhat, yhat)
        quad = sum(
            a0[i] * self._xtx_inv[i][j] * a0[j]
            for i in range(len(a0))
            for j in range(len(a0))
        )
        se_pred = self.resid_std * math.sqrt(max(0.0, 1.0 + quad))
        half = student_t_ppf(1.0 - alpha / 2.0, self.df) * se_pred
        return (yhat, yhat - half, yhat + half)


def fit_ols_multi(features: list[list[float]], y: list[float]) -> MultiRegression:
    """OLS of y on multiple features (intercept added automatically)."""
    n = len(y)
    if n != len(features):
        raise ValueError("features and y length mismatch")
    k = len(features[0]) if features else 0
    if any(len(row) != k for row in features):
        raise ValueError("ragged feature matrix")
    if n < k + 2:
        raise ValueError(f"need at least {k + 2} rows to fit {k} features (+intercept)")

    x = [[1.0, *row] for row in features]
    p = k + 1
    xtx = [[sum(x[r][i] * x[r][j] for r in range(n)) for j in range(p)] for i in range(p)]
    xty = [sum(x[r][i] * y[r] for r in range(n)) for i in range(p)]
    xtx_inv = _mat_inv(xtx)
    coef = [sum(xtx_inv[i][j] * xty[j] for j in range(p)) for i in range(p)]

    sse = sum((y[r] - sum(coef[j] * x[r][j] for j in range(p))) ** 2 for r in range(n))
    ybar = sum(y) / n
    sst = sum((v - ybar) ** 2 for v in y)
    df = n - p
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    resid_std = math.sqrt(sse / df) if df > 0 else 0.0
    se = [resid_std * math.sqrt(max(0.0, xtx_inv[i][i])) for i in range(p)]
    t = [coef[i] / se[i] if se[i] > 0 else math.inf for i in range(p)]
    pvals = [
        _two_sided_t_p(t[i], df) if df > 0 and math.isfinite(t[i]) else 0.0 for i in range(p)
    ]
    return MultiRegression(
        coef=coef, n=n, df=df, r2=r2, resid_std=resid_std, se=se, t=t, p=pvals, _xtx_inv=xtx_inv
    )
