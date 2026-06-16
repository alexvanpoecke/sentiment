import math

import pytest

from altsignal.features.stats import (
    betainc,
    fit_ols,
    fit_ols_multi,
    pearson,
    student_t_cdf,
    student_t_ppf,
)


def test_pearson_perfect_positive():
    r, p, n = pearson([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert abs(r - 1.0) < 1e-9
    assert n == 5
    assert p < 1e-6


def test_pearson_perfect_negative():
    r, _, _ = pearson([1, 2, 3, 4], [4, 3, 2, 1])
    assert abs(r + 1.0) < 1e-9


def test_betainc_symmetry():
    assert abs(betainc(0.5, 0.5, 0.5) - 0.5) < 1e-6


def test_t_cdf_basic():
    assert abs(student_t_cdf(0.0, 10) - 0.5) < 1e-9
    # symmetry: F(-t) = 1 - F(t)
    assert abs(student_t_cdf(-1.5, 7) + student_t_cdf(1.5, 7) - 1.0) < 1e-9
    # known value: P(T<=2) with df=10 ~ 0.9633
    assert abs(student_t_cdf(2.0, 10) - 0.9633) < 2e-3


def test_t_ppf():
    # two-sided 95% critical value, df=10, ~ 2.228
    assert abs(student_t_ppf(0.975, 10) - 2.2281) < 1e-2


def test_ols_exact_fit():
    xs = [0, 1, 2, 3, 4]
    ys = [1, 3, 5, 7, 9]  # y = 2x + 1
    reg = fit_ols(xs, ys)
    assert abs(reg.slope - 2.0) < 1e-9
    assert abs(reg.intercept - 1.0) < 1e-9
    assert abs(reg.r2 - 1.0) < 1e-9
    yhat, lo, hi = reg.predict(5.0)
    assert abs(yhat - 11.0) < 1e-9
    # perfect fit -> zero residual std -> degenerate (tight) interval
    assert abs(hi - lo) < 1e-6


def test_ols_interval_widens_with_noise():
    xs = [0, 1, 2, 3, 4, 5]
    ys = [0.1, 1.0, 2.2, 2.9, 4.1, 5.2]
    reg = fit_ols(xs, ys)
    yhat, lo, hi = reg.predict(2.5, alpha=0.2)
    assert lo < yhat < hi
    assert not math.isnan(reg.p_slope)


def test_fit_ols_multi_exact_plane():
    feats = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [1, 2]]
    y = [1 + 2 * a + 3 * b for a, b in feats]  # y = 1 + 2*x1 + 3*x2
    reg = fit_ols_multi(feats, y)
    assert abs(reg.coef[0] - 1.0) < 1e-6
    assert abs(reg.coef[1] - 2.0) < 1e-6
    assert abs(reg.coef[2] - 3.0) < 1e-6
    assert abs(reg.r2 - 1.0) < 1e-9
    yhat, lo, hi = reg.predict([3, 3])
    assert abs(yhat - 16.0) < 1e-6
    assert abs(hi - lo) < 1e-6  # exact fit -> degenerate (tight) interval


def test_fit_ols_multi_matches_simple_ols_on_one_feature():
    xs = [0, 1, 2, 3, 4]
    ys = [1.0, 3.1, 4.9, 7.2, 8.8]
    simple = fit_ols(xs, ys)
    multi = fit_ols_multi([[x] for x in xs], ys)
    assert abs(multi.coef[0] - simple.intercept) < 1e-9
    assert abs(multi.coef[1] - simple.slope) < 1e-9


def test_fit_ols_multi_rejects_collinear_features():
    feats = [[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]]  # x2 = 2*x1
    with pytest.raises(ValueError):
        fit_ols_multi(feats, [1, 2, 3, 4, 5])
