import math

from altsignal.features.stats import betainc, fit_ols, pearson, student_t_cdf, student_t_ppf


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
