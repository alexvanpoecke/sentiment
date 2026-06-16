from datetime import date

from altsignal.features.transforms import diff, qoq, yoy, zscore


def test_yoy_quarterly():
    s = [
        (date(2022, 3, 31), 100.0),
        (date(2022, 6, 30), 100.0),
        (date(2022, 9, 30), 100.0),
        (date(2022, 12, 31), 100.0),
        (date(2023, 3, 31), 120.0),
    ]
    out = yoy(s, periods=4)
    assert len(out) == 1
    assert out[0][0] == date(2023, 3, 31)
    assert abs(out[0][1] - 0.20) < 1e-12


def test_yoy_skips_nonpositive_base():
    s = [(date(2022, 3, 31), 0.0), (date(2023, 3, 31), 50.0)]
    assert yoy(s, periods=1) == []


def test_qoq_and_diff():
    s = [(date(2023, 3, 31), 100.0), (date(2023, 6, 30), 110.0)]
    assert abs(qoq(s)[0][1] - 0.10) < 1e-12
    assert abs(diff(s)[0][1] - 10.0) < 1e-12


def test_zscore_mean_zero():
    s = [(date(2023, 1, 1), v) for v in (1.0, 2.0, 3.0, 4.0, 5.0)]
    z = zscore(s)
    assert abs(sum(v for _, v in z)) < 1e-9
