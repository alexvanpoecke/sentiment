from datetime import date

from altsignal.features.align import (
    add_quarters,
    align,
    calendar_quarter_end,
    prev_quarter,
    to_quarterly,
)


def test_calendar_quarter_end():
    assert calendar_quarter_end(date(2023, 1, 1)) == date(2023, 3, 31)
    assert calendar_quarter_end(date(2023, 2, 15)) == date(2023, 3, 31)
    assert calendar_quarter_end(date(2023, 5, 9)) == date(2023, 6, 30)
    assert calendar_quarter_end(date(2023, 11, 25)) == date(2023, 12, 31)


def test_add_and_prev_quarters():
    assert add_quarters(date(2023, 3, 31), 1) == date(2023, 6, 30)
    assert add_quarters(date(2023, 12, 31), 1) == date(2024, 3, 31)
    assert add_quarters(date(2023, 3, 31), -4) == date(2022, 3, 31)
    assert prev_quarter(date(2024, 3, 31), 1) == date(2023, 12, 31)


def test_to_quarterly_mean_and_sum():
    s = [
        (date(2023, 1, 31), 10.0),
        (date(2023, 2, 28), 20.0),
        (date(2023, 3, 31), 30.0),
    ]
    assert to_quarterly(s, "mean")[date(2023, 3, 31)] == 20.0
    assert to_quarterly(s, "sum")[date(2023, 3, 31)] == 60.0


def test_align_inner_join():
    a = {date(2023, 3, 31): 1.0, date(2023, 6, 30): 2.0}
    b = {date(2023, 6, 30): 9.0, date(2023, 9, 30): 8.0}
    keys, av, bv = align(a, b)
    assert keys == [date(2023, 6, 30)]
    assert av == [2.0] and bv == [9.0]


def test_yoy_quarterly_is_gap_safe():
    from altsignal.features.align import yoy_quarterly

    q0 = date(2019, 3, 31)
    qmap = {add_quarters(q0, i): 100.0 + 10 * i for i in range(8)}
    del qmap[add_quarters(q0, 4)]  # drop 2020-03-31 to create a gap

    y = yoy_quarterly(qmap)
    q5 = add_quarters(q0, 5)  # 2020-06-30, value 150; true year-ago is 2019-06-30 = 110
    # Index-based YoY would wrongly pair q5 with 2019-03-31 (100) -> 0.50.
    assert abs(y[q5] - (150.0 / 110.0 - 1.0)) < 1e-12
    assert add_quarters(q0, 4) not in y  # the missing quarter has no YoY


def test_quarter_of_and_lookback_start():
    from altsignal.features.align import lookback_start, quarter_of

    assert quarter_of(date(2023, 2, 15)) == 1
    assert quarter_of(date(2023, 12, 31)) == 4
    # 4 calendar quarters before the quarter containing 2023-05-10 (Q2'23) -> Q2'22 end
    assert lookback_start(date(2023, 5, 10), 4) == date(2022, 6, 30)
