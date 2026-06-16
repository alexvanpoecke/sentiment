"""Unit tests for the XBRL quarterly-revenue derivation (no network)."""

from datetime import date

from altsignal.connectors.edgar import EdgarConnector


def _e(start, end, val, form="10-Q", filed="2022-01-01"):
    return {"start": start, "end": end, "val": val, "form": form, "filed": filed}


def test_cumulative_ytd_filer_is_differenced():
    # A filer that reports cumulative year-to-date figures each quarter.
    entries = [
        _e("2022-01-01", "2022-03-31", 100, "10-Q", "2022-04-20"),
        _e("2022-01-01", "2022-06-30", 210, "10-Q", "2022-07-20"),
        _e("2022-01-01", "2022-09-30", 330, "10-Q", "2022-10-20"),
        _e("2022-01-01", "2022-12-31", 460, "10-K", "2023-02-20"),
    ]
    disc = EdgarConnector._discrete_quarters(entries)
    assert disc[date(2022, 3, 31)][0] == 100
    assert disc[date(2022, 6, 30)][0] == 110
    assert disc[date(2022, 9, 30)][0] == 120
    assert disc[date(2022, 12, 31)][0] == 130


def test_discrete_quarter_filer_with_annual_total():
    # A filer that reports discrete quarters; the annual 10-K total must NOT
    # be mistaken for a quarter.
    entries = [
        _e("2021-01-01", "2021-03-31", 50, "10-Q", "2021-04-20"),
        _e("2021-04-01", "2021-06-30", 60, "10-Q", "2021-07-20"),
        _e("2021-07-01", "2021-09-30", 70, "10-Q", "2021-10-20"),
        _e("2021-10-01", "2021-12-31", 80, "10-K", "2022-02-20"),
        _e("2021-01-01", "2021-12-31", 260, "10-K", "2022-02-20"),  # annual
    ]
    disc = EdgarConnector._discrete_quarters(entries)
    assert {d: v for d, (v, _) in disc.items()} == {
        date(2021, 3, 31): 50,
        date(2021, 6, 30): 60,
        date(2021, 9, 30): 70,
        date(2021, 12, 31): 80,
    }


def test_latest_filing_wins_on_restatement():
    entries = [
        _e("2022-01-01", "2022-03-31", 100, "10-Q", "2022-04-20"),
        _e("2022-01-01", "2022-03-31", 105, "10-Q/A", "2022-09-01"),  # restated, later filing
    ]
    disc = EdgarConnector._discrete_quarters(entries)
    assert disc[date(2022, 3, 31)][0] == 105


def test_negative_difference_is_dropped():
    # 9-month (900) filed first, FY (850) restated DOWN in the 10-K -> Q4 would be
    # 850-900 = -50; a negative quarter must be dropped, not emitted.
    entries = [
        _e("2023-01-01", "2023-03-31", 300, "10-Q", "2023-04-20"),
        _e("2023-01-01", "2023-06-30", 600, "10-Q", "2023-07-20"),
        _e("2023-01-01", "2023-09-30", 900, "10-Q", "2023-10-20"),
        _e("2023-01-01", "2023-12-31", 850, "10-K", "2024-02-20"),
    ]
    disc = EdgarConnector._discrete_quarters(entries)
    assert {d: v for d, (v, _) in disc.items()} == {
        date(2023, 3, 31): 300,
        date(2023, 6, 30): 300,
        date(2023, 9, 30): 300,
    }
    assert date(2023, 12, 31) not in disc
