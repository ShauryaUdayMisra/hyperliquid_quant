"""Phase 1's acceptance criteria: no gaps, no future timestamps."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import BASE_MS, make_raw_candles, make_raw_funding
from config.settings import FUNDING_INTERVAL_MS, INTERVAL_MS
from data.quality import (
    check_candles,
    check_funding,
    count_missing,
    count_unsorted,
    find_duplicates,
    find_future_timestamps,
    find_gaps,
    find_invalid_ohlc,
)
from data.schemas import parse_candles, parse_funding_history

STEP = INTERVAL_MS["1m"]
FAR_FUTURE = BASE_MS + 10_000 * STEP


def candles(**kwargs) -> pd.DataFrame:
    return parse_candles(make_raw_candles(**kwargs), now_ms=FAR_FUTURE)


# -- gap detection ---------------------------------------------------------

def test_no_gaps_in_a_contiguous_series() -> None:
    assert find_gaps([BASE_MS + i * STEP for i in range(10)], STEP) == []


def test_single_missing_bar_is_reported_with_exact_bounds() -> None:
    stamps = [BASE_MS + i * STEP for i in range(6) if i != 3]
    gaps = find_gaps(stamps, STEP)
    assert gaps == [(BASE_MS + 3 * STEP, BASE_MS + 3 * STEP)]
    assert count_missing(gaps, STEP) == 1


def test_multi_bar_gap_counts_every_missing_bar() -> None:
    stamps = [BASE_MS + i * STEP for i in range(10) if i not in (4, 5, 6)]
    gaps = find_gaps(stamps, STEP)
    assert gaps == [(BASE_MS + 4 * STEP, BASE_MS + 6 * STEP)]
    assert count_missing(gaps, STEP) == 3


def test_several_separate_gaps() -> None:
    stamps = [BASE_MS + i * STEP for i in range(12) if i not in (2, 7, 8)]
    gaps = find_gaps(stamps, STEP)
    assert len(gaps) == 2
    assert count_missing(gaps, STEP) == 3


def test_gap_detection_ignores_input_order_and_duplicates() -> None:
    stamps = [BASE_MS + i * STEP for i in (5, 0, 1, 1, 2, 4, 3)]
    assert find_gaps(stamps, STEP) == []


def test_series_shorter_than_two_has_no_gaps() -> None:
    assert find_gaps([BASE_MS], STEP) == []
    assert find_gaps([], STEP) == []


def test_candle_report_flags_the_hole_and_scores_completeness() -> None:
    report = check_candles(
        candles(count=20, skip={7, 8}), coin="BTC", interval="1m", reference_ms=FAR_FUTURE
    )
    assert not report.ok
    assert report.missing_bars == 2
    assert report.expected_bars == 20
    assert report.completeness == pytest.approx(0.9)


def test_clean_candle_series_passes_every_check() -> None:
    report = check_candles(candles(count=50), coin="BTC", interval="1m", reference_ms=FAR_FUTURE)
    assert report.ok, report.describe()
    assert report.completeness == 1.0
    assert "OK" in report.describe()


# -- future timestamps -----------------------------------------------------

def test_future_open_timestamp_is_caught() -> None:
    df = candles(count=5)
    df.loc[2, "ts_ms"] = FAR_FUTURE + 10 * STEP
    assert len(find_future_timestamps(df, reference_ms=FAR_FUTURE, tolerance_ms=5_000)) == 1


def test_future_close_timestamp_is_caught_even_when_open_looks_fine() -> None:
    """This is the in-progress-bar leak: open is past, close is not."""
    df = candles(count=5)
    now = int(df["ts_ms"].iloc[-1]) + 10  # inside the last bar
    hits = find_future_timestamps(
        df, reference_ms=now, tolerance_ms=0, columns=("ts_ms", "close_ts_ms")
    )
    assert len(hits) == 1
    report = check_candles(df, coin="BTC", interval="1m", reference_ms=now, tolerance_ms=0)
    assert report.future_rows == 1
    assert not report.ok


def test_small_clock_skew_is_tolerated() -> None:
    df = candles(count=3)
    reference = int(df["close_ts_ms"].max()) - 3_000
    columns = ("ts_ms", "close_ts_ms")
    assert find_future_timestamps(
        df, reference_ms=reference, tolerance_ms=5_000, columns=columns
    ).empty
    assert len(
        find_future_timestamps(df, reference_ms=reference, tolerance_ms=0, columns=columns)
    ) == 1


def test_parser_output_never_contains_future_timestamps() -> None:
    """End-to-end: the default parse path cannot emit an unfinished bar."""
    raw = make_raw_candles(count=200)
    now = BASE_MS + 137 * STEP + 12_345
    df = parse_candles(raw, now_ms=now)
    assert find_future_timestamps(
        df, reference_ms=now, tolerance_ms=0, columns=("ts_ms", "close_ts_ms")
    ).empty
    assert len(df) == 137


# -- other invariants ------------------------------------------------------

def test_duplicate_detection_returns_both_copies() -> None:
    df = candles(count=4)
    doubled = pd.concat([df, df.iloc[[1]]], ignore_index=True)
    assert len(find_duplicates(doubled, ("coin", "interval", "ts_ms"))) == 2


def test_unsorted_rows_are_counted() -> None:
    df = candles(count=5)
    assert count_unsorted(df) == 0
    shuffled = df.iloc[[0, 3, 1, 2, 4]]
    assert count_unsorted(shuffled) == 1


@pytest.mark.parametrize(
    "column,value",
    [("high", 1.0), ("low", 1e9), ("volume", -1.0), ("close", 0.0)],
)
def test_impossible_ohlc_is_flagged(column: str, value: float) -> None:
    df = candles(count=5)
    df.loc[2, column] = value
    assert len(find_invalid_ohlc(df)) == 1


def test_misaligned_bar_timestamp_is_flagged() -> None:
    df = candles(count=5)
    df.loc[3, "ts_ms"] = int(df.loc[3, "ts_ms"]) + 137
    report = check_candles(df, coin="BTC", interval="1m", reference_ms=FAR_FUTURE)
    assert report.misaligned_rows == 1
    assert not report.ok


def test_unknown_interval_skips_gap_check_but_still_reports() -> None:
    df = candles(count=5)
    df["interval"] = "7m"
    report = check_candles(df, coin="BTC", interval="7m", reference_ms=FAR_FUTURE)
    assert report.missing_bars == 0
    assert any("unknown interval" in note for note in report.notes)


def test_empty_input_reports_no_rows() -> None:
    report = check_candles(candles(count=0), coin="BTC", interval="1m")
    assert report.rows == 0
    assert "no rows" in report.describe()


# -- funding ---------------------------------------------------------------

def test_clean_funding_series_passes() -> None:
    df = parse_funding_history(make_raw_funding(count=48))
    report = check_funding(df, coin="BTC", reference_ms=FAR_FUTURE)
    assert report.ok, report.describe()


def test_missing_funding_hour_is_reported() -> None:
    raw = make_raw_funding(count=10)
    del raw[4]
    report = check_funding(parse_funding_history(raw), coin="BTC", reference_ms=FAR_FUTURE)
    assert report.missing_bars == 1
    assert report.gap_ranges == [(BASE_MS + 4 * FUNDING_INTERVAL_MS,) * 2]


def test_off_the_hour_funding_stamp_is_flagged() -> None:
    raw = make_raw_funding(count=5)
    raw[2]["time"] += 61_000
    report = check_funding(parse_funding_history(raw), coin="BTC", reference_ms=FAR_FUTURE)
    assert report.misaligned_rows == 1


def test_extreme_funding_is_noted_not_deleted() -> None:
    raw = make_raw_funding(count=5)
    raw[1]["fundingRate"] = "0.05"
    report = check_funding(parse_funding_history(raw), coin="BTC", reference_ms=FAR_FUTURE)
    assert report.rows == 5
    assert any("above 1%/hr" in note for note in report.notes)


# -- publication jitter (found against real Hyperliquid data) ---------------

def test_hourly_jitter_is_not_reported_as_thousands_of_gaps() -> None:
    """Real funding lands at 06:00:00.019, not 06:00:00.000.

    A strict grid comparison called every one of those a gap, and computed
    ranges whose end preceded their start.
    """
    stamps = [BASE_MS + i * FUNDING_INTERVAL_MS + (i * 17) % 120 for i in range(200)]
    assert find_gaps(stamps, FUNDING_INTERVAL_MS, tolerance_ms=60_000) == []


def test_a_gap_range_never_inverts() -> None:
    stamps = [BASE_MS, BASE_MS + FUNDING_INTERVAL_MS + 19]
    for start, end in find_gaps(stamps, FUNDING_INTERVAL_MS):
        assert start <= end


def test_a_real_missing_hour_is_still_caught_through_the_jitter() -> None:
    stamps = [BASE_MS + i * FUNDING_INTERVAL_MS + (i * 13) % 90 for i in range(20) if i != 8]
    gaps = find_gaps(stamps, FUNDING_INTERVAL_MS, tolerance_ms=60_000)
    assert len(gaps) == 1
    assert count_missing(gaps, FUNDING_INTERVAL_MS) == 1


def test_jittered_funding_passes_the_full_check() -> None:
    raw = make_raw_funding(count=100)
    for i, record in enumerate(raw):
        record["time"] += (i * 19) % 100          # milliseconds past the hour
    report = check_funding(parse_funding_history(raw), coin="BTC", reference_ms=FAR_FUTURE)
    assert report.ok, report.describe()
    assert report.misaligned_rows == 0


def test_funding_genuinely_off_schedule_is_still_flagged() -> None:
    raw = make_raw_funding(count=20)
    raw[5]["time"] += 20 * 60_000                  # 20 minutes late
    report = check_funding(parse_funding_history(raw), coin="BTC", reference_ms=FAR_FUTURE)
    assert report.misaligned_rows == 1
