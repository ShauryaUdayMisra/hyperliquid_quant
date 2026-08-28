"""Data-integrity checks.

Phase 1's acceptance test is "no gaps and no future timestamps", but a few
more invariants are cheap to check and catch the same class of bug:

* duplicate bars (a botched backfill merge),
* non-monotonic time (files concatenated in the wrong order),
* impossible OHLC (high below the open, negative volume),
* bars whose close time is in the future (an in-progress bar that leaked
  through, which is the look-ahead failure mode we care most about).

Everything here is a pure function over a DataFrame so the tests never need
a network or a filesystem.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from config.settings import FUNDING_INTERVAL_MS, INTERVAL_MS


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class QualityReport:
    """Findings for one (dataset, coin, interval) slice."""

    dataset: str
    coin: str
    interval: str | None = None
    rows: int = 0
    start_ms: int | None = None
    end_ms: int | None = None
    duplicate_rows: int = 0
    future_rows: int = 0
    unsorted_rows: int = 0
    invalid_ohlc_rows: int = 0
    misaligned_rows: int = 0
    missing_bars: int = 0
    expected_bars: int = 0
    gap_ranges: list[tuple[int, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.duplicate_rows == 0
            and self.future_rows == 0
            and self.unsorted_rows == 0
            and self.invalid_ohlc_rows == 0
            and self.misaligned_rows == 0
            and self.missing_bars == 0
        )

    @property
    def completeness(self) -> float:
        if self.expected_bars <= 0:
            return 1.0
        return (self.expected_bars - self.missing_bars) / self.expected_bars

    def describe(self) -> str:
        label = f"{self.dataset}/{self.coin}" + (f"/{self.interval}" if self.interval else "")
        if self.rows == 0:
            return f"{label}: no rows"
        span = (
            f"{pd.Timestamp(self.start_ms, unit='ms', tz='UTC')} -> "
            f"{pd.Timestamp(self.end_ms, unit='ms', tz='UTC')}"
        )
        status = "OK " if self.ok else "FAIL"
        detail = (
            f"rows={self.rows} complete={self.completeness:.4%} "
            f"dupes={self.duplicate_rows} future={self.future_rows} "
            f"unsorted={self.unsorted_rows} bad_ohlc={self.invalid_ohlc_rows} "
            f"misaligned={self.misaligned_rows} missing={self.missing_bars}"
        )
        lines = [f"[{status}] {label}  {span}  {detail}"]
        for gap_start, gap_end in self.gap_ranges[:5]:
            lines.append(
                f"         gap: {pd.Timestamp(gap_start, unit='ms', tz='UTC')} -> "
                f"{pd.Timestamp(gap_end, unit='ms', tz='UTC')}"
            )
        if len(self.gap_ranges) > 5:
            lines.append(f"         ... and {len(self.gap_ranges) - 5} more gaps")
        lines.extend(f"         note: {n}" for n in self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Primitive checks
# --------------------------------------------------------------------------

def find_future_timestamps(
    df: pd.DataFrame,
    *,
    reference_ms: int | None = None,
    tolerance_ms: int = 5_000,
    columns: Sequence[str] = ("ts_ms",),
) -> pd.DataFrame:
    """Rows carrying a timestamp beyond ``reference_ms + tolerance_ms``."""
    if df.empty:
        return df
    cutoff = (now_ms() if reference_ms is None else reference_ms) + tolerance_ms
    mask = np.zeros(len(df), dtype=bool)
    for column in columns:
        if column in df.columns:
            mask |= df[column].to_numpy() > cutoff
    return df.loc[mask]


def find_duplicates(df: pd.DataFrame, key: Sequence[str]) -> pd.DataFrame:
    """Rows sharing a primary key with another row."""
    if df.empty:
        return df
    return df.loc[df.duplicated(subset=list(key), keep=False)]


def count_unsorted(df: pd.DataFrame, column: str = "ts_ms") -> int:
    """Number of positions where time moves backwards."""
    if len(df) < 2:
        return 0
    values = df[column].to_numpy()
    return int((np.diff(values) < 0).sum())


def find_invalid_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Bars that violate high >= max(open, close) >= min(open, close) >= low."""
    if df.empty:
        return df
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    open_, close = df["open"].to_numpy(), df["close"].to_numpy()
    bad = (
        (high < low)
        | (high < open_)
        | (high < close)
        | (low > open_)
        | (low > close)
        | (df["volume"].to_numpy() < 0)
        | ~np.isfinite(close)
        | (close <= 0)
    )
    return df.loc[bad]


def find_gaps(
    timestamps: Sequence[int] | np.ndarray,
    step_ms: int,
    *,
    tolerance_ms: int = 0,
) -> list[tuple[int, int]]:
    """Contiguous ranges of missing timestamps on a fixed grid.

    Returns ``(first_missing_ms, last_missing_ms)`` pairs, both inclusive.

    ``tolerance_ms`` absorbs publication jitter. Hyperliquid funding is
    hourly but lands a few tens of milliseconds past the hour, so a strict
    comparison reports thousands of one-millisecond "gaps" that do not
    exist. Candles sit on an exact grid and use the default of zero.
    """
    values = np.asarray(sorted(set(int(t) for t in timestamps)), dtype=np.int64)
    if values.size < 2:
        return []
    diffs = np.diff(values)
    gaps = []
    for idx in np.flatnonzero(diffs > step_ms + tolerance_ms):
        start = int(values[idx] + step_ms)
        end = int(values[idx + 1] - step_ms)
        # Jitter can make the computed range invert; that is not a gap.
        if start <= end:
            gaps.append((start, end))
    return gaps


def count_missing(gap_ranges: Sequence[tuple[int, int]], step_ms: int) -> int:
    return int(sum((end - start) // step_ms + 1 for start, end in gap_ranges))


# --------------------------------------------------------------------------
# Dataset-level checks
# --------------------------------------------------------------------------

def check_candles(
    df: pd.DataFrame,
    *,
    coin: str,
    interval: str,
    reference_ms: int | None = None,
    tolerance_ms: int = 5_000,
) -> QualityReport:
    """Full integrity report for one coin/interval candle series."""
    step = INTERVAL_MS.get(interval)
    report = QualityReport(dataset="candles", coin=coin, interval=interval, rows=len(df))
    if step is None:
        report.notes.append(f"unknown interval '{interval}'; gap check skipped")
    if df.empty:
        return report

    df = df.sort_values("ts_ms")
    report.start_ms = int(df["ts_ms"].iloc[0])
    report.end_ms = int(df["ts_ms"].iloc[-1])

    # A bar whose CLOSE is in the future is an in-progress bar: including it
    # would let a feature see a price that has not finished forming.
    report.future_rows = len(
        find_future_timestamps(
            df,
            reference_ms=reference_ms,
            tolerance_ms=tolerance_ms,
            columns=("ts_ms", "close_ts_ms"),
        )
    )
    report.duplicate_rows = len(find_duplicates(df, ("coin", "interval", "ts_ms")))
    report.unsorted_rows = count_unsorted(df)
    report.invalid_ohlc_rows = len(find_invalid_ohlc(df))

    if step is not None:
        misaligned = df.loc[df["ts_ms"] % step != 0]
        report.misaligned_rows = len(misaligned)
        report.gap_ranges = find_gaps(df["ts_ms"].to_numpy(), step)
        report.missing_bars = count_missing(report.gap_ranges, step)
        report.expected_bars = (report.end_ms - report.start_ms) // step + 1

    return report


#: Funding prints are hourly but not to the millisecond. Anything within a
#: minute of the hour is on schedule; beyond that is worth flagging.
FUNDING_JITTER_TOLERANCE_MS = 60_000


def check_funding(
    df: pd.DataFrame,
    *,
    coin: str,
    reference_ms: int | None = None,
    tolerance_ms: int = 5_000,
    jitter_tolerance_ms: int = FUNDING_JITTER_TOLERANCE_MS,
) -> QualityReport:
    """Integrity report for one coin's hourly funding series."""
    report = QualityReport(dataset="funding", coin=coin, rows=len(df))
    if df.empty:
        return report

    df = df.sort_values("ts_ms")
    report.start_ms = int(df["ts_ms"].iloc[0])
    report.end_ms = int(df["ts_ms"].iloc[-1])
    report.future_rows = len(
        find_future_timestamps(df, reference_ms=reference_ms, tolerance_ms=tolerance_ms)
    )
    report.duplicate_rows = len(find_duplicates(df, ("coin", "ts_ms")))
    report.unsorted_rows = count_unsorted(df)
    # Distance to the nearest hour, in either direction.
    offset = df["ts_ms"] % FUNDING_INTERVAL_MS
    drift = np.minimum(offset, FUNDING_INTERVAL_MS - offset)
    report.misaligned_rows = int((drift > jitter_tolerance_ms).sum())

    report.gap_ranges = find_gaps(
        df["ts_ms"].to_numpy(), FUNDING_INTERVAL_MS, tolerance_ms=jitter_tolerance_ms
    )
    report.missing_bars = count_missing(report.gap_ranges, FUNDING_INTERVAL_MS)
    report.expected_bars = (report.end_ms - report.start_ms) // FUNDING_INTERVAL_MS + 1

    extreme = df.loc[df["funding_rate"].abs() > 0.01]
    if len(extreme):
        report.notes.append(
            f"{len(extreme)} funding rate(s) above 1%/hr - verify, do not assume bad"
        )
    return report
