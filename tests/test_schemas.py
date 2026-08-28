"""Parser contracts: dtypes, look-ahead protection, and loud failure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import (
    BASE_MS,
    make_raw_candles,
    make_raw_funding,
    make_raw_l2_book,
    make_raw_meta_and_ctxs,
    make_raw_trades,
)
from config.settings import INTERVAL_MS
from data.schemas import (
    ASSET_CTX_COLUMNS,
    CANDLE_COLUMNS,
    FUNDING_COLUMNS,
    ORDERBOOK_COLUMNS,
    TRADE_COLUMNS,
    SchemaError,
    parse_candles,
    parse_funding_history,
    parse_l2_book,
    parse_meta_and_asset_ctxs,
    parse_trades,
    with_ts,
)

STEP = INTERVAL_MS["1m"]
FAR_FUTURE = BASE_MS + 10_000 * STEP


def test_candles_match_the_column_contract() -> None:
    df = parse_candles(make_raw_candles(count=5), now_ms=FAR_FUTURE)
    assert list(df.columns) == list(CANDLE_COLUMNS)
    for name, dtype in CANDLE_COLUMNS.items():
        assert df[name].dtype == dtype, name


def test_candle_strings_become_floats_with_the_right_values() -> None:
    df = parse_candles(make_raw_candles(count=1), now_ms=FAR_FUTURE)
    row = df.iloc[0]
    assert row["open"] == pytest.approx(100_000.0)
    assert row["close"] == pytest.approx(100_010.0)
    assert row["high"] == pytest.approx(100_015.0)
    assert row["low"] == pytest.approx(99_995.0)
    assert row["volume"] == pytest.approx(1.5)
    assert row["trades"] == 40


def test_open_and_close_timestamps_bracket_exactly_one_interval() -> None:
    df = parse_candles(make_raw_candles(count=4), now_ms=FAR_FUTURE)
    assert ((df["close_ts_ms"] - df["ts_ms"]) == STEP - 1).all()


def test_in_progress_bar_is_dropped() -> None:
    """The forming bar's close is not final; storing it would be look-ahead."""
    raw = make_raw_candles(count=5)
    # Wall clock sits inside the final bar: 4 complete bars, 1 forming.
    now = BASE_MS + 4 * STEP + 30_000
    df = parse_candles(raw, now_ms=now)
    assert len(df) == 4
    assert df["close_ts_ms"].max() < now


def test_incomplete_bar_is_kept_when_explicitly_requested() -> None:
    raw = make_raw_candles(count=5)
    now = BASE_MS + 4 * STEP + 30_000
    assert len(parse_candles(raw, now_ms=now, drop_incomplete=False)) == 5


def test_bar_closing_exactly_now_is_treated_as_still_forming() -> None:
    """Boundary case: at t == close_ts the final millisecond has not elapsed."""
    raw = make_raw_candles(count=3)
    now = BASE_MS + 3 * STEP - 1
    df = parse_candles(raw, now_ms=now)
    assert len(df) == 2


def test_duplicate_bars_collapse_to_the_latest() -> None:
    raw = make_raw_candles(count=3)
    stale = dict(raw[1], c="1.0")
    df = parse_candles([raw[0], stale, raw[1], raw[2]], now_ms=FAR_FUTURE)
    assert len(df) == 3
    assert df.loc[df["ts_ms"] == raw[1]["t"], "close"].iloc[0] == pytest.approx(100_020.0)


def test_candles_come_back_sorted() -> None:
    raw = make_raw_candles(count=6)
    df = parse_candles(list(reversed(raw)), now_ms=FAR_FUTURE)
    assert df["ts_ms"].is_monotonic_increasing


def test_empty_candle_payload_yields_typed_empty_frame() -> None:
    df = parse_candles([])
    assert df.empty
    assert list(df.columns) == list(CANDLE_COLUMNS)


@pytest.mark.parametrize("field", ["t", "T", "o", "c", "h", "l", "v", "n"])
def test_malformed_candle_raises_rather_than_guessing(field: str) -> None:
    raw = make_raw_candles(count=1)
    del raw[0][field]
    with pytest.raises(SchemaError, match="malformed candle"):
        parse_candles(raw, now_ms=FAR_FUTURE)


def test_non_numeric_candle_field_raises() -> None:
    raw = make_raw_candles(count=1)
    raw[0]["c"] = "not-a-price"
    with pytest.raises(SchemaError):
        parse_candles(raw, now_ms=FAR_FUTURE)


def test_with_ts_matches_ts_ms() -> None:
    df = with_ts(parse_candles(make_raw_candles(count=3), now_ms=FAR_FUTURE))
    assert df["ts"].dt.tz is not None
    assert df["ts"].iloc[0] == pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    assert df["ts"].equals(pd.to_datetime(df["ts_ms"], unit="ms", utc=True))


# -- funding ---------------------------------------------------------------

def test_funding_parses_and_sorts() -> None:
    df = parse_funding_history(make_raw_funding(count=4))
    assert list(df.columns) == list(FUNDING_COLUMNS)
    assert df["ts_ms"].is_monotonic_increasing
    assert df["funding_rate"].iloc[0] == pytest.approx(0.0000125)


def test_funding_tolerates_a_missing_premium() -> None:
    raw = make_raw_funding(count=2)
    raw[0].pop("premium")
    df = parse_funding_history(raw)
    assert np.isnan(df["premium"].iloc[0])
    assert not np.isnan(df["premium"].iloc[1])


def test_funding_duplicates_collapse() -> None:
    raw = make_raw_funding(count=3)
    assert len(parse_funding_history(raw + raw)) == 3


# -- asset contexts --------------------------------------------------------

def test_asset_ctx_aligns_universe_with_contexts() -> None:
    df = parse_meta_and_asset_ctxs(make_raw_meta_and_ctxs(), recv_ts_ms=BASE_MS)
    assert list(df.columns) == list(ASSET_CTX_COLUMNS)
    assert list(df["coin"]) == ["BTC", "ETH", "SOL"]
    # ETH is index 1, so its markPx must be 100_002, not BTC's 100_001.
    assert df.loc[df["coin"] == "ETH", "mark_px"].iloc[0] == pytest.approx(100_002.0)
    assert df.loc[df["coin"] == "SOL", "open_interest"].iloc[0] == pytest.approx(1020.0)


def test_asset_ctx_filters_to_requested_coins() -> None:
    df = parse_meta_and_asset_ctxs(
        make_raw_meta_and_ctxs(), recv_ts_ms=BASE_MS, coins=["eth"]
    )
    assert list(df["coin"]) == ["ETH"]


def test_asset_ctx_rejects_length_mismatch() -> None:
    """Misalignment would silently attribute one coin's funding to another."""
    meta, ctxs = make_raw_meta_and_ctxs()
    with pytest.raises(SchemaError, match="length mismatch"):
        parse_meta_and_asset_ctxs([meta, ctxs[:2]], recv_ts_ms=BASE_MS)


def test_asset_ctx_rejects_wrong_envelope() -> None:
    with pytest.raises(SchemaError, match="2-element array"):
        parse_meta_and_asset_ctxs([{"universe": []}], recv_ts_ms=BASE_MS)


def test_asset_ctx_missing_field_becomes_nan_not_zero() -> None:
    """A missing open interest must never read as 'zero open interest'."""
    meta, ctxs = make_raw_meta_and_ctxs(["BTC"])
    ctxs[0].pop("openInterest")
    df = parse_meta_and_asset_ctxs([meta, ctxs], recv_ts_ms=BASE_MS)
    assert np.isnan(df["open_interest"].iloc[0])


# -- order book ------------------------------------------------------------

def test_l2_book_flattens_both_sides_in_order() -> None:
    df = parse_l2_book(make_raw_l2_book(depth=3), recv_ts_ms=BASE_MS + 40, depth=10)
    assert list(df.columns) == list(ORDERBOOK_COLUMNS)
    bids = df[df["side"] == "bid"].sort_values("level")
    asks = df[df["side"] == "ask"].sort_values("level")
    assert len(bids) == len(asks) == 3
    assert bids["px"].is_monotonic_decreasing
    assert asks["px"].is_monotonic_increasing
    assert bids["px"].iloc[0] < asks["px"].iloc[0]


def test_l2_book_truncates_to_configured_depth() -> None:
    df = parse_l2_book(make_raw_l2_book(depth=20), recv_ts_ms=BASE_MS, depth=5)
    assert len(df) == 10
    assert df["level"].max() == 4


def test_l2_book_records_receive_latency() -> None:
    df = parse_l2_book(make_raw_l2_book(ts_ms=BASE_MS), recv_ts_ms=BASE_MS + 37, depth=3)
    assert ((df["recv_ts_ms"] - df["ts_ms"]) == 37).all()


def test_l2_book_rejects_bad_envelope() -> None:
    with pytest.raises(SchemaError, match="levels"):
        parse_l2_book({"coin": "BTC", "time": BASE_MS, "levels": []}, recv_ts_ms=BASE_MS, depth=5)


# -- trades ----------------------------------------------------------------

def test_trades_parse_with_side_preserved() -> None:
    df = parse_trades(make_raw_trades(count=4))
    assert list(df.columns) == list(TRADE_COLUMNS)
    assert set(df["side"]) <= {"B", "A"}
    assert df["ts_ms"].is_monotonic_increasing
    assert df["tid"].is_unique


def test_empty_trades_payload() -> None:
    assert parse_trades([]).empty
