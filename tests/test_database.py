"""Storage: idempotent upserts, append semantics, DuckDB views."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import BASE_MS, make_raw_candles, make_raw_funding, make_raw_l2_book, make_raw_trades
from config.settings import INTERVAL_MS
from data.database import MarketDatabase, ParquetStore
from data.schemas import parse_candles, parse_funding_history, parse_l2_book, parse_trades

STEP = INTERVAL_MS["1m"]
FAR_FUTURE = BASE_MS + 10_000 * STEP


def candles(**kw) -> pd.DataFrame:
    return parse_candles(make_raw_candles(**kw), now_ms=FAR_FUTURE)


def test_upsert_then_read_back_round_trips(store: ParquetStore) -> None:
    df = candles(count=10)
    assert store.upsert("candles", df) == 10
    with MarketDatabase(store) as db:
        back = db.query("SELECT * FROM candles ORDER BY ts_ms")
    assert len(back) == 10
    assert back["close"].tolist() == df["close"].tolist()


def test_upsert_is_idempotent(store: ParquetStore) -> None:
    """Re-running a backfill must not duplicate or grow the dataset."""
    df = candles(count=10)
    store.upsert("candles", df)
    assert store.upsert("candles", df) == 0
    with MarketDatabase(store) as db:
        assert db.query("SELECT count(*) c FROM candles").iloc[0]["c"] == 10


def test_upsert_overwrites_a_revised_bar(store: ParquetStore) -> None:
    df = candles(count=5)
    store.upsert("candles", df)
    revised = df.copy()
    revised.loc[2, "close"] = 123.0
    store.upsert("candles", revised)
    with MarketDatabase(store) as db:
        row = db.query("SELECT close FROM candles ORDER BY ts_ms").iloc[2]
    assert row["close"] == pytest.approx(123.0)


def test_upsert_extends_an_existing_series(store: ParquetStore) -> None:
    store.upsert("candles", candles(count=5))
    later = candles(count=5, start_ms=BASE_MS + 5 * STEP)
    assert store.upsert("candles", later) == 5
    with MarketDatabase(store) as db:
        assert db.query("SELECT count(*) c FROM candles").iloc[0]["c"] == 10


def test_partitions_split_by_coin_and_interval(store: ParquetStore) -> None:
    store.upsert("candles", candles(coin="BTC", interval="1m", count=3))
    store.upsert("candles", candles(coin="ETH", interval="1m", count=3))
    store.upsert("candles", candles(coin="BTC", interval="5m", count=3))
    root = store.dataset_dir("candles")
    assert (root / "coin=BTC" / "interval=1m").is_dir()
    assert (root / "coin=BTC" / "interval=5m").is_dir()
    assert (root / "coin=ETH" / "interval=1m").is_dir()
    with MarketDatabase(store) as db:
        counts = db.query("SELECT coin, interval, count(*) c FROM candles GROUP BY 1,2")
    assert len(counts) == 3


def test_data_spanning_midnight_splits_into_day_files(store: ParquetStore) -> None:
    start = BASE_MS - 3 * STEP  # 2025-12-31 23:57Z
    store.upsert("candles", candles(count=6, start_ms=start))
    files = sorted(p.name for p in store.dataset_dir("candles").rglob("*.parquet"))
    assert files == ["2025-12-31.parquet", "2026-01-01.parquet"]
    with MarketDatabase(store) as db:
        assert db.query("SELECT count(*) c FROM candles").iloc[0]["c"] == 6


def test_funding_partitions_by_month(store: ParquetStore) -> None:
    store.upsert("funding", parse_funding_history(make_raw_funding(count=30)))
    files = [p.name for p in store.dataset_dir("funding").rglob("*.parquet")]
    assert files == ["2026-01.parquet"]


def test_append_creates_new_part_files_and_keeps_every_row(store: ParquetStore) -> None:
    for i in range(3):
        store.append("trades", parse_trades(make_raw_trades(count=2, start_ms=BASE_MS + i * 1000)))
    assert store.file_count("trades") == 3
    with MarketDatabase(store) as db:
        assert db.query("SELECT count(*) c FROM trades").iloc[0]["c"] == 6


def test_orderbook_round_trips_with_latency_preserved(store: ParquetStore) -> None:
    df = parse_l2_book(make_raw_l2_book(), recv_ts_ms=BASE_MS + 42, depth=5)
    store.append("orderbook", df)
    with MarketDatabase(store) as db:
        latency = db.query("SELECT DISTINCT recv_ts_ms - ts_ms AS lat FROM orderbook")
    assert latency["lat"].tolist() == [42]


def test_views_expose_a_derived_ts_column(store: ParquetStore) -> None:
    store.upsert("candles", candles(count=3))
    with MarketDatabase(store) as db:
        ts = db.query("SELECT ts FROM candles ORDER BY ts_ms LIMIT 1").iloc[0]["ts"]
    assert pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None) == pd.Timestamp("2026-01-01")


def test_empty_store_has_no_views_and_a_zeroed_summary(store: ParquetStore) -> None:
    with MarketDatabase(store) as db:
        summary = db.table_summary()
        assert (summary["rows"] == 0).all()
        assert db.last_candle_ts("BTC", "1m") is None
        assert db.last_funding_ts("BTC") is None


def test_resume_pointers_report_the_newest_row(store: ParquetStore) -> None:
    store.upsert("candles", candles(count=7))
    store.upsert("funding", parse_funding_history(make_raw_funding(count=4)))
    with MarketDatabase(store) as db:
        assert db.last_candle_ts("BTC", "1m") == BASE_MS + 6 * STEP
        assert db.last_candle_ts("ETH", "1m") is None
        assert db.last_funding_ts("BTC") == BASE_MS + 3 * 3_600_000


def test_atomic_write_leaves_no_temp_files(store: ParquetStore) -> None:
    store.upsert("candles", candles(count=4))
    assert not list(store.dataset_dir("candles").rglob("*.tmp"))


def test_writing_an_empty_frame_is_a_no_op(store: ParquetStore) -> None:
    assert store.upsert("candles", candles(count=0)) == 0
    assert store.append("trades", parse_trades([])) == 0
    assert not store.has_data("candles")
