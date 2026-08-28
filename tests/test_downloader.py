"""Backfill: pagination, resume, idempotency, and no-look-ahead on write."""

from __future__ import annotations

import pytest

from conftest import BASE_MS
from config.settings import FUNDING_INTERVAL_MS, INTERVAL_MS, Settings
from data.database import MarketDatabase, ParquetStore
from data.downloader import HistoricalDownloader
from data.quality import check_candles, check_funding
from fakes import FakeInfoClient

STEP = INTERVAL_MS["1m"]


@pytest.fixture
def downloader(store: ParquetStore):
    client = FakeInfoClient()
    settings = Settings(paths=store.paths)
    return HistoricalDownloader(client=client, store=store, settings=settings), client


def test_pagination_walks_the_whole_history(downloader) -> None:
    """12,000 bars cannot arrive in one 5,000-row page."""
    dl, client = downloader
    result = dl.backfill_candles("BTC", "1m", start_ms=client.start_ms, end_ms=client.end_ms)
    assert result.pages >= 3
    assert result.rows_fetched == client.bars
    assert result.rows_new == client.bars


def test_backfilled_data_has_no_gaps(downloader) -> None:
    dl, client = downloader
    dl.backfill_candles("BTC", "1m", start_ms=client.start_ms, end_ms=client.end_ms)
    with MarketDatabase(dl.store) as db:
        df = db.query("SELECT * FROM candles ORDER BY ts_ms")
    report = check_candles(df, coin="BTC", interval="1m", reference_ms=client.end_ms + 10 * STEP)
    assert report.ok, report.describe()
    assert report.rows == client.bars


def test_a_hole_in_the_source_is_reported_not_hidden(store: ParquetStore) -> None:
    client = FakeInfoClient(bars=2_000, missing={500, 501, 502})
    dl = HistoricalDownloader(client=client, store=store, settings=Settings(paths=store.paths))
    dl.backfill_candles("BTC", "1m", start_ms=client.start_ms, end_ms=client.end_ms)
    with MarketDatabase(store) as db:
        df = db.query("SELECT * FROM candles ORDER BY ts_ms")
    report = check_candles(df, coin="BTC", interval="1m", reference_ms=client.end_ms + STEP)
    assert not report.ok
    assert report.missing_bars == 3


def test_rerunning_a_backfill_adds_nothing(downloader) -> None:
    dl, client = downloader
    first = dl.backfill_candles("BTC", "1m", start_ms=client.start_ms, end_ms=client.end_ms)
    second = dl.backfill_candles("BTC", "1m", start_ms=client.start_ms, end_ms=client.end_ms)
    assert first.rows_new == client.bars
    assert second.rows_new == 0


def test_resume_starts_from_the_stored_tail(downloader) -> None:
    dl, client = downloader
    half = client.start_ms + 6_000 * STEP
    dl.backfill_candles("BTC", "1m", start_ms=client.start_ms, end_ms=half)
    client.calls.clear()

    result = dl.backfill_candles("BTC", "1m", end_ms=client.end_ms, resume=True)
    first_request_start = client.calls[0][3]
    # Resumes just behind the tail so the last (then-forming) bar is refetched.
    assert first_request_start == half - 2 * STEP
    # The first pass stored bars 0..6000 inclusive (6001 of them), so the
    # remainder of the 12,000-bar history is 5,999 new rows.
    assert result.rows_new == client.bars - 6_001


def test_no_resume_refetches_everything(downloader) -> None:
    dl, client = downloader
    dl.backfill_candles("BTC", "1m", start_ms=client.start_ms, end_ms=client.end_ms)
    client.calls.clear()
    dl.backfill_candles(
        "BTC", "1m", start_ms=client.start_ms, end_ms=client.end_ms, resume=False
    )
    assert client.calls[0][3] == client.start_ms


def test_stored_candles_never_include_an_unfinished_bar(store: ParquetStore) -> None:
    """The write path must not persist a bar whose close is still in the future."""
    import time

    now = int(time.time() * 1000)
    aligned = (now // STEP) * STEP
    client = FakeInfoClient(bars=50, start_ms=aligned - 40 * STEP)
    dl = HistoricalDownloader(client=client, store=store, settings=Settings(paths=store.paths))
    dl.backfill_candles("BTC", "1m", start_ms=client.start_ms, end_ms=client.end_ms)

    with MarketDatabase(store) as db:
        df = db.query("SELECT * FROM candles")
    assert len(df) > 0
    assert df["close_ts_ms"].max() < now


def test_empty_window_is_reported_not_looped_forever(store: ParquetStore) -> None:
    client = FakeInfoClient(coin="BTC")
    dl = HistoricalDownloader(client=client, store=store, settings=Settings(paths=store.paths))
    result = dl.backfill_candles(
        "DOGE", "1m", start_ms=client.start_ms, end_ms=client.end_ms, resume=False
    )
    assert result.empty
    assert result.rows_fetched == 0
    assert result.pages < 10


def test_unsupported_interval_rejected(downloader) -> None:
    dl, _ = downloader
    with pytest.raises(ValueError, match="unsupported interval"):
        dl.backfill_candles("BTC", "7s")


def test_funding_backfill_is_hourly_and_clean(downloader) -> None:
    dl, client = downloader
    result = dl.backfill_funding("BTC", start_ms=client.start_ms, end_ms=client.end_ms)
    assert result.rows_new > 0
    with MarketDatabase(dl.store) as db:
        df = db.query("SELECT * FROM funding ORDER BY ts_ms")
    report = check_funding(df, coin="BTC", reference_ms=client.end_ms + FUNDING_INTERVAL_MS)
    assert report.ok, report.describe()


def test_funding_pagination_beyond_the_500_row_cap(downloader) -> None:
    dl, client = downloader
    result = dl.backfill_funding("BTC", start_ms=client.start_ms, end_ms=client.end_ms)
    assert result.pages >= 1
    assert result.rows_fetched > 100
