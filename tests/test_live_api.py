"""Smoke tests against the real Hyperliquid API.

Deselected by default. Run explicitly with:

    pytest -m live

These are the tests that prove data really flows; everything else in the
suite runs offline against fixtures.
"""

from __future__ import annotations

import time

import pytest

from config.settings import SETTINGS
from data.hyperliquid_client import HyperliquidInfoClient
from data.quality import check_candles, find_future_timestamps
from data.schemas import parse_candles, parse_l2_book, parse_meta_and_asset_ctxs

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def client():
    with HyperliquidInfoClient() as c:
        yield c


def test_configured_markets_are_listed(client) -> None:
    universe = {a["name"] for a in client.meta()["universe"]}
    assert set(SETTINGS.data.markets) <= universe


def test_live_candles_are_clean_and_not_from_the_future(client) -> None:
    now = int(time.time() * 1000)
    raw = client.candle_snapshot("BTC", "1m", now - 6 * 3_600_000, now)
    df = parse_candles(raw, now_ms=now)
    assert len(df) > 100
    assert find_future_timestamps(
        df, reference_ms=now, tolerance_ms=0, columns=("ts_ms", "close_ts_ms")
    ).empty
    report = check_candles(df, coin="BTC", interval="1m", reference_ms=now)
    assert report.ok, report.describe()


def test_live_order_book_is_crossed_correctly(client) -> None:
    now = int(time.time() * 1000)
    df = parse_l2_book(client.l2_book("BTC"), recv_ts_ms=now, depth=10)
    best_bid = df[(df["side"] == "bid") & (df["level"] == 0)]["px"].iloc[0]
    best_ask = df[(df["side"] == "ask") & (df["level"] == 0)]["px"].iloc[0]
    assert best_bid < best_ask


def test_live_asset_contexts_carry_funding_and_open_interest(client) -> None:
    now = int(time.time() * 1000)
    df = parse_meta_and_asset_ctxs(
        client.meta_and_asset_ctxs(), recv_ts_ms=now, coins=SETTINGS.data.markets
    )
    assert len(df) == len(SETTINGS.data.markets)
    assert (df["mark_px"] > 0).all()
    assert df["open_interest"].notna().all()
    assert df["funding"].abs().max() < 0.01
