"""In-memory stand-in for the Hyperliquid API, used by the downloader tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from conftest import BASE_MS, make_raw_candles, make_raw_funding
from config.settings import FUNDING_INTERVAL_MS, INTERVAL_MS


@dataclass
class FakeInfoClient:
    """Serves a fixed synthetic history with the real server's page cap.

    ``page_limit`` mirrors Hyperliquid's 5000-row cap so pagination logic is
    genuinely exercised rather than short-circuited by a small fixture.
    """

    coin: str = "BTC"
    interval: str = "1m"
    bars: int = 12_000
    start_ms: int = BASE_MS
    page_limit: int = 5_000
    missing: set[int] = field(default_factory=set)
    calls: list[tuple] = field(default_factory=list)
    closed: bool = False

    def __post_init__(self) -> None:
        self._candles = {
            c["t"]: c
            for c in make_raw_candles(
                self.coin, self.interval, self.bars, self.start_ms, skip=self.missing
            )
        }
        self._funding = {
            f["time"]: f
            for f in make_raw_funding(self.coin, self.bars // 60 + 1, self.start_ms)
        }

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.bars * INTERVAL_MS[self.interval]

    def candle_snapshot(self, coin, interval, start_ms, end_ms):
        self.calls.append(("candles", coin, interval, start_ms, end_ms))
        if coin != self.coin or interval != self.interval:
            return []
        rows = [c for t, c in sorted(self._candles.items()) if start_ms <= t <= end_ms]
        return rows[: self.page_limit]

    def funding_history(self, coin, start_ms, end_ms=None):
        self.calls.append(("funding", coin, start_ms, end_ms))
        if coin != self.coin:
            return []
        rows = [
            f
            for t, f in sorted(self._funding.items())
            if start_ms <= t and (end_ms is None or t <= end_ms)
        ]
        return rows[:500]

    def close(self) -> None:
        self.closed = True
