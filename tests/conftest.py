"""Shared fixtures. Nothing here touches the network."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import INTERVAL_MS, Paths  # noqa: E402
from data.database import ParquetStore  # noqa: E402

#: A fixed, aligned point in time so tests never depend on the wall clock.
#: 2026-01-01T00:00:00Z
BASE_MS = 1_767_225_600_000


def make_raw_candles(
    coin: str = "BTC",
    interval: str = "1m",
    count: int = 10,
    start_ms: int = BASE_MS,
    *,
    start_price: float = 100_000.0,
    drift: float = 10.0,
    skip: set[int] = frozenset(),
) -> list[dict]:
    """Build a Hyperliquid-shaped candleSnapshot payload.

    Values are JSON strings, exactly as the API returns them. ``skip`` holds
    bar indices to omit, which is how gap tests create holes.
    """
    step = INTERVAL_MS[interval]
    out = []
    for i in range(count):
        if i in skip:
            continue
        open_ = start_price + drift * i
        close = open_ + drift
        out.append(
            {
                "t": start_ms + i * step,
                "T": start_ms + (i + 1) * step - 1,
                "s": coin,
                "i": interval,
                "o": f"{open_:.1f}",
                "c": f"{close:.1f}",
                "h": f"{max(open_, close) + 5:.1f}",
                "l": f"{min(open_, close) - 5:.1f}",
                "v": f"{1.5 + i * 0.1:.5f}",
                "n": 40 + i,
            }
        )
    return out


def make_raw_funding(coin: str = "BTC", count: int = 5, start_ms: int = BASE_MS) -> list[dict]:
    return [
        {
            "coin": coin,
            "fundingRate": f"{0.0000125 + i * 1e-7:.10f}",
            "premium": f"{0.00001 * (i + 1):.8f}",
            "time": start_ms + i * 3_600_000,
        }
        for i in range(count)
    ]


def make_raw_l2_book(coin: str = "BTC", ts_ms: int = BASE_MS, depth: int = 3) -> dict:
    bids = [{"px": f"{100_000 - i:.1f}", "sz": f"{1.0 + i:.4f}", "n": 2 + i} for i in range(depth)]
    asks = [{"px": f"{100_001 + i:.1f}", "sz": f"{0.5 + i:.4f}", "n": 1 + i} for i in range(depth)]
    return {"coin": coin, "time": ts_ms, "levels": [bids, asks]}


def make_raw_meta_and_ctxs(coins: list[str] | None = None) -> list:
    coins = coins or ["BTC", "ETH", "SOL"]
    universe = [{"name": c, "szDecimals": 5, "maxLeverage": 50} for c in coins]
    ctxs = [
        {
            "funding": "0.0000125",
            "openInterest": f"{1000 + i * 10}.0",
            "prevDayPx": f"{99_000 + i}.0",
            "dayNtlVlm": f"{1e9 + i}",
            "premium": "0.00001",
            "oraclePx": f"{100_000 + i}.0",
            "markPx": f"{100_001 + i}.0",
            "midPx": f"{100_000.5 + i}",
            "impactPxs": [f"{100_000 + i}.0", f"{100_002 + i}.0"],
        }
        for i, _ in enumerate(coins)
    ]
    return [{"universe": universe}, ctxs]


def make_raw_trades(coin: str = "BTC", count: int = 3, start_ms: int = BASE_MS) -> list[dict]:
    return [
        {
            "coin": coin,
            "side": "B" if i % 2 == 0 else "A",
            "px": f"{100_000 + i:.1f}",
            "sz": f"{0.01 * (i + 1):.4f}",
            "time": start_ms + i * 250,
            "hash": f"0x{i:064x}",
            "tid": 9_000_000 + i,
        }
        for i in range(count)
    ]


@pytest.fixture
def store(tmp_path: Path) -> ParquetStore:
    """A ParquetStore rooted in a throwaway directory."""
    paths = Paths(
        root=tmp_path,
        storage=tmp_path / "storage",
        parquet=tmp_path / "storage" / "parquet",
        duckdb_file=tmp_path / "storage" / "db" / "market.duckdb",
        logs=tmp_path / "logs",
        models=tmp_path / "models",
    )
    return ParquetStore(paths)


# --------------------------------------------------------------------------
# Phase 2 helpers: synthetic bars and cost configurations
# --------------------------------------------------------------------------

def bars_from_prices(
    prices,
    *,
    coin: str = "BTC",
    interval: str = "1h",
    start_ms: int = BASE_MS,
    volume: float = 1_000_000.0,
    funding_rate: float = 0.0,
    range_fraction: float = 0.0,
):
    """Build a bar frame from a list of prices.

    ``range_fraction`` controls the high/low spread around each price; the
    default of zero makes bars frictionless so arithmetic checks have an
    exact expected answer. Open == close == price, so an order filling at
    the open fills at a price the test knows exactly.
    """
    import pandas as pd

    step = INTERVAL_MS[interval]
    rows = []
    for i, price in enumerate(prices):
        half = price * range_fraction / 2.0
        rows.append(
            {
                "ts_ms": start_ms + i * step,
                "coin": coin,
                "interval": interval,
                "open": float(price),
                "high": float(price) + half,
                "low": float(price) - half,
                "close": float(price),
                "volume": float(volume),
                "funding_rate": float(funding_rate),
            }
        )
    return pd.DataFrame(rows)


def frictionless_config():
    """An ExecutionConfig with every cost switched off.

    Used to isolate the accounting from the cost model: if buy-and-hold does
    not reproduce ``size * (exit - entry)`` exactly here, the bookkeeping is
    broken, and no amount of cost tuning would hide it.
    """
    from config.settings import ExecutionConfig

    return ExecutionConfig(
        taker_fee=0.0,
        maker_fee=0.0,
        default_half_spread=0.0,
        impact_coefficient=0.0,
        latency_ms=0,
        max_bar_volume_share=1.0,
        liquidation_penalty=0.0,
    )


def realistic_config(**overrides):
    """Hyperliquid base-tier costs, with room for per-test overrides."""
    from config.settings import ExecutionConfig

    defaults = dict(
        taker_fee=0.00045,
        maker_fee=0.00015,
        default_half_spread=0.00005,
        impact_coefficient=0.10,
        latency_ms=250,
        max_bar_volume_share=0.10,
        liquidation_penalty=0.01,
    )
    defaults.update(overrides)
    return ExecutionConfig(**defaults)


# --------------------------------------------------------------------------
# Phase 3 helpers: realistic-looking synthetic price paths
# --------------------------------------------------------------------------

def synthetic_bars(
    n: int = 1500,
    *,
    coin: str = "BTC",
    interval: str = "1h",
    start_ms: int = BASE_MS,
    start_price: float = 100.0,
    seed: int = 7,
    drift: float = 0.0002,
    vol: float = 0.01,
    funding_rate: float | None = 0.00005,
    volume_scale: float = 100.0,
):
    """A deterministic random walk with volatility clustering and OHLC.

    Deterministic on ``seed`` so feature tests are reproducible, but varied
    enough that rolling statistics are not degenerate.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    step = INTERVAL_MS[interval]

    # GARCH-ish clustering so volatility features have something to detect.
    shocks = rng.standard_normal(n)
    sigma = np.empty(n)
    sigma[0] = vol
    for i in range(1, n):
        sigma[i] = np.sqrt(0.00001 + 0.08 * (sigma[i - 1] * shocks[i - 1]) ** 2 + 0.90 * sigma[i - 1] ** 2)

    returns = drift + sigma * shocks
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[start_price], close[:-1]])
    wiggle = np.abs(rng.standard_normal(n)) * sigma * close
    high = np.maximum(open_, close) + wiggle
    low = np.minimum(open_, close) - wiggle
    # ``volume_scale`` exists because positions are now sized against the
    # market's traded notional. The default is a thin market on purpose --
    # that is the realistic case and the one that bites -- so a test about
    # something else must ask for a deep one explicitly.
    volume = np.abs(rng.lognormal(mean=3.0, sigma=0.6, size=n)) * volume_scale

    frame = pd.DataFrame(
        {
            "ts_ms": [start_ms + i * step for i in range(n)],
            "coin": coin,
            "interval": interval,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "trades": rng.integers(20, 500, size=n),
        }
    )
    if funding_rate is not None:
        frame["funding_rate"] = funding_rate + rng.standard_normal(n) * 1e-5
    return frame


def synthetic_universe(n: int = 1500, coins=("BTC", "ETH", "SOL"), **kwargs):
    """Several correlated-ish coins on a shared timestamp grid."""
    return {
        coin: synthetic_bars(n, coin=coin, seed=11 + i, start_price=100.0 * (i + 1), **kwargs)
        for i, coin in enumerate(coins)
    }


def synthetic_book(bars, *, snapshots_per_bar: int = 6, depth: int = 20, seed: int = 3):
    """Order-book snapshots spread across each bar's window."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    step = INTERVAL_MS[bars["interval"].iloc[0]]
    gap = step // snapshots_per_bar
    rows = []
    for _, bar in bars.iterrows():
        for k in range(snapshots_per_bar):
            ts = int(bar["ts_ms"]) + k * gap
            mid = float(bar["close"])
            spread = mid * 0.0001
            # Bid-heavy by construction so imbalance tests have a known sign.
            lean = rng.uniform(1.3, 2.5)
            for level in range(depth):
                rows.append({
                    "ts_ms": ts, "recv_ts_ms": ts + 30, "coin": bar["coin"],
                    "side": "bid", "level": level,
                    "px": mid - spread / 2 - level * spread,
                    "sz": lean * (1.0 + level * 0.3), "n_orders": 2 + level,
                })
                rows.append({
                    "ts_ms": ts, "recv_ts_ms": ts + 30, "coin": bar["coin"],
                    "side": "ask", "level": level,
                    "px": mid + spread / 2 + level * spread,
                    "sz": (1.0 + level * 0.3), "n_orders": 2 + level,
                })
    return pd.DataFrame(rows)
