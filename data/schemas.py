"""Canonical on-disk schemas and the parsers that produce them.

Every dataset in this project obeys three rules:

1. Time is stored as ``ts_ms`` -- int64 UTC epoch milliseconds. A derived
   ``ts`` column (datetime64[ms, UTC]) exists for convenience but ``ts_ms``
   is the source of truth, because float/date round-tripping is a classic
   source of silent off-by-one-bar look-ahead.
2. Prices and sizes arrive from Hyperliquid as JSON *strings*. They are
   parsed to float64 exactly once, here, so no downstream module has to
   guess.
3. ``ts_ms`` for a bar is the bar's OPEN time, and a bar is only written
   once it has CLOSED. See :func:`parse_candles`.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

# --------------------------------------------------------------------------
# Column contracts
# --------------------------------------------------------------------------

CANDLE_COLUMNS: dict[str, str] = {
    "ts_ms": "int64",        # bar open time, UTC epoch ms
    "close_ts_ms": "int64",  # bar close time (inclusive end), UTC epoch ms
    "coin": "string",
    "interval": "string",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",     # base-asset volume
    "trades": "int64",       # number of trades in the bar
}

FUNDING_COLUMNS: dict[str, str] = {
    "ts_ms": "int64",
    "coin": "string",
    "funding_rate": "float64",  # per-hour rate
    "premium": "float64",
}

ASSET_CTX_COLUMNS: dict[str, str] = {
    "ts_ms": "int64",
    "coin": "string",
    "mark_px": "float64",
    "oracle_px": "float64",
    "mid_px": "float64",
    "funding": "float64",         # current hourly funding rate
    "open_interest": "float64",   # in base units
    "day_ntl_volume": "float64",  # 24h notional volume, USD
    "prev_day_px": "float64",
    "premium": "float64",
    "impact_bid_px": "float64",
    "impact_ask_px": "float64",
}

TRADE_COLUMNS: dict[str, str] = {
    "ts_ms": "int64",
    "coin": "string",
    "side": "string",   # "B" = aggressor bought, "A" = aggressor sold
    "px": "float64",
    "sz": "float64",
    "tid": "int64",
    "hash": "string",
}

#: Order-book snapshots are stored flat (one row per level) so that Parquet
#: stays columnar and DuckDB can aggregate without unnesting.
ORDERBOOK_COLUMNS: dict[str, str] = {
    "ts_ms": "int64",     # exchange-reported snapshot time
    "recv_ts_ms": "int64",  # local receipt time; recv - ts = observed latency
    "coin": "string",
    "side": "string",     # "bid" | "ask"
    "level": "int64",     # 0 = best
    "px": "float64",
    "sz": "float64",
    "n_orders": "int64",
}


class SchemaError(ValueError):
    """Raised when raw API data does not match the expected shape."""


def _empty(columns: Mapping[str, str]) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dt) for name, dt in columns.items()})


def coerce(df: pd.DataFrame, columns: Mapping[str, str]) -> pd.DataFrame:
    """Project ``df`` onto the contract: exact column set, order and dtypes."""
    missing = set(columns) - set(df.columns)
    if missing:
        raise SchemaError(f"missing columns: {sorted(missing)}")
    out = df.loc[:, list(columns)].copy()
    for name, dtype in columns.items():
        out[name] = out[name].astype(dtype)
    return out


def with_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Add the convenience ``ts`` datetime column derived from ``ts_ms``."""
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts_ms"], unit="ms", utc=True)
    return out


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------

def parse_candles(
    raw: Sequence[Mapping[str, Any]],
    *,
    now_ms: int | None = None,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Parse a ``candleSnapshot`` response.

    Hyperliquid includes the *currently forming* bar in its response. That
    bar's close is not yet its final close, so persisting it would inject
    look-ahead-shaped garbage into every feature computed from it. With
    ``drop_incomplete`` (the default) any bar whose close time has not yet
    passed is discarded.

    ``now_ms`` exists so tests can pin the clock; production leaves it None.
    """
    if not raw:
        return _empty(CANDLE_COLUMNS)

    rows = []
    for i, c in enumerate(raw):
        try:
            rows.append(
                {
                    "ts_ms": int(c["t"]),
                    "close_ts_ms": int(c["T"]),
                    "coin": str(c["s"]),
                    "interval": str(c["i"]),
                    "open": float(c["o"]),
                    "high": float(c["h"]),
                    "low": float(c["l"]),
                    "close": float(c["c"]),
                    "volume": float(c["v"]),
                    "trades": int(c["n"]),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"malformed candle at index {i}: {c!r}") from exc

    df = pd.DataFrame(rows)

    if drop_incomplete:
        cutoff = _now_ms() if now_ms is None else now_ms
        # A bar is complete once its (inclusive) close millisecond is in the
        # past. Hyperliquid reports T as open + interval - 1ms.
        df = df.loc[df["close_ts_ms"] < cutoff]

    df = df.sort_values("ts_ms").drop_duplicates(subset=["coin", "interval", "ts_ms"], keep="last")
    return coerce(df.reset_index(drop=True), CANDLE_COLUMNS)


def parse_funding_history(raw: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Parse a ``fundingHistory`` response."""
    if not raw:
        return _empty(FUNDING_COLUMNS)

    rows = []
    for i, f in enumerate(raw):
        try:
            rows.append(
                {
                    "ts_ms": int(f["time"]),
                    "coin": str(f["coin"]),
                    "funding_rate": float(f["fundingRate"]),
                    # ``premium`` is occasionally absent on very old records.
                    "premium": float(f["premium"]) if f.get("premium") is not None else float("nan"),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"malformed funding record at index {i}: {f!r}") from exc

    df = pd.DataFrame(rows).sort_values("ts_ms")
    df = df.drop_duplicates(subset=["coin", "ts_ms"], keep="last")
    return coerce(df.reset_index(drop=True), FUNDING_COLUMNS)


def _f(value: Any) -> float:
    """Tolerant float parse -- Hyperliquid omits fields for some assets."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_meta_and_asset_ctxs(
    raw: Sequence[Any],
    *,
    recv_ts_ms: int,
    coins: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Parse ``metaAndAssetCtxs`` into per-coin funding / OI / mark rows.

    The response is a two-element array: the universe metadata and a
    positionally-aligned array of asset contexts. Misalignment here would
    silently attribute BTC's funding to SOL, so the lengths are checked.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise SchemaError(f"metaAndAssetCtxs must be a 2-element array, got {type(raw)}")

    meta, ctxs = raw
    universe = meta.get("universe") if isinstance(meta, Mapping) else None
    if universe is None:
        raise SchemaError("metaAndAssetCtxs[0] has no 'universe'")
    if len(universe) != len(ctxs):
        raise SchemaError(
            f"universe/context length mismatch: {len(universe)} vs {len(ctxs)}"
        )

    wanted = {c.upper() for c in coins} if coins is not None else None
    rows = []
    for asset, ctx in zip(universe, ctxs):
        name = str(asset.get("name", ""))
        if wanted is not None and name.upper() not in wanted:
            continue
        impact = ctx.get("impactPxs") or [None, None]
        rows.append(
            {
                "ts_ms": int(recv_ts_ms),
                "coin": name,
                "mark_px": _f(ctx.get("markPx")),
                "oracle_px": _f(ctx.get("oraclePx")),
                "mid_px": _f(ctx.get("midPx")),
                "funding": _f(ctx.get("funding")),
                "open_interest": _f(ctx.get("openInterest")),
                "day_ntl_volume": _f(ctx.get("dayNtlVlm")),
                "prev_day_px": _f(ctx.get("prevDayPx")),
                "premium": _f(ctx.get("premium")),
                "impact_bid_px": _f(impact[0] if len(impact) > 0 else None),
                "impact_ask_px": _f(impact[1] if len(impact) > 1 else None),
            }
        )

    if not rows:
        return _empty(ASSET_CTX_COLUMNS)
    return coerce(pd.DataFrame(rows), ASSET_CTX_COLUMNS)


def parse_l2_book(raw: Mapping[str, Any], *, recv_ts_ms: int, depth: int) -> pd.DataFrame:
    """Flatten an ``l2Book`` snapshot into one row per price level."""
    try:
        coin = str(raw["coin"])
        exchange_ts = int(raw["time"])
        levels = raw["levels"]
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError(f"malformed l2Book payload: {raw!r}") from exc

    if not isinstance(levels, (list, tuple)) or len(levels) != 2:
        raise SchemaError("l2Book 'levels' must be [bids, asks]")

    rows = []
    for side, side_levels in (("bid", levels[0]), ("ask", levels[1])):
        for idx, lvl in enumerate(side_levels[:depth]):
            try:
                rows.append(
                    {
                        "ts_ms": exchange_ts,
                        "recv_ts_ms": int(recv_ts_ms),
                        "coin": coin,
                        "side": side,
                        "level": idx,
                        "px": float(lvl["px"]),
                        "sz": float(lvl["sz"]),
                        "n_orders": int(lvl.get("n", 0)),
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise SchemaError(f"malformed l2Book level {side}[{idx}]: {lvl!r}") from exc

    if not rows:
        return _empty(ORDERBOOK_COLUMNS)
    return coerce(pd.DataFrame(rows), ORDERBOOK_COLUMNS)


def parse_trades(raw: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Parse a WebSocket ``trades`` payload."""
    if not raw:
        return _empty(TRADE_COLUMNS)

    rows = []
    for i, t in enumerate(raw):
        try:
            rows.append(
                {
                    "ts_ms": int(t["time"]),
                    "coin": str(t["coin"]),
                    "side": str(t["side"]),
                    "px": float(t["px"]),
                    "sz": float(t["sz"]),
                    "tid": int(t.get("tid", 0)),
                    "hash": str(t.get("hash", "")),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"malformed trade at index {i}: {t!r}") from exc

    df = pd.DataFrame(rows).sort_values("ts_ms")
    return coerce(df.reset_index(drop=True), TRADE_COLUMNS)


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)
