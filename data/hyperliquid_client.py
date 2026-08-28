"""Read-only client for Hyperliquid's public market-data API.

Deliberately minimal: this module knows how to ask the public ``/info``
endpoint questions and how to hold a WebSocket subscription open. It has no
concept of an account, an order, or a signature, and it never will.

Rate limiting: Hyperliquid budgets ``/info`` by request weight (1200 per
minute per IP). A token bucket keeps us under a configured ceiling so that a
backfill loop cannot get the machine throttled or banned.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Mapping

import httpx

from config.settings import SETTINGS, HyperliquidConfig

log = logging.getLogger(__name__)


class HyperliquidAPIError(RuntimeError):
    """A request to Hyperliquid failed after exhausting retries."""


class TLSInterceptionError(HyperliquidAPIError):
    """TLS verification failed, almost certainly a network doing SSL inspection."""

    HINT = (
        "TLS verification failed for the Hyperliquid API. This usually means a "
        "firewall/proxy on your network is intercepting HTTPS with its own CA. "
        "Either move to an uninspected network, or point HL_CA_BUNDLE in .env at "
        "a PEM bundle that includes your organisation's root CA. See README.md."
    )


@dataclass
class _TokenBucket:
    """Simple monotonic-clock token bucket, shared by sync and async paths."""

    capacity: int
    refill_per_second: float
    _tokens: float = 0.0
    _last: float = 0.0

    def __post_init__(self) -> None:
        self._tokens = float(self.capacity)
        self._last = time.monotonic()

    def _take(self, cost: float) -> float:
        """Consume ``cost`` tokens, returning the seconds a caller must wait."""
        now = time.monotonic()
        self._tokens = min(
            self.capacity, self._tokens + (now - self._last) * self.refill_per_second
        )
        self._last = now
        if self._tokens >= cost:
            self._tokens -= cost
            return 0.0
        deficit = cost - self._tokens
        self._tokens = 0.0
        return deficit / self.refill_per_second

    def acquire(self, cost: float = 1.0) -> None:
        wait = self._take(cost)
        if wait > 0:
            time.sleep(wait)

    async def acquire_async(self, cost: float = 1.0) -> None:
        wait = self._take(cost)
        if wait > 0:
            await asyncio.sleep(wait)


def _is_tls_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "certificate" in text or "ssl" in text or "tlsv" in text


class HyperliquidInfoClient:
    """Synchronous client for the public ``/info`` POST endpoint."""

    def __init__(
        self,
        config: HyperliquidConfig | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config or SETTINGS.hyperliquid
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=self.config.request_timeout_s,
            verify=self.config.verify(),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "hyperliquid-quant-research/0.1 (read-only paper trading)",
            },
        )
        self._bucket = _TokenBucket(
            capacity=max(1, self.config.rate_limit_per_minute // 6),
            refill_per_second=self.config.rate_limit_per_minute / 60.0,
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "HyperliquidInfoClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- transport ---------------------------------------------------------

    def _post(self, payload: Mapping[str, Any], *, weight: float = 1.0) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries):
            self._bucket.acquire(weight)
            try:
                response = self._client.post(self.config.info_url, json=payload)
            except httpx.HTTPError as exc:
                if _is_tls_error(exc):
                    raise TLSInterceptionError(
                        f"{TLSInterceptionError.HINT}\n\nUnderlying error: {exc}"
                    ) from exc
                last_exc = exc
                log.warning("request error (attempt %d): %s", attempt + 1, exc)
            else:
                if response.status_code == 429 or response.status_code >= 500:
                    last_exc = HyperliquidAPIError(
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                    log.warning("retryable status %s (attempt %d)", response.status_code, attempt + 1)
                elif response.status_code >= 400:
                    # 4xx other than 429 means we asked a malformed question;
                    # retrying would just repeat the mistake.
                    raise HyperliquidAPIError(
                        f"HTTP {response.status_code} for {payload.get('type')}: {response.text[:300]}"
                    )
                else:
                    try:
                        return response.json()
                    except json.JSONDecodeError as exc:
                        last_exc = exc
                        log.warning("non-JSON response (attempt %d)", attempt + 1)

            self._sleep_backoff(attempt)

        raise HyperliquidAPIError(
            f"'{payload.get('type')}' failed after {self.config.max_retries} attempts"
        ) from last_exc

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(self.config.backoff_base_s * (2 ** attempt), self.config.backoff_max_s)
        # Full jitter: avoids a fleet of retries marching in lockstep.
        time.sleep(random.uniform(0, delay))

    # -- endpoints ---------------------------------------------------------

    def meta(self) -> dict[str, Any]:
        """Perpetuals universe: names, size decimals, max leverage."""
        return self._post({"type": "meta"}, weight=2.0)

    def meta_and_asset_ctxs(self) -> list[Any]:
        """Universe plus live per-asset context (funding, OI, mark, oracle)."""
        return self._post({"type": "metaAndAssetCtxs"}, weight=2.0)

    def all_mids(self) -> dict[str, str]:
        """Current mid price for every asset."""
        return self._post({"type": "allMids"}, weight=1.0)

    def l2_book(self, coin: str, *, n_sig_figs: int | None = None) -> dict[str, Any]:
        """Current L2 order book for one coin."""
        payload: dict[str, Any] = {"type": "l2Book", "coin": coin}
        if n_sig_figs is not None:
            payload["nSigFigs"] = n_sig_figs
        return self._post(payload, weight=1.0)

    def candle_snapshot(
        self, coin: str, interval: str, start_ms: int, end_ms: int
    ) -> list[dict[str, Any]]:
        """Raw candles in ``[start_ms, end_ms]``. Server caps the page at 5000."""
        return self._post(
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": int(start_ms),
                    "endTime": int(end_ms),
                },
            },
            weight=2.0,
        )

    def funding_history(
        self, coin: str, start_ms: int, end_ms: int | None = None
    ) -> list[dict[str, Any]]:
        """Hourly funding rates in ``[start_ms, end_ms]``."""
        payload: dict[str, Any] = {
            "type": "fundingHistory",
            "coin": coin,
            "startTime": int(start_ms),
        }
        if end_ms is not None:
            payload["endTime"] = int(end_ms)
        return self._post(payload, weight=2.0)

    def ping(self) -> bool:
        """Cheap reachability probe used by ``main.py status``."""
        self.meta()
        return True


class HyperliquidWebSocket:
    """Async WebSocket subscriber with automatic reconnect.

    Yields ``(channel, data)`` pairs. Reconnection re-sends every
    subscription, and the caller is told about the gap via a synthetic
    ``("_reconnected", {...})`` message so it can record the outage rather
    than pretend the stream was continuous.
    """

    def __init__(
        self,
        subscriptions: Iterable[Mapping[str, Any]],
        config: HyperliquidConfig | None = None,
    ) -> None:
        self.config = config or SETTINGS.hyperliquid
        self.subscriptions = [dict(s) for s in subscriptions]
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def stream(self) -> AsyncIterator[tuple[str, Any]]:
        import ssl

        import websockets

        ssl_context = None
        if self.config.ca_bundle:
            ssl_context = ssl.create_default_context(cafile=self.config.ca_bundle)

        attempt = 0
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self.config.ws_url,
                    ssl=ssl_context,
                    ping_interval=self.config.ws_ping_interval_s,
                    ping_timeout=self.config.ws_ping_interval_s,
                    max_queue=4096,
                ) as ws:
                    for sub in self.subscriptions:
                        await ws.send(json.dumps({"method": "subscribe", "subscription": sub}))
                    if attempt:
                        yield "_reconnected", {"attempt": attempt, "ts_ms": int(time.time() * 1000)}
                    attempt = 0
                    log.info("websocket connected with %d subscriptions", len(self.subscriptions))

                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            log.warning("dropping non-JSON websocket frame")
                            continue
                        channel = msg.get("channel")
                        if channel in (None, "subscriptionResponse", "pong"):
                            continue
                        yield channel, msg.get("data")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on anything
                if _is_tls_error(exc):
                    raise TLSInterceptionError(
                        f"{TLSInterceptionError.HINT}\n\nUnderlying error: {exc}"
                    ) from exc
                attempt += 1
                delay = min(
                    self.config.backoff_base_s * (2 ** min(attempt, 8)),
                    self.config.backoff_max_s,
                )
                log.warning("websocket dropped (%s); reconnecting in %.1fs", exc, delay)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
