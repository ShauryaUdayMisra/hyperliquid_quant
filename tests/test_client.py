"""Transport behaviour: retries, rate limiting, and honest TLS errors."""

from __future__ import annotations

import httpx
import pytest

from conftest import make_raw_candles, make_raw_meta_and_ctxs
from config.settings import HyperliquidConfig
from data.hyperliquid_client import (
    HyperliquidAPIError,
    HyperliquidInfoClient,
    TLSInterceptionError,
    _TokenBucket,
)

FAST = HyperliquidConfig(max_retries=3, backoff_base_s=0.0, backoff_max_s=0.0)


def client_with(handler) -> HyperliquidInfoClient:
    transport = httpx.MockTransport(handler)
    return HyperliquidInfoClient(FAST, client=httpx.Client(transport=transport))


def test_meta_round_trips() -> None:
    payload = make_raw_meta_and_ctxs()[0]
    with client_with(lambda r: httpx.Response(200, json=payload)) as client:
        assert [a["name"] for a in client.meta()["universe"]] == ["BTC", "ETH", "SOL"]


def test_request_body_matches_the_api_contract() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json=make_raw_candles(count=1))

    with client_with(handler) as client:
        client.candle_snapshot("BTC", "1m", 1_000, 2_000)

    assert seen["type"] == "candleSnapshot"
    assert seen["req"] == {"coin": "BTC", "interval": "1m", "startTime": 1_000, "endTime": 2_000}


def test_server_error_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json={"universe": []})

    with client_with(handler) as client:
        assert client.meta() == {"universe": []}
    assert calls["n"] == 3


def test_rate_limit_response_is_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429 if calls["n"] == 1 else 200, json={"ok": True})

    with client_with(handler) as client:
        assert client.all_mids() == {"ok": True}
    assert calls["n"] == 2


def test_retries_are_bounded_and_then_raise() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="boom")

    with client_with(handler) as client:
        with pytest.raises(HyperliquidAPIError, match="after 3 attempts"):
            client.meta()
    assert calls["n"] == 3


def test_client_error_is_not_retried() -> None:
    """A 400 means our request is wrong; repeating it just wastes budget."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad coin")

    with client_with(handler) as client:
        with pytest.raises(HyperliquidAPIError, match="HTTP 400"):
            client.l2_book("NOPE")
    assert calls["n"] == 1


def test_tls_interception_produces_an_actionable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] self-signed certificate")

    with client_with(handler) as client:
        with pytest.raises(TLSInterceptionError, match="HL_CA_BUNDLE"):
            client.meta()


def test_malformed_json_is_retried_not_swallowed() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, text="<html>proxy error</html>")
        return httpx.Response(200, json={"universe": []})

    with client_with(handler) as client:
        assert client.meta() == {"universe": []}
    assert calls["n"] == 2


# -- rate limiter ----------------------------------------------------------

def test_token_bucket_allows_a_burst_then_throttles() -> None:
    bucket = _TokenBucket(capacity=5, refill_per_second=10.0)
    assert all(bucket._take(1.0) == 0.0 for _ in range(5))
    wait = bucket._take(1.0)
    assert wait == pytest.approx(0.1, abs=0.02)


def test_token_bucket_refills_over_time() -> None:
    bucket = _TokenBucket(capacity=2, refill_per_second=100.0)
    bucket._take(2.0)
    import time

    time.sleep(0.05)
    assert bucket._take(1.0) == 0.0
