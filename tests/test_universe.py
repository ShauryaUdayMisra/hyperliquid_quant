"""Choosing which markets to trade.

MARKETS grew from a hand-written list of three coins into something that can
ask the exchange for its most liquid perps, or for all of them.
"""

from __future__ import annotations

import pytest

from data.universe import EXCLUDED, explicit_markets, parse_spec, rank_by_volume, resolve


def meta(*names):
    return {"universe": [{"name": n, "maxLeverage": 10} for n in names]}


def ctxs(*volumes):
    return [{"dayNtlVlm": str(v)} for v in volumes]


def test_a_plain_list_is_left_alone() -> None:
    assert parse_spec("BTC,ETH,SOL") == ("list", None)
    assert explicit_markets("BTC, eth ,Sol") == ("BTC", "ETH", "SOL")


def test_top_n_and_all_are_recognised() -> None:
    assert parse_spec("top:25") == ("top", 25)
    assert parse_spec("TOP:5") == ("top", 5)
    assert parse_spec("all") == ("all", None)


def test_a_nonsense_count_is_refused_rather_than_guessed() -> None:
    for bad in ("top:", "top:0", "top:-3", "top:many"):
        with pytest.raises(ValueError):
            parse_spec(bad)


def test_markets_rank_by_traded_volume() -> None:
    ranked = rank_by_volume(meta("A", "B", "C"), ctxs(1_000, 9_000, 5_000))
    assert ranked == ["B", "C", "A"]


def test_delisted_and_untraded_markets_are_dropped() -> None:
    """A delisted perp still appears in the universe but its candles stop."""
    m = meta("LIVE", "DEAD", "QUIET")
    m["universe"][1]["isDelisted"] = True
    assert rank_by_volume(m, ctxs(100, 900, 0)) == ["LIVE"]


def test_the_exchanges_own_casing_is_preserved() -> None:
    """Hyperliquid lists "kPEPE", not "KPEPE".

    Upper-casing the name produced a market that every later API call failed
    to find: it could never be refreshed, never had bars, and took the whole
    trading cycle down with it.
    """
    ranked = rank_by_volume(meta("kPEPE", "BTC", "kBONK"), ctxs(9, 8, 7))
    assert ranked == ["kPEPE", "BTC", "kBONK"]


def test_stablecoins_are_never_traded() -> None:
    names = tuple(EXCLUDED) + ("BTC",)
    ranked = rank_by_volume(meta(*names), ctxs(*([9_000] * len(EXCLUDED)), 1))
    assert ranked == ["BTC"]

    # ...whatever case they are listed in.
    assert rank_by_volume(meta("usdc", "BTC"), ctxs(9_000, 1)) == ["BTC"]


def test_mismatched_universe_and_contexts_raise_rather_than_pair_up_blindly() -> None:
    with pytest.raises(ValueError):
        rank_by_volume(meta("A", "B"), ctxs(1))


class _Client:
    def __init__(self, payload=None, error=None):
        self.payload, self.error = payload, error

    def meta_and_asset_ctxs(self):
        if self.error:
            raise self.error
        return self.payload


def test_top_n_returns_the_n_most_traded() -> None:
    client = _Client([meta("A", "B", "C", "D"), ctxs(1, 4, 3, 2)])
    assert resolve("top:2", client) == ("B", "C")


def test_all_returns_every_live_market() -> None:
    client = _Client([meta("A", "B", "C"), ctxs(1, 4, 3)])
    assert resolve("all", client) == ("B", "C", "A")


def test_an_unreachable_exchange_degrades_instead_of_aborting() -> None:
    """A smaller universe is useful; no universe is not."""
    client = _Client(error=OSError("network down"))
    assert resolve("top:25", client, fallback=["BTC", "ETH"]) == ("BTC", "ETH")


def test_a_literal_spec_needs_no_network_at_all() -> None:
    client = _Client(error=AssertionError("must not be called"))
    assert resolve("BTC,ETH", client) == ("BTC", "ETH")
